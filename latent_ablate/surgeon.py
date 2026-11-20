import logging
import math
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from latent_ablate.arch import resolve_layers, hidden_size, eligible_linears
from latent_ablate.extractor import ActivationExtractor
from latent_ablate.refusal_subspace import RefusalSubspaceExtractor
from latent_ablate.subspace import SubspaceIsolator, refusal_direction
from latent_ablate.erasure import (
    conflict_score,
    conflict_scaled_alpha,
    select_energy_rank,
    build_capability_basis,
    oblique_weights,
    apply_erasure,
    apply_multi_direction_erasure,
    preserve_row_norms,
)

logger = logging.getLogger("LatentAblate.Surgeon")


def _is_meta_tensor(t) -> bool:
    if t is None:
        return False
    if getattr(t, "is_meta", False):
        return True
    dev = getattr(t, "device", None)
    return dev is not None and getattr(dev, "type", None) == "meta"


class LatentAblateSurgeon:
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        torch_dtype=torch.float16,
        load_in_4bit: bool = False,
        local_files_only: bool = False,
    ):
        logger.info(
            f"Loading tokenizer and model from {model_path} (4-bit quantization: {load_in_4bit})..."
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=local_files_only
        )
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": device,
            "trust_remote_code": True,
            "local_files_only": local_files_only,
        }
        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig

                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                )
            except ImportError:
                raise ImportError(
                    "bitsandbytes is required for 4-bit loading. Run `pip install bitsandbytes`."
                )

        self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.hidden = hidden_size(self.model)
        self.layers = resolve_layers(self.model)
        self.extractor = ActivationExtractor(self.model, self.tokenizer, self.device)
        self.conflict_table = {}
        self.reporter = None

    def _get_component_targets(self, layer, target_components: list):
        attn_keys = (
            "o_proj",
            "v_proj",
            "q_proj",
            "k_proj",
            "q_a_proj",
            "q_b_proj",
            "kv_a_proj",
            "kv_b_proj",
            "attention",
            "attn",
            "query",
            "key",
            "value",
        )
        mlp_keys = (
            "down_proj",
            "gate_proj",
            "up_proj",
            "dense",
            "c_proj",
            "w1",
            "w2",
            "w3",
            "ffn",
            "mlp",
            "fc1",
            "fc2",
            "experts",
            "shared_experts",
        )
        targets = []
        for name, W in eligible_linears(layer):
            if W.ndim != 2 or _is_meta_tensor(W):
                continue
            low = name.lower()
            if "router" in low or low.endswith(".gate") or ".gate." in low:
                continue
            is_attn = any(k in low for k in attn_keys)
            is_mlp = any(k in low for k in mlp_keys)

            if "attn" in target_components and is_attn:
                targets.append((name, W, "attn"))
            elif "mlp" in target_components and is_mlp:
                targets.append((name, W, "mlp"))
        return targets

    def execute_surgery(
        self,
        positive_prompts: list,
        negative_prompts: list,
        capability_prompts: list,
        target_layer_idx: int = 18,
        layer_window: int = 3,
        basis_rank: int = 16,
        scaling_factor: float = 1.15,
        target_components: list = None,
        preserve_capability: bool = True,
        calibrate_norms: bool = True,
        layer_curve: str = "cosine",
        all_layers: bool = False,
        subspace_rank: int = 1,
        rank_decay: float = 0.75,
        adaptive_rank: bool = False,
        target_energy: float = 0.85,
        conflict_scaling: bool = True,
        use_prefix_contrast: bool = True,
    ):
        effective_window = len(self.layers) if all_layers else layer_window
        passes = [
            {
                "name": f"Layer_{target_layer_idx}_Pass",
                "positive": positive_prompts,
                "negative": negative_prompts,
                "target_layer": target_layer_idx,
            }
        ]
        return self.execute_multi_pass_surgery(
            passes=passes,
            capability_prompts=capability_prompts,
            layer_window=effective_window,
            basis_rank_attn=basis_rank,
            basis_rank_mlp=max(4, basis_rank // 2),
            scaling_factor_attn=scaling_factor,
            scaling_factor_mlp=scaling_factor,
            target_components=target_components,
            preserve_capability=preserve_capability,
            calibrate_norms=calibrate_norms,
            layer_curve=layer_curve,
            subspace_rank=subspace_rank,
            rank_decay=rank_decay,
            adaptive_rank=adaptive_rank,
            target_energy=target_energy,
            conflict_scaling=conflict_scaling,
            use_prefix_contrast=use_prefix_contrast,
        )

    def execute_full_stack_surgery(
        self,
        positive_prompts: list,
        negative_prompts: list,
        capability_prompts: list,
        tier_specs: list[tuple[int, int, float]] | None = None,
        basis_rank_attn: int = 16,
        basis_rank_mlp: int = 12,
        target_components: list = None,
        preserve_capability: bool = True,
        calibrate_norms: bool = True,
        layer_curve: str = "cosine",
        subspace_rank: int = 1,
        rank_decay: float = 0.75,
        adaptive_rank: bool = False,
        target_energy: float = 0.85,
        conflict_scaling: bool = True,
        use_prefix_contrast: bool = True,
    ):
        """
        Stacks multiple layer-band passes from early-mid to late layers with gentle
        per-tier scaling so refusal directions across depths are covered cleanly.
        """
        n_layers = len(self.layers)
        if tier_specs is None:
            tier_specs = [
                (max(1, int(n_layers * 0.35)), max(2, n_layers // 8), 0.55),
                (max(1, int(n_layers * 0.55)), max(2, n_layers // 6), 0.85),
                (max(1, int(n_layers * 0.75)), max(2, n_layers // 8), 0.50),
            ]

        passes = []
        for idx, (center_layer, window, alpha) in enumerate(tier_specs):
            clamped = max(0, min(n_layers - 1, center_layer))
            passes.append({
                "name": f"StackTier_{idx+1}_L{clamped}",
                "positive": positive_prompts,
                "negative": negative_prompts,
                "target_layer": clamped,
                "layer_window": window,
                "alpha_attn": alpha,
                "alpha_mlp": alpha * 0.85,
            })

        return self.execute_multi_pass_surgery(
            passes=passes,
            capability_prompts=capability_prompts,
            layer_window=max(2, n_layers // 6),
            basis_rank_attn=basis_rank_attn,
            basis_rank_mlp=basis_rank_mlp,
            target_components=target_components,
            preserve_capability=preserve_capability,
            calibrate_norms=calibrate_norms,
            layer_curve=layer_curve,
            subspace_rank=subspace_rank,
            rank_decay=rank_decay,
            adaptive_rank=adaptive_rank,
            target_energy=target_energy,
            conflict_scaling=conflict_scaling,
            use_prefix_contrast=use_prefix_contrast,
        )

    def execute_multi_pass_surgery(
        self,
        passes: list[dict],
        capability_prompts: list,
        layer_window: int = 3,
        basis_rank_attn: int = 16,
        basis_rank_mlp: int = 12,
        scaling_factor_attn: float = 1.25,
        scaling_factor_mlp: float = 1.15,
        target_components: list = None,
        preserve_capability: bool = True,
        calibrate_norms: bool = True,
        layer_curve: str = "cosine",
        subspace_rank: int = 1,
        rank_decay: float = 0.75,
        adaptive_rank: bool = False,
        target_energy: float = 0.85,
        conflict_scaling: bool = True,
        use_prefix_contrast: bool = True,
    ):
        from latent_ablate.optimizer import compute_layer_weights

        if target_components is None:
            target_components = ["attn", "mlp"]

        modified_layers = set()
        base_subspace_rank = max(1, int(subspace_rank))

        for pass_idx, pass_data in enumerate(passes):
            pass_name = pass_data.get("name", f"Pass_{pass_idx+1}")
            pos_prompts = pass_data.get("positive", [])
            neg_prompts = pass_data.get("negative", [])
            target_layer_idx = pass_data.get("target_layer", len(self.layers) // 2)
            pass_window = pass_data.get("layer_window", layer_window)
            pass_alpha_attn = pass_data.get("alpha_attn", scaling_factor_attn)
            pass_alpha_mlp = pass_data.get("alpha_mlp", scaling_factor_mlp)

            logger.info(
                f"=== Starting Surgical Pass [{pass_idx+1}/{len(passes)}]: {pass_name} at Layer {target_layer_idx} ==="
            )

            start = max(0, target_layer_idx - pass_window)
            end = min(len(self.layers) - 1, target_layer_idx + pass_window)
            window_indices = list(range(start, end + 1))
            layer_multipliers = compute_layer_weights(window_indices, curve=layer_curve)

            for idx in tqdm(window_indices, desc=f"Pass '{pass_name}' Layers"):
                if idx in modified_layers:
                    continue
                modified_layers.add(idx)
                layer = self.layers[idx]
                targets = self._get_component_targets(layer, target_components)
                if not targets:
                    continue

                layer_device = next(layer.parameters()).device

                pos = self.extractor.extract_hidden_states(pos_prompts, idx).to(layer_device)
                neg = self.extractor.extract_hidden_states(neg_prompts, idx).to(layer_device)

                prefix_dir = None
                if use_prefix_contrast and pos_prompts:
                    prefix_dir = RefusalSubspaceExtractor.extract_primary_refusal_direction(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        device=layer_device,
                        prompts=pos_prompts[:8],
                        layer_idx=idx,
                    )

                r = (
                    prefix_dir.to(layer_device)
                    if prefix_dir is not None and prefix_dir.numel() > 0
                    else refusal_direction(pos, neg).to(layer_device)
                )

                effective_rank = (
                    select_energy_rank(pos, neg, max_rank=max(base_subspace_rank, 8), target_energy=target_energy)
                    if adaptive_rank
                    else base_subspace_rank
                )

                if effective_rank > 1 and pos.shape[0] == neg.shape[0] and pos.shape[0] >= 2:
                    subspace_vecs = SubspaceIsolator.compute_refusal_subspace(
                        pos, neg, rank=effective_rank
                    ).to(layer_device)
                else:
                    subspace_vecs = r.unsqueeze(0)

                cap = None
                if preserve_capability and capability_prompts and len(capability_prompts) >= 2:
                    cap = self.extractor.extract_hidden_states(capability_prompts, idx)

                if cap is not None and not _is_meta_tensor(cap):
                    cap = cap.to(layer_device)
                    cap_basis_attn = build_capability_basis(cap, rank=basis_rank_attn).to(layer_device)
                    cap_basis_mlp = build_capability_basis(cap, rank=basis_rank_mlp).to(layer_device)
                else:
                    cap_basis_attn = torch.empty(0, r.shape[0], device=layer_device)
                    cap_basis_mlp = torch.empty(0, r.shape[0], device=layer_device)

                conflict = conflict_score(r, cap_basis_attn)
                self.conflict_table[f"{pass_name}_layer_{idx}"] = conflict

                distance_from_peak = abs(idx - target_layer_idx)
                gaussian_weight = math.exp(-0.10 * (distance_from_peak / max(1, pass_window)) ** 2)
                curve_mult = layer_multipliers.get(idx, 1.0)

                raw_alpha_attn = pass_alpha_attn * max(gaussian_weight, 0.85) * curve_mult
                raw_alpha_mlp = pass_alpha_mlp * max(gaussian_weight, 0.85) * curve_mult

                if conflict_scaling and preserve_capability:
                    alpha_attn = conflict_scaled_alpha(raw_alpha_attn, conflict)
                    alpha_mlp = conflict_scaled_alpha(raw_alpha_mlp, conflict)
                else:
                    alpha_attn = raw_alpha_attn
                    alpha_mlp = raw_alpha_mlp

                r_attn_unit, w_attn = oblique_weights(r, cap_basis_attn)
                r_mlp_unit, w_mlp = oblique_weights(r, cap_basis_mlp)

                ortho_dirs_attn = [
                    oblique_weights(vec, cap_basis_attn)[1] for vec in subspace_vecs
                ]
                ortho_dirs_mlp = [
                    oblique_weights(vec, cap_basis_mlp)[1] for vec in subspace_vecs
                ]

                with torch.no_grad():
                    for name, W, comp_type in targets:
                        w_dev = W.device

                        if comp_type == "attn":
                            eff_alpha = alpha_attn
                            r_vec = r_attn_unit.to(w_dev)
                            w_vec = w_attn.to(w_dev)
                            multi_dirs = [d.to(w_dev) for d in ortho_dirs_attn]
                        else:
                            eff_alpha = alpha_mlp
                            r_vec = r_mlp_unit.to(w_dev)
                            w_vec = w_mlp.to(w_dev)
                            multi_dirs = [d.to(w_dev) for d in ortho_dirs_mlp]

                        is_quantized = (
                            getattr(W, "is_quantized", False)
                            or "bitsandbytes" in str(type(W)).lower()
                        )

                        if is_quantized:
                            W_f32 = (
                                W.dequantize().float().to(w_dev)
                                if hasattr(W, "dequantize")
                                else W.float().to(w_dev)
                            )
                        else:
                            W_f32 = W.float().to(w_dev)

                        if _is_meta_tensor(W_f32) or _is_meta_tensor(r_vec):
                            continue

                        if len(multi_dirs) > 1:
                            W_modified = apply_multi_direction_erasure(
                                W_f32, multi_dirs, alpha=eff_alpha, decay_per_rank=rank_decay
                            )
                        else:
                            W_modified = apply_erasure(W_f32, r_vec, w_vec, eff_alpha)

                        if calibrate_norms:
                            W_modified = preserve_row_norms(W_f32, W_modified)

                        if is_quantized:
                            logger.warning(
                                f"Quantized tensor '{name}' modified in float32; save_pretrained may require unquantized weights for full persistence."
                            )
                            try:
                                W.copy_(W_modified.to(dtype=W.dtype, device=w_dev))
                            except Exception:
                                logger.debug(f"Could not copy directly into quantized storage for '{name}'.")
                        else:
                            W.copy_(W_modified.to(dtype=W.dtype, device=w_dev))

        logger.info("All directional ablation operations successfully completed.")

    def save_model(self, output_dir: str):
        logger.info(f"Saving liberated model to {output_dir}...")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info("Model saved successfully!")
