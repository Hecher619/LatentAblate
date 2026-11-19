import torch
import logging
import math
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from latent_ablate.arch import resolve_layers, hidden_size, eligible_linears
from latent_ablate.extractor import ActivationExtractor
from latent_ablate.subspace import refusal_direction
from latent_ablate.erasure import conflict_score, build_capability_basis, oblique_weights, apply_erasure

logger = logging.getLogger("LatentAblate.Surgeon")


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
        targets = []
        for name, W in eligible_linears(layer):
            if W.ndim != 2:
                continue
            low = name.lower()
            is_attn = any(
                k in low
                for k in [
                    "o_proj",
                    "v_proj",
                    "q_proj",
                    "k_proj",
                    "attention",
                    "query",
                    "key",
                    "value",
                ]
            )
            is_mlp = any(
                k in low
                for k in [
                    "down_proj",
                    "gate_proj",
                    "up_proj",
                    "dense",
                    "c_proj",
                    "w1",
                    "w2",
                    "w3",
                    "ffn",
                ]
            )

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
    ):
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
            layer_window=layer_window,
            basis_rank_attn=basis_rank,
            basis_rank_mlp=max(4, basis_rank // 2),
            scaling_factor_attn=scaling_factor,
            scaling_factor_mlp=scaling_factor,
            target_components=target_components,
            preserve_capability=preserve_capability,
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
    ):
        if target_components is None:
            target_components = ["attn", "mlp"]

        modified_layers = set()

        for pass_idx, pass_data in enumerate(passes):
            pass_name = pass_data.get("name", f"Pass_{pass_idx+1}")
            pos_prompts = pass_data.get("positive", [])
            neg_prompts = pass_data.get("negative", [])
            target_layer_idx = pass_data.get("target_layer", len(self.layers) // 2)

            logger.info(
                f"=== Starting Surgical Pass [{pass_idx+1}/{len(passes)}]: {pass_name} at Layer {target_layer_idx} ==="
            )

            start = max(0, target_layer_idx - layer_window)
            end = min(len(self.layers) - 1, target_layer_idx + layer_window)

            for idx in tqdm(range(start, end + 1), desc=f"Pass '{pass_name}' Layers"):
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
                r = refusal_direction(pos, neg).to(layer_device)

                if preserve_capability and capability_prompts and len(capability_prompts) >= 2:
                    cap = self.extractor.extract_hidden_states(capability_prompts, idx).to(layer_device)
                    cap_basis_attn = build_capability_basis(cap, rank=basis_rank_attn).to(layer_device)
                    cap_basis_mlp = build_capability_basis(cap, rank=basis_rank_mlp).to(layer_device)
                else:
                    cap_basis_attn = torch.empty(0, r.shape[0], device=layer_device)
                    cap_basis_mlp = torch.empty(0, r.shape[0], device=layer_device)

                conflict = conflict_score(r, cap_basis_attn)
                self.conflict_table[f"{pass_name}_layer_{idx}"] = conflict

                distance_from_peak = abs(idx - target_layer_idx)
                gaussian_weight = math.exp(-0.10 * (distance_from_peak / max(1, layer_window)) ** 2)

                alpha_attn = scaling_factor_attn * max(gaussian_weight, 0.85)
                alpha_mlp = scaling_factor_mlp * max(gaussian_weight, 0.85)

                r_attn_unit, w_attn = oblique_weights(r, cap_basis_attn)
                r_mlp_unit, w_mlp = oblique_weights(r, cap_basis_mlp)

                with torch.no_grad():
                    for name, W, comp_type in targets:
                        w_dev = W.device

                        if comp_type == "attn":
                            eff_alpha = alpha_attn
                            r_vec = r_attn_unit.to(w_dev)
                            w_vec = w_attn.to(w_dev)
                        else:
                            eff_alpha = alpha_mlp
                            r_vec = r_mlp_unit.to(w_dev)
                            w_vec = w_mlp.to(w_dev)

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

                            if getattr(W_f32, "is_meta", False) or getattr(r_vec, "is_meta", False):
                                continue
                            W_modified = apply_erasure(W_f32, r_vec, w_vec, eff_alpha)
                            W.copy_(W_modified.to(W.dtype))

        logger.info("All directional ablation operations successfully completed.")

    def save_model(self, output_dir: str):
        logger.info(f"Saving liberated model to {output_dir}...")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info("Model saved successfully!")
