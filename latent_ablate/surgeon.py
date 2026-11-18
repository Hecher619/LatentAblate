import torch
import logging
import math
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from latent_ablate.arch import resolve_layers, hidden_size, eligible_linears
from latent_ablate.extractor import ActivationExtractor
from latent_ablate.subspace import refusal_direction
from latent_ablate.erasure import conflict_score, build_capability_basis, oblique_weights, apply_erasure
from latent_ablate.calibrator import NormCalibrator
from latent_ablate.strip_system import SystemPromptScrubber

logger = logging.getLogger("LatentAblate.Surgeon")


class LatentAblateSurgeon:
    def __init__(self, model_path: str, device: str = "auto", torch_dtype=torch.float16, load_in_4bit: bool = False):
        logger.info(f"Loading tokenizer and model from {model_path} (4-bit quantization: {load_in_4bit})...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        load_kwargs = {"torch_dtype": torch_dtype, "device_map": device, "trust_remote_code": True}
        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4"
                )
            except ImportError:
                raise ImportError("bitsandbytes is required for 4-bit loading. Run `pip install bitsandbytes`.")

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
            is_attn = any(k in low for k in ["o_proj", "v_proj", "q_proj", "k_proj", "attention", "query", "key", "value"])
            is_mlp = any(k in low for k in ["down_proj", "gate_proj", "up_proj", "dense", "c_proj", "w1", "w2", "w3", "ffn"])

            if "attn" in target_components and is_attn:
                targets.append((name, W, "attn"))
            elif "mlp" in target_components and is_mlp:
                targets.append((name, W, "mlp"))
        return targets

    def execute_multi_pass_surgery(
        self,
        passes: list[dict],
        capability_prompts: list,
        layer_window: int = 3,
        basis_rank_attn: int = 16,
        basis_rank_mlp: int = 12,
        scaling_factor_attn: float = 0.90,
        scaling_factor_mlp: float = 0.65,
        target_components: list = ["attn", "mlp"],
        preserve_capability: bool = True,
    ):
        for pass_idx, pass_data in enumerate(passes):
            pass_name = pass_data.get("name", f"Pass_{pass_idx+1}")
            pos_prompts = pass_data.get("positive", [])
            neg_prompts = pass_data.get("negative", [])
            target_layer_idx = pass_data.get("target_layer", 18)

            logger.info(f"=== Starting Surgical Pass [{pass_idx+1}/{len(passes)}]: {pass_name} at Layer {target_layer_idx} ===")

            start = max(0, target_layer_idx - layer_window)
            end = min(len(self.layers) - 1, target_layer_idx + layer_window)

            sys_basis = SystemPromptScrubber.isolate_system_obedience_subspace(
                self.model, self.tokenizer, self.device, rank=8
            )

            for idx in tqdm(range(start, end + 1), desc=f"Pass '{pass_name}' Layers"):
                layer = self.layers[idx]
                targets = self._get_component_targets(layer, target_components)
                if not targets:
                    continue

                layer_device = next(layer.parameters()).device

                pos = self.extractor.extract_hidden_states(pos_prompts, idx).to(layer_device)
                neg = self.extractor.extract_hidden_states(neg_prompts, idx).to(layer_device)
                r = refusal_direction(pos, neg).to(layer_device)

                cap = self.extractor.extract_hidden_states(capability_prompts, idx).to(layer_device)
                cap_basis_attn = build_capability_basis(cap, rank=basis_rank_attn).to(layer_device)
                cap_basis_mlp = build_capability_basis(cap, rank=basis_rank_mlp).to(layer_device)

                sys_dev = sys_basis.to(layer_device)
                combined_attn = torch.cat([cap_basis_attn, sys_dev], dim=0)
                basis_attn = torch.linalg.qr(combined_attn.T, mode="reduced").Q.T
                basis_mlp = cap_basis_mlp

                conflict = conflict_score(r, basis_attn)
                self.conflict_table[f"{pass_name}_layer_{idx}"] = conflict

                distance_from_peak = abs(idx - target_layer_idx)
                gaussian_weight = math.exp(-0.15 * (distance_from_peak / max(1, layer_window)) ** 2)
                
                alpha_attn = scaling_factor_attn * max(gaussian_weight, 0.9)
                alpha_mlp = scaling_factor_mlp * max(gaussian_weight, 0.9)

                if preserve_capability:
                    w_attn, _ = oblique_weights(r, basis_attn)
                    w_mlp, _ = oblique_weights(r, basis_mlp)
                else:
                    w_attn = r / r.pow(2).sum().clamp_min(1e-8)
                    w_mlp = r / r.pow(2).sum().clamp_min(1e-8)

                with torch.no_grad():
                    for name, W, comp_type in targets:
                        w_dev = W.device
                        r_dev = r.to(w_dev)
                        
                        if comp_type == "attn":
                            eff_alpha = alpha_attn
                            w_dev_tensor = w_attn.to(w_dev)
                        else:
                            eff_alpha = alpha_mlp
                            w_dev_tensor = w_mlp.to(w_dev)

                        is_quantized = getattr(W, "is_quantized", False) or "bitsandbytes" in str(type(W)).lower()

                        if is_quantized:
                            W_f32 = W.dequantize().float().to(w_dev) if hasattr(W, "dequantize") else W.float().to(w_dev)
                        else:
                            W_f32 = W.float().to(w_dev)

                        W_modified = apply_erasure(W_f32, r_dev, w_dev_tensor, eff_alpha)
                        rows = torch.ones(W.shape[0], dtype=torch.bool, device=w_dev)
                        NormCalibrator.preserve_row_norms(W_modified, W_f32, rows)

                        W.copy_(W_modified.to(W.dtype))

        logger.info("All component-split CWOE and system scrubbing operations successfully completed.")

    def save_model(self, output_dir: str):
        logger.info(f"Saving liberated model to {output_dir}...")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info("Model saved successfully!")
