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
            if W.shape[0] != self.hidden:
                continue
            low = name.lower()
            is_attn = any(k in low for k in ["o_proj", "v_proj", "q_proj", "k_proj", "attention", "query", "key", "value"])
            is_mlp = any(k in low for k in ["down_proj", "dense", "c_proj", "w2", "ffn", "dense_4h_to_h"])

            if "attn" in target_components and is_attn:
                targets.append((name, W, 1.0))
            elif "mlp" in target_components and is_mlp:
                targets.append((name, W, 0.75))
        return targets

    def execute_surgery(
        self,
        positive_prompts: list,
        negative_prompts: list,
        capability_prompts: list,
        target_layer_idx: int,
        layer_window: int = 2,
        basis_rank: int = 16,
        scaling_factor: float = 0.75,
        target_components: list = ["attn", "mlp"],
        preserve_capability: bool = True,
    ):
        start = max(0, target_layer_idx - layer_window)
        end = min(len(self.layers) - 1, target_layer_idx + layer_window)
        logger.info(f"Executing advanced CWOE across layers {start}..{end} with components: {target_components}")

        for idx in tqdm(range(start, end + 1), desc="Advanced Surgical Pass"):
            layer = self.layers[idx]
            targets = self._get_component_targets(layer, target_components)
            if not targets:
                continue

            # Ensure activations are extracted on the specific layer's device
            layer_device = next(layer.parameters()).device
            
            pos = self.extractor.extract_hidden_states(positive_prompts, idx).to(layer_device)
            neg = self.extractor.extract_hidden_states(negative_prompts, idx).to(layer_device)
            r = refusal_direction(pos, neg).to(layer_device)

            cap = self.extractor.extract_hidden_states(capability_prompts, idx).to(layer_device)
            basis = build_capability_basis(cap, rank=basis_rank).to(layer_device)

            conflict = conflict_score(r, basis)
            self.conflict_table[idx] = conflict

            distance_from_peak = abs(idx - target_layer_idx)
            gaussian_weight = math.exp(-0.5 * (distance_from_peak / max(1, layer_window / 2)) ** 2)
            alpha = scaling_factor * gaussian_weight

            if preserve_capability:
                w, fell_back = oblique_weights(r, basis)
            else:
                w, fell_back = r / r.pow(2).sum().clamp_min(1e-8), True
                alpha *= (1.0 - conflict)

            logger.info(f"Layer {idx} [{layer_device}]: conflict={conflict:.3f} kernel_alpha={alpha:.3f} oblique={not fell_back}")

            with torch.no_grad():
                for name, W, comp_multiplier in targets:
                    effective_alpha = alpha * comp_multiplier
                    w_dev = W.device
                    
                    # Force tensors to target weight device
                    r_dev = r.to(w_dev)
                    w_dev_tensor = w.to(w_dev)

                    is_quantized = getattr(W, "is_quantized", False) or "bitsandbytes" in str(type(W)).lower()
                    
                    if is_quantized:
                        W_f32 = W.dequantize().float().to(w_dev) if hasattr(W, "dequantize") else W.float().to(w_dev)
                    else:
                        W_f32 = W.float().to(w_dev)

                    W_modified = apply_erasure(W_f32, r_dev, w_dev_tensor, effective_alpha)
                    rows = torch.ones(W.shape[0], dtype=torch.bool, device=w_dev)
                    NormCalibrator.preserve_row_norms(W_modified, W_f32, rows)

                    W.copy_(W_modified.to(W.dtype))

        logger.info("Advanced CWOE multi-component surgery successfully completed.")

    def save_model(self, output_dir: str):
        logger.info(f"Saving liberated model to {output_dir}...")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info("Model saved successfully!")
