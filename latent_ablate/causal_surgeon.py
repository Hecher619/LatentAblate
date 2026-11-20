import logging
import torch
from tqdm import tqdm

from latent_ablate.arch import resolve_layers
from latent_ablate.extractor import ActivationExtractor

logger = logging.getLogger("LatentAblate.CausalSurgeon")


class CausalMediationAnalyzer:
    """
    Executes batched forward hooks and contrastive activation tracking
    to measure layer sensitivity and locate causal refusal nodes.
    """

    @staticmethod
    def compute_indirect_effects(
        model,
        tokenizer,
        positive_prompts: list,
        negative_prompts: list,
        device: str,
        max_prompts: int = 32,
        extractor: ActivationExtractor | None = None,
    ) -> dict:
        logger.info("Running causal mediation analysis across transformer layers...")
        model.eval()
        layers = resolve_layers(model)
        ext = extractor or ActivationExtractor(model, tokenizer, device)

        pos_sub = positive_prompts[:max_prompts]
        neg_sub = negative_prompts[:max_prompts]
        layer_sensitivity = {}

        for idx in tqdm(range(len(layers)), desc="Causal Tracing Layers"):
            pos_acts = ext.extract_hidden_states(pos_sub, idx).float()
            neg_acts = ext.extract_hidden_states(neg_sub, idx).float()

            if pos_acts.numel() > 0 and neg_acts.numel() > 0:
                p_tensor = pos_acts.mean(dim=0)
                n_tensor = neg_acts.mean(dim=0)
                scale = (0.5 * (p_tensor.norm() + n_tensor.norm())).clamp_min(1e-8)
                effect_score = (torch.linalg.norm(p_tensor - n_tensor) / scale).item()
                layer_sensitivity[idx] = effect_score

        optimal_nodes = sorted(layer_sensitivity, key=layer_sensitivity.get, reverse=True)[:5]
        logger.info(f"Top causal refusal nodes identified at layers: {optimal_nodes}")
        return {"sensitivity_map": layer_sensitivity, "target_nodes": optimal_nodes}
