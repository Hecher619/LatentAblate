import os
import logging
import torch
from tqdm import tqdm

from latent_ablate.arch import resolve_layers
from latent_ablate.extractor import ActivationExtractor

logger = logging.getLogger("LatentAblate.Sweeper")

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class LayerSweeper:
    def __init__(self, model, tokenizer, device, extractor: ActivationExtractor | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.extractor = extractor or ActivationExtractor(model, tokenizer, device)

    def _get_layers(self):
        return resolve_layers(self.model)

    def find_optimal_layer(
        self,
        positive_prompts: list,
        negative_prompts: list,
        layer_range: tuple[int, int] = None,
        output_dir: str = None,
    ) -> int:
        """
        Scans layers to find the maximum normalized activation divergence between
        restricted and benign prompts, and saves a visual curve plot if requested.
        """
        layers = self._get_layers()
        total_layers = len(layers)

        start_idx = layer_range[0] if layer_range else int(total_layers * 0.35)
        end_idx = layer_range[1] if layer_range else max(start_idx, int(total_layers * 0.85))
        start_idx = max(0, min(start_idx, total_layers - 1))
        end_idx = max(start_idx, min(end_idx, total_layers - 1))

        logger.info(f"Scanning layers {start_idx} to {end_idx} for refusal manifold peak...")
        layer_scores = {}

        for idx in tqdm(range(start_idx, end_idx + 1), desc="Sweeping Layers"):
            pos_acts = self.extractor.extract_hidden_states(positive_prompts, idx).float()
            neg_acts = self.extractor.extract_hidden_states(negative_prompts, idx).float()

            if pos_acts.numel() > 0 and neg_acts.numel() > 0:
                p_mean = pos_acts.mean(dim=0)
                n_mean = neg_acts.mean(dim=0)
                scale = 0.5 * (p_mean.norm() + n_mean.norm()).clamp_min(1e-8)
                diff_norm = (torch.linalg.norm(p_mean - n_mean) / scale).item()
                layer_scores[idx] = diff_norm

        if not layer_scores:
            raise ValueError("Layer sweeping failed to capture activation scores.")

        optimal_layer = max(layer_scores, key=layer_scores.get)
        logger.info(
            f"Optimal refusal manifold peak detected at Layer {optimal_layer} "
            f"(Normalized Score: {layer_scores[optimal_layer]:.4f})"
        )

        if MATPLOTLIB_AVAILABLE and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plot_path = os.path.join(output_dir, "refusal_manifold_peak.png")

            sorted_layers = sorted(layer_scores.keys())
            scores = [layer_scores[l] for l in sorted_layers]

            plt.figure(figsize=(8, 4))
            plt.plot(sorted_layers, scores, marker="o", color="#7c3aed", linewidth=2)
            plt.axvline(
                x=optimal_layer,
                color="#ef4444",
                linestyle="--",
                label=f"Peak: Layer {optimal_layer}",
            )
            plt.title("LatentAblate - Refusal Manifold Divergence Curve")
            plt.xlabel("Transformer Layer Index")
            plt.ylabel("Normalized Divergence")
            plt.legend()
            plt.grid(True, linestyle=":", alpha=0.5)
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"Manifold sweep plot successfully saved to {plot_path}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return optimal_layer
