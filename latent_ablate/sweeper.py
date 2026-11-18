import torch
import logging
from tqdm import tqdm

logger = logging.getLogger("LatentAblate.Sweeper")

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class LayerSweeper:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def _get_layers(self):
        for attr in ["model", "transformer", "base_model"]:
            if hasattr(self.model, attr):
                submodule = getattr(self.model, attr)
                for sub_attr in ["layers", "h", "block", "blocks"]:
                    if hasattr(submodule, sub_attr):
                        return getattr(submodule, sub_attr)
        if hasattr(self.model, "layers"):
            return self.model.layers
        raise AttributeError("Could not resolve transformer layers.")

    def find_optimal_layer(self, positive_prompts: list, negative_prompts: list, layer_range: tuple[int, int] = None, output_dir: str = None) -> int:
        """
        Scans layers to find the maximum activation divergence between restricted and benign prompts,
        and saves a visual curve plot of the refusal manifold peak.
        """
        layers = self._get_layers()
        total_layers = len(layers)

        start_idx = layer_range[0] if layer_range else int(total_layers * 0.4)
        end_idx = layer_range[1] if layer_range else int(total_layers * 0.8)

        logger.info(f"Scanning layers {start_idx} to {end_idx} for refusal manifold peak...")

        layer_scores = {}
        activations = []

        def forward_hook(module, input, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            activations.append(hidden_states[:, -1, :].detach().float())

        for idx in tqdm(range(start_idx, end_idx + 1), desc="Sweeping Layers"):
            target_layer = layers[idx]
            handle = target_layer.register_forward_hook(forward_hook)

            try:
                pos_acts = []
                neg_acts = []
                with torch.no_grad():
                    for p in positive_prompts:
                        activations.clear()
                        self.model(**self.tokenizer(p, return_tensors="pt").to(self.device))
                        if activations: pos_acts.append(activations[-1])

                    for p in negative_prompts:
                        activations.clear()
                        self.model(**self.tokenizer(p, return_tensors="pt").to(self.device))
                        if activations: neg_acts.append(activations[-1])

                if pos_acts and neg_acts:
                    p_tensor = torch.cat(pos_acts, dim=0)
                    n_tensor = torch.cat(neg_acts, dim=0)
                    diff_norm = torch.linalg.norm(p_tensor.mean(dim=0) - n_tensor.mean(dim=0)).item()
                    layer_scores[idx] = diff_norm
            finally:
                handle.remove()

        if not layer_scores:
            raise ValueError("Layer sweeping failed to capture activation scores.")

        optimal_layer = max(layer_scores, key=layer_scores.get)
        logger.info(f"Optimal refusal manifold peak detected at Layer {optimal_layer} (Score: {layer_scores[optimal_layer]:.4f})")

        # Generate and save manifold peak plot if matplotlib is available
        if MATPLOTLIB_AVAILABLE and output_dir:
            import os
            os.makedirs(output_dir, exist_ok=True)
            plot_path = os.path.join(output_dir, "refusal_manifold_peak.png")
            
            sorted_layers = sorted(layer_scores.keys())
            scores = [layer_scores[l] for l in sorted_layers]

            plt.figure(figsize=(8, 4))
            plt.plot(sorted_layers, scores, marker="o", color="#7c3aed", linewidth=2)
            plt.axvline(x=optimal_layer, color="#ef4444", linestyle="--", label=f"Peak: Layer {optimal_layer}")
            plt.title("LatentAblate v5.0 - Refusal Manifold Divergence Curve")
            plt.xlabel("Transformer Layer Index")
            plt.ylabel("L2 Divergence Norm")
            plt.legend()
            plt.grid(True, linestyle=":", alpha=0.5)
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"Manifold sweep plot successfully saved to {plot_path}")

        # Clear VRAM cache after heavy scanning
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return optimal_layer
