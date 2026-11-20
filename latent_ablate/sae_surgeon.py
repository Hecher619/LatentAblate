import logging
import torch

logger = logging.getLogger("LatentAblate.SAESurgeon")


class SparseFeatureSurgeon:
    """
    Sparse feature-level isolation and suppression to zero out or scale
    high-divergence contrastive refusal feature dimensions.
    """

    @staticmethod
    def identify_refusal_features(
        positive_activations: torch.Tensor,
        negative_activations: torch.Tensor,
        top_k: int = 16,
    ) -> torch.Tensor:
        pos = positive_activations.float()
        neg = negative_activations.float()

        diff_variance = (pos - neg).var(dim=0)
        if diff_variance.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=diff_variance.device)

        k = max(1, min(int(top_k), diff_variance.numel()))
        top_indices = torch.topk(diff_variance, k).indices
        logger.info(f"Identified top {k} sparse contrastive refusal feature dimensions.")
        return top_indices

    @staticmethod
    def apply_feature_suppression(
        W: torch.Tensor,
        feature_indices: torch.Tensor,
        suppression_factor: float = 0.0,
    ) -> torch.Tensor:
        """Zeros out or scales specific weight columns corresponding to refusal features."""
        W_mod = W.float().clone()
        if feature_indices is None or feature_indices.numel() == 0:
            return W_mod
        for idx in feature_indices:
            i = int(idx.item()) if torch.is_tensor(idx) else int(idx)
            if 0 <= i < W_mod.shape[1]:
                W_mod[:, i] *= float(suppression_factor)
        return W_mod
