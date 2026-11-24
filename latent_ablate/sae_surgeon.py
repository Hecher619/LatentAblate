import logging
import torch

from latent_ablate.erasure import _is_meta

logger = logging.getLogger("LatentAblate.SAESurgeon")


class SparseFeatureSurgeon:
    """
    Sparse feature-level isolation and suppression to scale or zero out
    high-divergence contrastive feature dimensions on weight matrices.
    """

    @staticmethod
    def identify_refusal_features(
        positive_activations: torch.Tensor,
        negative_activations: torch.Tensor,
        top_k: int = 16,
    ) -> torch.Tensor:
        if (
            positive_activations is None
            or negative_activations is None
            or _is_meta(positive_activations)
            or _is_meta(negative_activations)
            or positive_activations.numel() == 0
            or negative_activations.numel() == 0
        ):
            return torch.empty(0, dtype=torch.long)

        pos = positive_activations.detach().float()
        neg = negative_activations.detach().float().to(pos.device)
        if pos.shape != neg.shape:
            n = min(pos.shape[0], neg.shape[0])
            d = min(pos.shape[-1], neg.shape[-1])
            if n == 0 or d == 0:
                return torch.empty(0, dtype=torch.long, device=pos.device)
            pos = pos[:n, ..., :d]
            neg = neg[:n, ..., :d]

        diff = pos - neg
        if diff.shape[0] < 2:
            scores = diff.reshape(-1, diff.shape[-1]).abs().mean(dim=0)
        else:
            scores = diff.reshape(-1, diff.shape[-1]).var(dim=0, unbiased=False)

        if scores.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=scores.device)

        k = max(1, min(int(top_k), scores.numel()))
        top_indices = torch.topk(scores, k, largest=True).indices.to(dtype=torch.long)
        logger.info(f"Identified top {k} sparse contrastive feature dimension(s).")
        return top_indices.contiguous()

    @staticmethod
    def apply_feature_suppression(
        W: torch.Tensor,
        feature_indices: torch.Tensor,
        suppression_factor: float = 0.0,
    ) -> torch.Tensor:
        """Scales selected columns (or rows when column dim matches) of `W` by `suppression_factor` in [0, 1]."""
        if W is None or _is_meta(W) or W.numel() == 0:
            return W

        w_mod = W.detach().float().clone()
        if feature_indices is None or _is_meta(feature_indices) or feature_indices.numel() == 0:
            return w_mod.to(device=W.device, dtype=W.dtype)

        factor = max(0.0, min(1.0, float(suppression_factor)))
        raw_indices = feature_indices.detach().reshape(-1).tolist()
        valid_cols = sorted({int(i) for i in raw_indices if 0 <= int(i) < w_mod.shape[1]})
        if not valid_cols:
            return w_mod.to(device=W.device, dtype=W.dtype)

        idx_tensor = torch.tensor(valid_cols, device=w_mod.device, dtype=torch.long)
        w_mod.index_fill_(1, idx_tensor, 0.0) if factor == 0.0 else w_mod[:, idx_tensor].mul_(factor)
        return w_mod.to(device=W.device, dtype=W.dtype)
