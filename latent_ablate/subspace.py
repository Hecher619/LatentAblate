import torch
import logging

logger = logging.getLogger("LatentAblate.Subspace")


def _winsorize(tensor: torch.Tensor, percentile: float = 0.995) -> torch.Tensor:
    """Clip extreme magnitude outliers to stabilize SVD calculation."""
    threshold = torch.quantile(tensor.abs(), percentile)
    return torch.clamp(tensor, -threshold, threshold)


def refusal_direction(pos_activations: torch.Tensor, neg_activations: torch.Tensor) -> torch.Tensor:
    """Unit mean-contrast direction. pos = refuses, neg = complies."""
    if pos_activations.shape[0] != neg_activations.shape[0]:
        raise ValueError(
            f"Contrastive pairs must match 1:1, got {pos_activations.shape[0]} pos vs {neg_activations.shape[0]} neg."
        )
    diff = _winsorize(pos_activations.float() - neg_activations.float())
    r = diff.mean(dim=0)
    return r / r.norm().clamp_min(1e-8)


class SubspaceIsolator:
    @classmethod
    def compute_refusal_subspace(cls, pos_activations, neg_activations, rank: int = 8) -> torch.Tensor:
        diff = _winsorize(pos_activations.float() - neg_activations.float())
        vh = torch.linalg.svd(diff, full_matrices=False).Vh
        actual = min(rank, vh.shape[0])
        if actual < rank:
            logger.info(f"Subspace rank capped at {actual} by pair count (requested {rank}).")
        return vh[:actual]
