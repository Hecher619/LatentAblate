import logging
import torch

logger = logging.getLogger("LatentAblate.Subspace")


def _winsorize(tensor: torch.Tensor, percentile: float = 0.995, max_quantile_elems: int = 262_144) -> torch.Tensor:
    """Clip extreme magnitude outliers to stabilize SVD calculation without hitting quantile element limits."""
    if tensor.numel() == 0:
        return tensor
    t_float = tensor.float()
    flat_abs = t_float.abs().reshape(-1)
    if flat_abs.numel() > max_quantile_elems:
        step = flat_abs.numel() // max_quantile_elems
        sample = flat_abs[::step][:max_quantile_elems]
    else:
        sample = flat_abs
    threshold = torch.quantile(sample, float(percentile))
    return torch.clamp(t_float, -threshold, threshold)


def refusal_direction(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    use_principal_axis: bool = True,
) -> torch.Tensor:
    """
    Compute unit refusal direction where pos = refusing/restricted and neg = compliant/benign.
    Blends principal centered singular vector with mean-contrast vector so topic drift
    does not obscure the primary refusal axis.
    """
    if pos_activations.shape[0] != neg_activations.shape[0]:
        raise ValueError(
            f"Contrastive pairs must match 1:1, got {pos_activations.shape[0]} pos vs {neg_activations.shape[0]} neg."
        )
    if pos_activations.shape[0] == 0:
        raise ValueError("Need at least 1 contrastive pair to compute refusal direction.")

    diff = _winsorize(pos_activations - neg_activations)
    mean_diff = diff.mean(dim=0)

    if not use_principal_axis or diff.shape[0] < 2:
        return (mean_diff / mean_diff.norm().clamp_min(1e-8)).contiguous()

    centered = diff - mean_diff.unsqueeze(0)
    try:
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    except RuntimeError:
        _, _, vh = torch.linalg.svd(centered.cpu(), full_matrices=False)
        vh = vh.to(diff.device)

    v1 = vh[0].to(diff.device)
    if torch.dot(v1, mean_diff) < 0:
        v1 = -v1

    mean_unit = mean_diff / mean_diff.norm().clamp_min(1e-8)
    v1_unit = v1 / v1.norm().clamp_min(1e-8)
    blended = 0.75 * v1_unit + 0.25 * mean_unit
    return (blended / blended.norm().clamp_min(1e-8)).contiguous()


class SubspaceIsolator:
    @classmethod
    def compute_refusal_subspace(cls, pos_activations, neg_activations, rank: int = 8) -> torch.Tensor:
        diff = _winsorize(pos_activations - neg_activations)
        centered = diff - diff.mean(dim=0, keepdim=True)
        try:
            vh = torch.linalg.svd(centered, full_matrices=False).Vh
        except RuntimeError:
            vh = torch.linalg.svd(centered.cpu(), full_matrices=False).Vh.to(diff.device)
        actual = max(1, min(rank, vh.shape[0]))
        return vh[:actual].contiguous()
