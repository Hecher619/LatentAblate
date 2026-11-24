import logging
import torch

logger = logging.getLogger("LatentAblate.Subspace")


def _is_meta(tensor: torch.Tensor | None) -> bool:
    if tensor is None:
        return False
    if getattr(tensor, "is_meta", False):
        return True
    dev = getattr(tensor, "device", None)
    return dev is not None and dev.type == "meta"


def _winsorize(
    tensor: torch.Tensor,
    percentile: float = 0.995,
    max_quantile_elems: int = 262_144,
) -> torch.Tensor:
    """Clip extreme magnitude outliers to stabilize SVD calculation without hitting quantile element limits."""
    if tensor is None or tensor.numel() == 0 or _is_meta(tensor):
        return tensor
    t_float = tensor.float()
    flat_abs = t_float.abs().reshape(-1)
    if flat_abs.numel() > max_quantile_elems:
        step = flat_abs.numel() // max_quantile_elems
        sample = flat_abs[::step][:max_quantile_elems]
    else:
        sample = flat_abs
    q_val = max(0.0, min(1.0, float(percentile)))
    threshold = torch.quantile(sample, q_val)
    return torch.clamp(t_float, -threshold, threshold)


def _centered_contrast_svd_vh(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (diff, mean_diff, vh) for winsorized contrastive activation pairs."""
    if pos_activations is None or neg_activations is None:
        raise ValueError("Contrastive activation tensors must not be None.")
    if pos_activations.shape[0] != neg_activations.shape[0]:
        raise ValueError(
            f"Contrastive pairs must match 1:1, got {pos_activations.shape[0]} pos vs {neg_activations.shape[0]} neg."
        )
    if pos_activations.shape[0] == 0 or pos_activations.numel() == 0 or neg_activations.numel() == 0:
        raise ValueError("Need at least 1 non-empty contrastive pair to compute refusal subspace.")

    dev = pos_activations.device
    pos_f = pos_activations.float().to(dev)
    neg_f = neg_activations.float().to(dev)
    diff = _winsorize(pos_f - neg_f)
    mean_diff = diff.mean(dim=0)

    if _is_meta(diff):
        hidden = diff.shape[-1] if diff.ndim >= 1 else 1
        eye_vh = torch.eye(1, hidden, device=dev, dtype=torch.float32)
        return diff, mean_diff, eye_vh

    if diff.shape[0] < 2:
        unit = mean_diff / mean_diff.norm().clamp_min(1e-8)
        return diff, mean_diff, unit.unsqueeze(0)

    centered = diff - mean_diff.unsqueeze(0)
    try:
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    except RuntimeError:
        _, _, vh = torch.linalg.svd(centered.cpu(), full_matrices=False)
        vh = vh.to(dev)

    return diff, mean_diff, vh


def refusal_direction(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    use_principal_axis: bool = True,
    principal_blend: float = 0.75,
) -> torch.Tensor:
    """
    Compute unit refusal direction where pos = refusing/restricted and neg = compliant/benign.
    Blends principal centered singular vector with mean-contrast vector so topic drift
    does not obscure the primary refusal axis.
    """
    _, mean_diff, vh = _centered_contrast_svd_vh(pos_activations, neg_activations)
    mean_unit = mean_diff / mean_diff.norm().clamp_min(1e-8)

    if not use_principal_axis or _is_meta(vh) or vh.shape[0] == 0:
        return mean_unit.contiguous()

    v1 = vh[0].to(mean_unit.device)
    if not _is_meta(v1) and float(torch.dot(v1, mean_diff).item()) < 0.0:
        v1 = -v1

    v1_unit = v1 / v1.norm().clamp_min(1e-8)
    blend = max(0.0, min(1.0, float(principal_blend)))
    blended = blend * v1_unit + (1.0 - blend) * mean_unit
    return (blended / blended.norm().clamp_min(1e-8)).contiguous()


class SubspaceIsolator:
    @classmethod
    def extract_refusal_direction(
        cls,
        pos_activations: torch.Tensor,
        neg_activations: torch.Tensor,
        use_principal_axis: bool = True,
        principal_blend: float = 0.75,
    ) -> torch.Tensor:
        return refusal_direction(
            pos_activations,
            neg_activations,
            use_principal_axis=use_principal_axis,
            principal_blend=principal_blend,
        )

    @classmethod
    def compute_refusal_subspace(
        cls,
        pos_activations: torch.Tensor,
        neg_activations: torch.Tensor,
        rank: int = 8,
    ) -> torch.Tensor:
        _, mean_diff, vh = _centered_contrast_svd_vh(pos_activations, neg_activations)
        if _is_meta(vh) or vh.shape[0] == 0:
            mean_unit = mean_diff / mean_diff.norm().clamp_min(1e-8)
            return mean_unit.unsqueeze(0).contiguous()
        actual = max(1, min(int(rank), vh.shape[0]))
        return vh[:actual].contiguous()
