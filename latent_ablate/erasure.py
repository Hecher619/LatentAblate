import logging
import torch

logger = logging.getLogger("LatentAblate.Erasure")


def _is_meta(tensor: torch.Tensor | None) -> bool:
    if tensor is None:
        return False
    if getattr(tensor, "is_meta", False):
        return True
    dev = getattr(tensor, "device", None)
    return dev is not None and dev.type == "meta"


def _normalize_1d(vec: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    v = vec.float()
    return v / v.norm().clamp_min(eps)


def build_capability_basis(
    capability_activations: torch.Tensor,
    rank: int = 16,
    energy_threshold: float = 0.99,
) -> torch.Tensor:
    """
    Constructs an orthonormal basis [k, hidden_dim] representing benign/capability
    subspace directions from centered capability activations via SVD.
    """
    if capability_activations is None or _is_meta(capability_activations):
        return torch.empty(0, dtype=torch.float32)

    acts = capability_activations.float()
    if acts.ndim == 1:
        acts = acts.unsqueeze(0)
    if acts.shape[0] < 2 or acts.shape[1] == 0:
        return torch.empty(0, acts.shape[-1] if acts.ndim == 2 else 0, device=acts.device, dtype=torch.float32)

    centered = acts - acts.mean(dim=0, keepdim=True)
    try:
        _, svals, vh = torch.linalg.svd(centered, full_matrices=False)
    except RuntimeError:
        _, svals, vh = torch.linalg.svd(centered.cpu(), full_matrices=False)
        svals, vh = svals.to(acts.device), vh.to(acts.device)

    if _is_meta(svals) or _is_meta(vh) or vh.shape[0] == 0:
        return torch.empty(0, acts.shape[-1], device=acts.device, dtype=torch.float32)

    max_k = max(1, min(int(rank), vh.shape[0]))
    energies = svals ** 2
    total_energy = float(energies.sum().item())
    if total_energy > 0.0 and 0.0 < float(energy_threshold) < 1.0:
        cum = torch.cumsum(energies[:max_k], dim=0) / total_energy
        kept = max_k
        for idx in range(max_k):
            if float(cum[idx].item()) >= float(energy_threshold):
                kept = idx + 1
                break
        max_k = kept

    basis = vh[:max_k].contiguous()
    return torch.nn.functional.normalize(basis, p=2, dim=-1)


def orthogonalize_against_basis(
    direction: torch.Tensor,
    basis: torch.Tensor | None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Projects a 1D direction orthogonal to a row-orthonormal basis [k, dim] and normalizes."""
    d = _normalize_1d(direction, eps=eps)
    if basis is None or _is_meta(basis) or basis.numel() == 0:
        return d

    b = basis.to(device=d.device, dtype=torch.float32)
    if b.ndim == 1:
        b = b.unsqueeze(0)
    if b.shape[0] == 0 or b.shape[-1] != d.shape[-1]:
        return d

    proj = b.T @ (b @ d)
    ortho = d - proj
    norm = ortho.norm()
    if float(norm.item()) <= 1e-6:
        return d
    return ortho / norm.clamp_min(eps)


def conflict_score(
    refusal_direction: torch.Tensor,
    capability_basis: torch.Tensor | None,
) -> float:
    """
    Measures alignment (0.0 to 1.0) between a refusal vector and the capability subspace.
    0.0 means orthogonal to capability basis; 1.0 means fully inside capability basis.
    """
    if (
        refusal_direction is None
        or _is_meta(refusal_direction)
        or capability_basis is None
        or _is_meta(capability_basis)
        or capability_basis.numel() == 0
    ):
        return 0.0

    r = _normalize_1d(refusal_direction)
    b = capability_basis.to(device=r.device, dtype=torch.float32)
    if b.ndim == 1:
        b = b.unsqueeze(0)
    if b.shape[0] == 0 or b.shape[-1] != r.shape[-1]:
        return 0.0

    q, _ = torch.linalg.qr(b.T, mode="reduced")
    proj = q @ (q.T @ r)
    score = float(proj.norm().clamp(0.0, 1.0).item())
    return score


def conflict_scaled_alpha(
    base_alpha: float,
    c_score: float,
    min_retention: float = 0.25,
    sensitivity: float = 0.75,
) -> float:
    """Attenuates ablation strength smoothly when subspace conflict is high."""
    clamped_c = max(0.0, min(1.0, float(c_score)))
    ret_floor = max(0.0, min(1.0, float(min_retention)))
    scale = max(ret_floor, 1.0 - float(sensitivity) * (clamped_c ** 1.5))
    return float(base_alpha) * scale


def preserve_row_norms(
    original: torch.Tensor,
    modified: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Rescales rows of `modified` so each row matches the Euclidean norm of `original`."""
    if modified is None or _is_meta(modified):
        return modified
    if original is None or _is_meta(original):
        return modified

    orig_f = original.float()
    mod_f = modified.float()
    if orig_f.shape != mod_f.shape:
        return modified

    if mod_f.ndim == 1:
        orig_n = orig_f.norm().clamp_min(eps)
        mod_n = mod_f.norm().clamp_min(eps)
        return (mod_f * (orig_n / mod_n)).to(dtype=modified.dtype, device=modified.device)

    orig_norms = orig_f.norm(dim=-1, keepdim=True).clamp_min(eps)
    mod_norms = mod_f.norm(dim=-1, keepdim=True).clamp_min(eps)
    scaled = mod_f * (orig_norms / mod_norms)
    return scaled.to(dtype=modified.dtype, device=modified.device)


def project_out_subspace(
    weight: torch.Tensor,
    direction: torch.Tensor,
    alpha: float = 1.0,
    capability_basis: torch.Tensor | None = None,
    calibrate_norms: bool = True,
) -> torch.Tensor:
    """
    Applies rank-1 directional projection W' = W - alpha * (W v) v^T along `direction`,
    optionally orthogonalized against `capability_basis` and row-norm calibrated.
    """
    if weight is None or _is_meta(weight) or direction is None or _is_meta(direction):
        return weight

    w_f = weight.float()
    dev = w_f.device
    v = orthogonalize_against_basis(direction.to(dev), capability_basis)
    eff_alpha = max(0.0, float(alpha))

    if w_f.ndim == 1:
        if w_f.shape[0] != v.shape[0]:
            return weight
        proj_scalar = torch.dot(w_f, v)
        out = w_f - eff_alpha * proj_scalar * v
    elif w_f.ndim == 2:
        dim = v.shape[0]
        if w_f.shape[1] == dim:
            proj = (w_f @ v).unsqueeze(1) * v.unsqueeze(0)
            out = w_f - eff_alpha * proj
        elif w_f.shape[0] == dim:
            v_col = v.unsqueeze(1)
            proj = v_col @ (v_col.T @ w_f)
            out = w_f - eff_alpha * proj
        else:
            return weight
    else:
        return weight

    if calibrate_norms:
        out = preserve_row_norms(w_f, out)
    return out.to(dtype=weight.dtype, device=weight.device)


def apply_multi_direction_erasure(
    weight: torch.Tensor,
    directions: list[torch.Tensor],
    alpha: float = 0.5,
    decay_per_rank: float = 0.80,
    capability_basis: torch.Tensor | None = None,
    calibrate_norms: bool = True,
) -> torch.Tensor:
    """Sequentially erases a ranked list of directions with geometrically decaying step scale."""
    if weight is None or _is_meta(weight) or not directions:
        return weight

    cur = weight.float().clone()
    decay = max(0.0, min(1.0, float(decay_per_rank)))
    for idx, d in enumerate(directions):
        if d is None or _is_meta(d) or d.numel() == 0:
            continue
        step_alpha = float(alpha) * (decay ** idx)
        cur = project_out_subspace(
            cur,
            d,
            alpha=step_alpha,
            capability_basis=capability_basis,
            calibrate_norms=False,
        ).float()

    if calibrate_norms:
        cur = preserve_row_norms(weight.float(), cur)
    return cur.to(dtype=weight.dtype, device=weight.device)


def apply_oblivious_erasure(
    weight: torch.Tensor,
    refusal_direction: torch.Tensor,
    alpha: float = 1.0,
    capability_basis: torch.Tensor | None = None,
    preserve_capability: bool = True,
    calibrate_norms: bool = True,
) -> torch.Tensor:
    """Convenience wrapper for single-direction erasure with optional capability protection."""
    cap = capability_basis if preserve_capability else None
    return project_out_subspace(
        weight=weight,
        direction=refusal_direction,
        alpha=alpha,
        capability_basis=cap,
        calibrate_norms=calibrate_norms,
    )
