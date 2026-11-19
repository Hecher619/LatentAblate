import torch
import logging

logger = logging.getLogger("LatentAblate.Erasure")

ORTHO_FALLBACK = 1e-3


def conflict_score(r: torch.Tensor, basis: torch.Tensor) -> float:
    """Fraction of ||r|| that lies inside the capability subspace."""
    if basis is None or basis.numel() == 0 or r.is_meta:
        return 0.0
    device = r.device
    if device.type == "meta":
        return 0.0
    r = r.to(device).float()
    r_unit = r / r.norm().clamp_min(1e-8)
    basis = basis.to(device).float()
    if basis.ndim == 1:
        basis = basis.unsqueeze(0)
    proj = basis.T @ (basis @ r_unit)
    return (proj.norm() / r_unit.norm().clamp_min(1e-8)).item()


def build_capability_basis(activations: torch.Tensor, rank: int = 16) -> torch.Tensor:
    """Orthonormal basis of dominant representation directions on capability prompts."""
    acts = activations.float()
    if acts.shape[0] < 2:
        raise ValueError("Need >= 2 capability activations to build a basis.")

    acts = acts - acts.mean(dim=0, keepdim=True)
    device = acts.device

    try:
        vh = torch.linalg.svd(acts, full_matrices=False).Vh
    except RuntimeError:
        logger.warning("SVD failed on device, falling back to CPU for stable computation.")
        vh = torch.linalg.svd(acts.cpu(), full_matrices=False).Vh.to(device)

    k = min(rank, vh.shape[0])
    basis = vh[:k].to(device)
    if basis.shape[0] > 0:
        basis = torch.linalg.qr(basis.T, mode="reduced").Q.T
    return basis.contiguous()


def oblique_weights(r: torch.Tensor, basis: torch.Tensor) -> tuple:
    """Return normalized refusal direction and orthogonalized direction for capability preservation."""
    r_f = r.float()
    if r.is_meta or r.device.type == "meta":
        r_unit = r_f / r_f.norm().clamp_min(1e-8)
        return r_unit, r_unit

    device = r.device
    r_f = r_f.to(device)
    r_unit = r_f / r_f.norm().clamp_min(1e-8)

    if basis is not None and basis.numel() > 0 and not basis.is_meta and basis.device.type != "meta":
        b = basis.to(device).float()
        if b.ndim == 1:
            b = b.unsqueeze(0)
        r_perp = r_unit - b.T @ (b @ r_unit)
        norm_perp = r_perp.norm()
        if norm_perp > ORTHO_FALLBACK:
            return r_unit, r_perp / norm_perp

    return r_unit, r_unit


def apply_erasure(
    W: torch.Tensor,
    r: torch.Tensor,
    w: torch.Tensor | None = None,
    alpha: float = 1.0,
) -> torch.Tensor:
    """
    Applies directional ablation along unit direction `w` (if provided and non-zero),
    falling back to normalized `r`, on a 2D linear weight matrix W [out_features, in_features].
    """
    device = W.device
    W_f32 = W.float().to(device)
    if W_f32.ndim != 2:
        return W_f32

    if getattr(W, "is_meta", False) or getattr(r, "is_meta", False) or (w is not None and getattr(w, "is_meta", False)):
        return W
    target_vec = w if (w is not None and w.numel() > 0 and w.norm().item() > 1e-8) else r
    u = target_vec.to(device).float()
    u = u / u.norm().clamp_min(1e-8)

    out_features, in_features = W_f32.shape
    dim = u.shape[0]

    # Residual stream in output space (e.g., o_proj, down_proj: [hidden, d_inner])
    if out_features == dim:
        proj = torch.mv(W_f32.T, u)
        delta = alpha * torch.outer(u, proj)
        return W_f32 - delta

    # Residual stream in input space (e.g., q_proj, k_proj, v_proj, gate_proj, up_proj: [d_proj, hidden])
    if in_features == dim:
        proj = torch.mv(W_f32, u)
        delta = alpha * torch.outer(proj, u)
        return W_f32 - delta

    return W_f32


def apply_multi_direction_erasure(
    W: torch.Tensor,
    directions: list[torch.Tensor],
    alpha: float = 1.0,
    decay_per_rank: float = 0.75,
) -> torch.Tensor:
    """
    Sequentially applies directional weight updates across a list of unit directions,
    scaling subsequent directions by `decay_per_rank ** i` to keep updates controlled.
    """
    updated = W.float().to(W.device)
    if updated.ndim != 2 or not directions:
        return updated

    for idx, vec in enumerate(directions):
        if vec is None or vec.numel() == 0:
            continue
        scaled_alpha = alpha * (decay_per_rank ** idx)
        updated = apply_erasure(updated, vec, w=vec, alpha=scaled_alpha)

    return updated


def preserve_row_norms(original_weight, updated_weight, eps: float = 1e-8):
    """Rescale rows of updated_weight to match original_weight row norms (MPOA-style)."""
    import torch

    orig_f = original_weight.float()
    upd_f = updated_weight.float()
    orig_norms = orig_f.norm(dim=-1, keepdim=True).clamp_min(eps)
    upd_norms = upd_f.norm(dim=-1, keepdim=True).clamp_min(eps)
    scaled = upd_f * (orig_norms / upd_norms)
    return scaled.to(original_weight.dtype)
