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
    """Compute unit ablation direction and dual projection vector preserving capability subspace."""
    if r.is_meta or (basis is not None and basis.numel() > 0 and basis.is_meta):
        r_f = r.float()
        r_unit = r_f / r_f.norm().clamp_min(1e-8)
        return r_unit, r_unit

    device = r.device
    if device.type == "meta":
        r_f = r.float()
        r_unit = r_f / r_f.norm().clamp_min(1e-8)
        return r_unit, r_unit

    r_f = r.float().to(device)
    r_unit = r_f / r_f.norm().clamp_min(1e-8)

    if basis is not None and basis.numel() > 0:
        b = basis.to(device).float()
        if b.ndim == 1:
            b = b.unsqueeze(0)
        # Oblique dual vector w satisfies <w, r_unit> = 1 and w orthogonal to capability basis
        r_perp = r_unit - b.T @ (b @ r_unit)
        denom = torch.dot(r_perp, r_unit)
        if denom.abs() < ORTHO_FALLBACK:
            return r_unit, r_unit
        w = r_perp / denom
        return r_unit, w

    return r_unit, r_unit


def apply_erasure(W: torch.Tensor, r: torch.Tensor, w: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Applies directional ablation along unit direction r (using dual vector w when provided)
    on a 2D linear weight matrix W of shape [out_features, in_features].
    """
    device = W.device
    W_f32 = W.float().to(device)
    r_f = r.to(device).float()
    r_unit = r_f / r_f.norm().clamp_min(1e-8)
    w_f = w.to(device).float() if w is not None else r_unit
    w_unit = w_f / w_f.norm().clamp_min(1e-8)

    if W_f32.ndim != 2:
        return W_f32

    out_features, in_features = W_f32.shape
    dim = r_unit.shape[0]

    # Case 1: Residual stream is the output space (e.g., o_proj, down_proj: [hidden, d_inner])
    # Project out r_unit from row space: W <- W - alpha * r_unit * (r_unit^T W)
    if out_features == dim:
        proj = torch.mv(W_f32.T, r_unit)  # shape [in_features]
        delta = alpha * torch.outer(r_unit, proj)
        return W_f32 - delta

    # Case 2: Residual stream is the input space (e.g., q_proj, k_proj, v_proj, gate_proj, up_proj: [d_proj, hidden])
    # Project out r_unit from column space: W <- W - alpha * (W r_unit) * r_unit^T
    if in_features == dim:
        proj = torch.mv(W_f32, r_unit)  # shape [out_features]
        delta = alpha * torch.outer(proj, w_unit if w_unit.shape[0] == in_features else r_unit)
        return W_f32 - delta

    logger.debug(f"Skipping weight {tuple(W_f32.shape)} with incompatible dim {dim}.")
    return W_f32
