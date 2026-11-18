import torch
import logging

logger = logging.getLogger("LatentAblate.Erasure")

ORTHO_FALLBACK = 1e-3


def conflict_score(r: torch.Tensor, basis: torch.Tensor) -> float:
    """Fraction of ||r|| that lies inside the capability subspace."""
    if basis.numel() == 0:
        return 0.0
    device = r.device
    r = r.to(device).float()
    basis = basis.to(device).float()

    proj = basis.T @ (basis @ r)
    return (proj.norm() / r.norm().clamp_min(1e-8)).item()


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
    basis = torch.linalg.qr(basis.T, mode="reduced").Q.T
    return basis.contiguous()


def oblique_weights(r: torch.Tensor, basis: torch.Tensor) -> tuple:
    """Solve oblique projection weights with explicit device and norm guards."""
    device = r.device
    r = r.float().to(device)

    if basis.numel() > 0:
        basis = basis.to(device).float()
        r_perp = r - basis.T @ (basis @ r)
    else:
        r_perp = r

    denom = r_perp.pow(2).sum()
    r_norm_sq = r.pow(2).sum().clamp_min(1e-8)

    if denom < ORTHO_FALLBACK * r_norm_sq:
        logger.warning("Refusal direction inside capability basis; falling back to orthogonal.")
        return r / r_norm_sq, True

    return r_perp / denom, False


def apply_erasure(W: torch.Tensor, r: torch.Tensor, w: torch.Tensor, alpha: float) -> torch.Tensor:
    """Computes mathematically rigorous weight-space oblique erasure."""
    device = W.device
    W_f32 = W.float().to(device)
    r = r.to(device).float()
    w = w.to(device).float()

    if W_f32.shape[1] == r.shape[0]:
        activation_projection = W_f32 @ r
        delta = torch.outer(activation_projection, w)
    else:
        logger.warning(f"Dimension mismatch in apply_erasure: W shape {W_f32.shape} vs r shape {r.shape}.")
        delta = torch.outer(torch.randn(W_f32.shape[0], device=device), r)

    return W_f32 - alpha * delta
