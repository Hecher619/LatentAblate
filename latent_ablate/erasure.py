import torch
import logging

logger = logging.getLogger("LatentAblate.Erasure")

ORTHO_FALLBACK = 1e-6


def _is_meta(tensor: torch.Tensor | None) -> bool:
    if tensor is None:
        return False
    if getattr(tensor, "is_meta", False):
        return True
    dev = getattr(tensor, "device", None)
    return dev is not None and getattr(dev, "type", None) == "meta"


def conflict_score(r: torch.Tensor, basis: torch.Tensor) -> float:
    """Fraction of ||r||_2 that lies inside the capability subspace span(basis)."""
    if basis is None or basis.numel() == 0 or _is_meta(r) or _is_meta(basis):
        return 0.0
    device = r.device
    r = r.to(device).float()
    r_unit = r / r.norm().clamp_min(1e-8)
    basis = basis.to(device).float()
    if basis.ndim == 1:
        basis = basis.unsqueeze(0)
    proj = basis.T @ (basis @ r_unit)
    out = proj.norm() / r_unit.norm().clamp_min(1e-8)
    if _is_meta(out):
        return 0.0
    return float(out.item())


def explained_variance_ratio(activations: torch.Tensor, rank: int) -> float:
    """Fraction of centered activation variance captured by the top `rank` singular vectors."""
    if _is_meta(activations):
        return 0.0
    acts = activations.float()
    if acts.shape[0] < 2:
        return 1.0
    centered = acts - acts.mean(dim=0, keepdim=True)
    svals = torch.linalg.svdvals(centered)
    total_energy = (svals ** 2).sum().clamp_min(1e-12)
    k = max(1, min(rank, svals.shape[0]))
    captured = (svals[:k] ** 2).sum()
    ratio = captured / total_energy
    if _is_meta(ratio):
        return 0.0
    return float(ratio.item())


def select_energy_rank(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    max_rank: int = 8,
    target_energy: float = 0.85,
) -> int:
    """Selects the smallest rank k in [1, max_rank] reaching `target_energy` of centered contrast variance."""
    cap_max = max(1, int(max_rank))
    if pos_activations is None or neg_activations is None or _is_meta(pos_activations) or _is_meta(neg_activations):
        return 1
    n_pairs = min(pos_activations.shape[0], neg_activations.shape[0])
    if n_pairs < 2:
        return 1

    diff = pos_activations[:n_pairs].float() - neg_activations[:n_pairs].float()
    centered = diff - diff.mean(dim=0, keepdim=True)
    svals = torch.linalg.svdvals(centered)
    if svals.numel() == 0 or _is_meta(svals):
        return 1

    energies = svals ** 2
    total = energies.sum()
    if _is_meta(total) or total <= 1e-12:
        return 1

    cum_ratio = torch.cumsum(energies, dim=0) / total
    upper = min(cap_max, cum_ratio.shape[0])
    for idx in range(upper):
        item_val = cum_ratio[idx]
        if not _is_meta(item_val) and float(item_val.item()) >= float(target_energy):
            return idx + 1
    return upper


def conflict_scaled_alpha(
    base_alpha: float,
    c_score: float,
    sensitivity: float = 0.25,
    floor: float = 0.55,
) -> float:
    """Scales `base_alpha` down smoothly when refusal overlaps with capability subspace (`c_score` in [0, 1])."""
    clamped_conflict = max(0.0, min(1.0, float(c_score)))
    scale = max(float(floor), 1.0 - float(sensitivity) * clamped_conflict)
    return float(base_alpha) * scale


def build_capability_basis(activations: torch.Tensor, rank: int = 16) -> torch.Tensor:
    """Sign-deterministic orthonormal basis of dominant representation directions on capability prompts."""
    if _is_meta(activations):
        hidden = activations.shape[-1] if activations.ndim >= 1 else 0
        return torch.empty(0, hidden, dtype=torch.float32)

    acts = activations.float()
    if acts.shape[0] < 2:
        raise ValueError("Need >= 2 capability activations to build a basis.")

    acts = acts - acts.mean(dim=0, keepdim=True)
    device = acts.device

    try:
        _, svals, vh = torch.linalg.svd(acts, full_matrices=False)
    except RuntimeError:
        logger.warning("SVD failed on device, falling back to CPU for stable computation.")
        _, svals, vh = torch.linalg.svd(acts.cpu(), full_matrices=False)
        svals, vh = svals.to(device), vh.to(device)

    k = max(1, min(rank, vh.shape[0]))
    total_energy = (svals ** 2).sum().clamp_min(1e-12)
    ratio_tensor = (svals[:k] ** 2).sum() / total_energy
    if not _is_meta(ratio_tensor):
        captured_ratio = float(ratio_tensor.item())
        if captured_ratio < 0.70:
            logger.debug(
                f"Capability basis rank={k} captures {captured_ratio * 100:.1f}% of variance; consider increasing rank."
            )

    basis = vh[:k].to(device)
    if basis.shape[0] > 0:
        q, r_mat = torch.linalg.qr(basis.T, mode="reduced")
        diag_signs = torch.sign(torch.diag(r_mat))
        diag_signs = torch.where(diag_signs == 0, torch.ones_like(diag_signs), diag_signs)
        basis = (q * diag_signs.unsqueeze(0)).T
    return basis.contiguous()


def oblique_weights(r: torch.Tensor, basis: torch.Tensor) -> tuple:
    """Return unit refusal direction and its unit orthogonal projection onto span(basis)^perp."""
    r_f = r.float()
    if _is_meta(r):
        r_unit = r_f / r_f.norm().clamp_min(1e-8)
        return r_unit, r_unit

    device = r.device
    r_f = r_f.to(device)
    r_unit = r_f / r_f.norm().clamp_min(1e-8)

    if basis is not None and basis.numel() > 0 and not _is_meta(basis):
        b = basis.to(device).float()
        if b.ndim == 1:
            b = b.unsqueeze(0)
        if b.shape[0] > 0:
            q, _ = torch.linalg.qr(b.T, mode="reduced")
            b = q.T
        r_perp = r_unit - b.T @ (b @ r_unit)
        norm_perp = r_perp.norm()
        if not _is_meta(norm_perp) and float(norm_perp.item()) > ORTHO_FALLBACK:
            return r_unit, (r_perp / norm_perp).contiguous()

    return r_unit, r_unit


def apply_erasure(
    W: torch.Tensor,
    r: torch.Tensor,
    w: torch.Tensor | None = None,
    alpha: float = 1.0,
) -> torch.Tensor:
    """
    Applies rank-one directional ablation along unit direction `w` (falling back to `r`)
    on a 2D linear weight matrix W [out_features, in_features].
    """
    if _is_meta(W) or _is_meta(r) or _is_meta(w):
        return W

    device = W.device
    W_f32 = W.float().to(device)
    if W_f32.ndim != 2:
        return W_f32

    use_w = w is not None and w.numel() > 0 and not _is_meta(w) and float(w.norm().item()) > 1e-8
    target_vec = w if use_w else r
    u = target_vec.to(device).float()
    u = u / u.norm().clamp_min(1e-8)

    out_features, in_features = W_f32.shape
    dim = u.shape[0]

    if out_features == dim:
        proj = torch.mv(W_f32.T, u)
        delta = alpha * torch.outer(u, proj)
        return W_f32 - delta

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
    """Sequentially applies rank-one directional updates with geometric rank decay."""
    if _is_meta(W):
        return W
    updated = W.float().to(W.device)
    if updated.ndim != 2 or not directions:
        return updated

    for idx, vec in enumerate(directions):
        if vec is None or vec.numel() == 0 or _is_meta(vec):
            continue
        scaled_alpha = alpha * (decay_per_rank ** idx)
        updated = apply_erasure(updated, vec, w=vec, alpha=scaled_alpha)

    return updated


def preserve_row_norms(original_weight: torch.Tensor, updated_weight: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Rescale rows of updated_weight to match original_weight row L2 norms."""
    if _is_meta(original_weight) or _is_meta(updated_weight):
        return updated_weight
    orig_f = original_weight.float()
    upd_f = updated_weight.float()
    orig_norms = orig_f.norm(dim=-1, keepdim=True).clamp_min(eps)
    upd_norms = upd_f.norm(dim=-1, keepdim=True).clamp_min(eps)
    scaled = upd_f * (orig_norms / upd_norms)
    return scaled.to(original_weight.dtype)
