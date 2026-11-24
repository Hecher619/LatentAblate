import logging
import torch
import torch.nn.functional as F

from latent_ablate.erasure import _is_meta

logger = logging.getLogger("LatentAblate.RiemannianSteering")


class RiemannianGeodesicEngine:
    """
    Computes spherical geodesic transport along tangent directions and
    regularized Gram metric inverses for subspace projection.
    """

    @staticmethod
    def spherical_geodesic_transport(
        h: torch.Tensor,
        tangent_vector: torch.Tensor,
        step_size: float,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Rotates `h` (1D vector or 2D row matrix) in the 2D plane spanned by `h`
        and `tangent_vector` by `step_size` radians while preserving row Euclidean norms.
        """
        if h is None or _is_meta(h) or h.numel() == 0:
            return h
        if tangent_vector is None or _is_meta(tangent_vector) or tangent_vector.numel() == 0:
            return h

        h_f = h.detach().float()
        was_1d = h_f.ndim == 1
        if was_1d:
            h_f = h_f.unsqueeze(0)

        floor = max(1e-12, float(eps))
        row_norms = h_f.norm(dim=-1, keepdim=True).clamp_min(floor)
        h_unit = h_f / row_norms

        t_f = tangent_vector.detach().float().to(h_f.device)
        if t_f.ndim > 1:
            t_f = t_f.reshape(-1)
        if t_f.shape[0] != h_f.shape[-1]:
            return h

        t_norm = t_f.norm()
        if float(t_norm.item()) <= floor:
            return h
        v = F.normalize(t_f, p=2, dim=0, eps=floor)

        proj = (h_unit @ v).unsqueeze(-1)
        v_perp = v.unsqueeze(0) - proj * h_unit
        v_perp_norm = v_perp.norm(dim=-1, keepdim=True)
        safe_v_perp = torch.where(
            v_perp_norm > 1e-7,
            v_perp / v_perp_norm.clamp_min(1e-7),
            torch.zeros_like(v_perp),
        )

        angle = torch.tensor(float(step_size), dtype=torch.float32, device=h_f.device)
        rotated_unit = torch.cos(angle) * h_unit + torch.sin(angle) * safe_v_perp
        rotated_unit = F.normalize(rotated_unit, p=2, dim=-1, eps=floor)
        transported = rotated_unit * row_norms

        if was_1d:
            transported = transported.squeeze(0)
        return transported.to(device=h.device, dtype=h.dtype)

    @staticmethod
    def parallel_transport(
        h: torch.Tensor,
        tangent_vector: torch.Tensor,
        step_size: float,
    ) -> torch.Tensor:
        """Backwards-compatible alias for norm-preserving spherical geodesic transport."""
        return RiemannianGeodesicEngine.spherical_geodesic_transport(h, tangent_vector, step_size)

    @staticmethod
    def regularized_gram_inverse(basis: torch.Tensor, ridge: float = 1e-5) -> torch.Tensor:
        """Computes (B B^T + ridge * I)^{-1} for oblique/non-orthonormal basis projection."""
        if basis is None or _is_meta(basis) or basis.numel() == 0:
            return torch.empty(0, dtype=torch.float32)

        b = basis.detach().float()
        if b.ndim == 1:
            b = b.unsqueeze(0)

        gram = b @ b.T
        reg = max(1e-12, float(ridge))
        eye = torch.eye(gram.shape[0], device=b.device, dtype=b.dtype)
        try:
            return torch.linalg.solve(gram + reg * eye, eye)
        except RuntimeError:
            sol = torch.linalg.solve(gram.cpu() + reg * eye.cpu(), eye.cpu())
            return sol.to(device=b.device, dtype=b.dtype)

    @staticmethod
    def compute_connection_coefficients(basis: torch.Tensor, ridge: float = 1e-5) -> torch.Tensor:
        """Backwards-compatible alias for regularized_gram_inverse."""
        return RiemannianGeodesicEngine.regularized_gram_inverse(basis, ridge=ridge)
