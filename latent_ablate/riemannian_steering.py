import torch
import logging

logger = logging.getLogger("LatentAblate.RiemannianSteering")


class RiemannianGeodesicEngine:
    """
    Computes exact spherical geodesic rotation / transport along tangent directions
    and regularized Gram metric inverses for subspace projection.
    """

    @staticmethod
    def spherical_geodesic_transport(h: torch.Tensor, tangent_vector: torch.Tensor, step_size: float) -> torch.Tensor:
        """
        Rotates each row vector of `h` in the 2D plane spanned by `h` and `tangent_vector`
        by angle `step_size` (radians), preserving exact row Euclidean norms.
        """
        h_f = h.float()
        row_norms = h_f.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        h_unit = h_f / row_norms

        v = torch.nn.functional.normalize(tangent_vector.float().to(h_f.device), dim=0)
        proj = (h_unit @ v).unsqueeze(-1)
        v_perp = v.unsqueeze(0) - proj * h_unit
        v_perp_norm = v_perp.norm(dim=-1, keepdim=True)
        safe_v_perp = torch.where(
            v_perp_norm > 1e-7,
            v_perp / v_perp_norm.clamp_min(1e-7),
            torch.zeros_like(v_perp),
        )

        angle = torch.tensor(float(step_size), dtype=h_f.dtype, device=h_f.device)
        rotated_unit = torch.cos(angle) * h_unit + torch.sin(angle) * safe_v_perp
        transported = rotated_unit * row_norms
        return transported.to(h.dtype)

    @staticmethod
    def parallel_transport(h: torch.Tensor, tangent_vector: torch.Tensor, step_size: float) -> torch.Tensor:
        """Backwards-compatible alias for norm-preserving spherical geodesic transport."""
        return RiemannianGeodesicEngine.spherical_geodesic_transport(h, tangent_vector, step_size)

    @staticmethod
    def regularized_gram_inverse(basis: torch.Tensor, ridge: float = 1e-5) -> torch.Tensor:
        """Computes (B B^T + ridge * I)^{-1} for oblique/non-orthonormal basis projection."""
        b = basis.float()
        if b.ndim == 1:
            b = b.unsqueeze(0)
        gram = b @ b.T
        eye = torch.eye(gram.shape[0], device=b.device, dtype=b.dtype)
        return torch.linalg.solve(gram + ridge * eye, eye)

    @staticmethod
    def compute_connection_coefficients(basis: torch.Tensor) -> torch.Tensor:
        """Backwards-compatible alias for regularized_gram_inverse."""
        return RiemannianGeodesicEngine.regularized_gram_inverse(basis)
