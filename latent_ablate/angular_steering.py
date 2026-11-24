import logging
import math
import torch
import torch.nn.functional as F

from latent_ablate.erasure import _is_meta

logger = logging.getLogger("LatentAblate.AngularSteering")


class AngularSteeringEngine:
    """Computes planar rotation components in the 2D subspace spanned by two vectors."""

    @staticmethod
    def compute_rotation_matrix(
        r_refusal: torch.Tensor,
        v_target: torch.Tensor,
        theta_deg: float,
        eps: float = 1e-8,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns `(cos_t, sin_t, r_unit, v_perp_unit)` spanning the 2D plane defined by
        `r_refusal` and `v_target`, rotated by `theta_deg` degrees.
        """
        if r_refusal is None or v_target is None or _is_meta(r_refusal) or _is_meta(v_target):
            raise ValueError("AngularSteeringEngine requires non-meta r_refusal and v_target tensors.")
        if r_refusal.numel() == 0 or v_target.numel() == 0 or r_refusal.shape != v_target.shape:
            raise ValueError("r_refusal and v_target must be non-empty tensors with matching shapes.")

        r_f = r_refusal.detach().float()
        v_f = v_target.detach().float().to(r_f.device)
        floor = max(1e-12, float(eps))

        r_norm = r_f.norm()
        if float(r_norm.item()) <= floor:
            raise ValueError("r_refusal norm is too close to zero to define a steering axis.")
        r_unit = (r_f / r_norm.clamp_min(floor)).contiguous()

        v_orth = v_f - torch.dot(v_f, r_unit) * r_unit
        v_orth_norm = v_orth.norm()
        if float(v_orth_norm.item()) <= 1e-6:
            # Fallback to a deterministic orthogonal vector if v_target is collinear with r_refusal.
            fallback = torch.zeros_like(r_unit)
            idx = int(torch.argmin(r_unit.abs()).item())
            fallback[idx] = 1.0
            v_orth = fallback - torch.dot(fallback, r_unit) * r_unit
            v_orth_norm = v_orth.norm().clamp_min(floor)

        v_unit = (v_orth / v_orth_norm.clamp_min(floor)).contiguous()

        theta_rad = math.radians(float(theta_deg))
        cos_t = torch.tensor(math.cos(theta_rad), device=r_unit.device, dtype=torch.float32)
        sin_t = torch.tensor(math.sin(theta_rad), device=r_unit.device, dtype=torch.float32)

        logger.debug(f"Computed planar rotation components at theta={float(theta_deg):.2f} deg.")
        return cos_t, sin_t, r_unit, v_unit

    @classmethod
    def rotate_vector_in_plane(
        cls,
        x: torch.Tensor,
        r_refusal: torch.Tensor,
        v_target: torch.Tensor,
        theta_deg: float,
    ) -> torch.Tensor:
        """Rotates `x` (1D or 2D rows) within the `(r_unit, v_unit)` plane by `theta_deg`."""
        if x is None or _is_meta(x) or x.numel() == 0:
            return x

        cos_t, sin_t, r_u, v_u = cls.compute_rotation_matrix(r_refusal, v_target, theta_deg)
        x_f = x.detach().float().to(r_u.device)
        c = cos_t.to(x_f.device)
        s = sin_t.to(x_f.device)

        if x_f.ndim == 1:
            a = torch.dot(x_f, r_u)
            b = torch.dot(x_f, v_u)
            out = x_f + (c - 1.0) * (a * r_u + b * v_u) + s * (b * r_u - a * v_u)
            return F.normalize(out, p=2, dim=0).mul(x_f.norm()).to(device=x.device, dtype=x.dtype)

        a = (x_f @ r_u).unsqueeze(-1)
        b = (x_f @ v_u).unsqueeze(-1)
        out = x_f + (c - 1.0) * (a * r_u.unsqueeze(0) + b * v_u.unsqueeze(0)) + s * (b * r_u.unsqueeze(0) - a * v_u.unsqueeze(0))
        return out.to(device=x.device, dtype=x.dtype)
