import logging
import math
import torch
import torch.nn.functional as F

from latent_ablate.angular_steering import AngularSteeringEngine
from latent_ablate.erasure import _is_meta
from latent_ablate.orthogonal_subspace import OrthogonalSubspaceProjector
from latent_ablate.riemannian_steering import RiemannianGeodesicEngine

logger = logging.getLogger("LatentAblate.ManifoldSteering")


class ManifoldSteeringController:
    """Applies capability-guarded geodesic and angular steering shifts to latent states or weights."""

    @staticmethod
    def steer_activations(
        activations: torch.Tensor,
        steering_direction: torch.Tensor,
        step_radians: float = 0.15,
        capability_basis: torch.Tensor | None = None,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Steers `activations` along `steering_direction` via spherical geodesic transport,
        projecting out any component aligned with `capability_basis` when provided.
        """
        if activations is None or _is_meta(activations) or activations.numel() == 0:
            return activations
        if steering_direction is None or _is_meta(steering_direction) or steering_direction.numel() == 0:
            return activations

        angle = float(step_radians)
        if math.isnan(angle) or math.isinf(angle) or abs(angle) <= 1e-12:
            return activations

        act_f = activations.detach().float()
        if not torch.isfinite(act_f).all():
            return activations

        dir_f = steering_direction.detach().float().to(act_f.device).reshape(-1)
        if dir_f.shape[0] != act_f.shape[-1]:
            return activations

        if capability_basis is not None and not _is_meta(capability_basis) and capability_basis.numel() > 0:
            dir_f = OrthogonalSubspaceProjector.project_out_orthogonal_subspace(
                dir_f, capability_basis.to(act_f.device), scale=1.0, eps=eps
            )

        floor = max(1e-12, float(eps))
        dir_norm = float(dir_f.norm().item())
        if dir_norm <= floor or math.isnan(dir_norm) or math.isinf(dir_norm):
            return activations

        unit_dir = (dir_f / dir_norm).contiguous()
        transported = RiemannianGeodesicEngine.spherical_geodesic_transport(
            act_f, unit_dir, step_size=angle, eps=floor
        )
        if transported is None or _is_meta(transported) or not torch.isfinite(transported.float()).all():
            return activations

        return transported.to(device=activations.device, dtype=activations.dtype)

    @staticmethod
    def steer_weight_rows(
        weight: torch.Tensor,
        source_axis: torch.Tensor,
        target_axis: torch.Tensor,
        theta_deg: float = 15.0,
        row_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Rotates rows of a 2D weight matrix within the `(source_axis, target_axis)` plane
        by `theta_deg`, optionally restricted to boolean/index `row_mask`.
        """
        if weight is None or _is_meta(weight) or weight.ndim != 2 or weight.numel() == 0:
            return weight
        if source_axis is None or target_axis is None or _is_meta(source_axis) or _is_meta(target_axis):
            return weight

        angle = float(theta_deg)
        if math.isnan(angle) or math.isinf(angle) or abs(angle) <= 1e-12:
            return weight

        w_f = weight.detach().float()
        if not torch.isfinite(w_f).all():
            return weight

        src_1d = source_axis.detach().float().to(w_f.device).reshape(-1)
        tgt_1d = target_axis.detach().float().to(w_f.device).reshape(-1)
        if src_1d.shape[0] != w_f.shape[1] or tgt_1d.shape[0] != w_f.shape[1]:
            return weight

        try:
            rotated = AngularSteeringEngine.rotate_vector_in_plane(w_f, src_1d, tgt_1d, angle)
        except ValueError as e:
            logger.debug(f"Skipping planar weight rotation: {e}")
            return weight

        if rotated is None or _is_meta(rotated) or not torch.isfinite(rotated.float()).all():
            return weight

        rot_f = rotated.detach().float().to(w_f.device)
        if row_mask is not None and not _is_meta(row_mask) and row_mask.numel() > 0:
            if row_mask.dtype == torch.bool and row_mask.shape[0] == w_f.shape[0]:
                mask_col = row_mask.to(w_f.device).unsqueeze(-1)
                merged = torch.where(mask_col, rot_f, w_f)
            else:
                valid_rows = [
                    int(i)
                    for i in row_mask.detach().reshape(-1).tolist()
                    if 0 <= int(i) < w_f.shape[0]
                ]
                if not valid_rows:
                    return weight
                merged = w_f.clone()
                idx_t = torch.tensor(valid_rows, device=w_f.device, dtype=torch.long)
                merged[idx_t] = rot_f[idx_t]
            # Preserve per-row norm after selective row blending
            norms = w_f.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            merged = F.normalize(merged, p=2, dim=-1, eps=1e-8) * norms
            return merged.to(device=weight.device, dtype=weight.dtype)

        return rot_f.to(device=weight.device, dtype=weight.dtype)
