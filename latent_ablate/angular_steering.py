import logging
import math
import torch

logger = logging.getLogger("LatentAblate.AngularSteering")


class AngularSteeringEngine:
    @staticmethod
    def compute_rotation_matrix(
        r_refusal: torch.Tensor,
        v_target: torch.Tensor,
        theta_deg: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes planar rotation components in the 2D plane spanned by the refusal direction
        and a target compliance vector, allowing smooth angular steering.
        """
        theta = torch.tensor(math.radians(float(theta_deg)), dtype=torch.float32)
        r_f = r_refusal.float()
        v_f = v_target.float().to(r_f.device)
        r = torch.nn.functional.normalize(r_f, dim=0)
        v = torch.nn.functional.normalize(v_f - torch.dot(v_f, r) * r, dim=0)

        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        logger.info(f"Initialized Angular Rotation operator at angle theta = {theta_deg}°")
        return cos_t, sin_t, r, v
