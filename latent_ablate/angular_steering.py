import torch
import logging

logger = logging.getLogger("LatentAblate.AngularSteering")

class AngularSteeringEngine:
    @staticmethod
    def compute_rotation_matrix(r_refusal: torch.Tensor, v_target: torch.Tensor, theta_deg: float) -> torch.Tensor:
        """
        Computes a 2D rotation matrix in the plane spanned by the refusal direction 
        and a target compliance vector, allowing smooth angular steering.
        """
        theta = torch.tensor(theta_deg * (3.1415926535 / 180.0), dtype=torch.float32)
        r = torch.nn.functional.normalize(r_refusal.float(), dim=0)
        v = torch.nn.functional.normalize(v_target.float() - torch.dot(v_target, r) * r, dim=0)

        # Construct orthogonal 2D subspace rotation operator
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        
        logger.info(f"Initialized Angular Rotation operator at angle theta = {theta_deg}°")
        return cos_t, sin_t, r, v
