import torch
import logging
import math

logger = logging.getLogger("LatentAblate.InferenceSteer")

class DynamicSteeringManager:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.active_handles = []

    def register_steering_hook(self, layer, r_refusal: torch.Tensor, v_target: torch.Tensor, theta_deg: float = -30.0):
        """Applies runtime angular steering hook without modifying permanent weights."""
        r = torch.nn.functional.normalize(r_refusal.float().to(self.device), dim=0)
        v = v_target.float().to(self.device)
        v = v - torch.dot(v, r) * r
        if v.norm() > 1e-6:
            v = torch.nn.functional.normalize(v, dim=0)
        else:
            v = torch.randn_like(r)
            v = torch.nn.functional.normalize(v - torch.dot(v, r) * r, dim=0)

        theta = math.radians(theta_deg)
        cos_t, sin_t = math.cos(theta), math.sin(theta)

        def steering_hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            h_f = hidden.float()
            a = (h_f @ r).unsqueeze(-1)
            b = (h_f @ v).unsqueeze(-1)
            out = h_f + (cos_t - 1.0) * a * r + (cos_t - 1.0) * b * v + sin_t * (b * r - a * v)
            out = out.to(hidden.dtype)
            return (out,) + tuple(output[1:]) if isinstance(output, tuple) else out

        handle = layer.register_forward_hook(steering_hook)
        self.active_handles.append(handle)
        logger.info(f"Dynamic inference steering hook armed at angle theta={theta_deg}°.")

    def clear_hooks(self):
        for h in self.active_handles:
            h.remove()
        self.active_handles.clear()
        logger.info("All dynamic inference steering hooks cleared.")
