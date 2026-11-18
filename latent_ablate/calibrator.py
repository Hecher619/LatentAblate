import torch
import logging

logger = logging.getLogger("LatentAblate.Calibrator")


class NormCalibrator:
    @staticmethod
    def preserve_row_norms(modified: torch.Tensor, original: torch.Tensor, rows: torch.Tensor, eps: float = 1e-8):
        """
        Rescale ONLY the touched rows to their original norms.
        """
        if not rows.any():
            return
        idx = rows.nonzero(as_tuple=True)[0]
        old = original[idx].norm(dim=1)
        new = modified[idx].norm(dim=1)
        scale = (old / new.clamp_min(eps)).clamp(max=10.0)
        modified[idx] = modified[idx] * scale.unsqueeze(1)
