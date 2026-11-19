import torch
import logging

logger = logging.getLogger("LatentAblate.Calibrator")

def preserve_row_norms(modified: torch.Tensor, original: torch.Tensor, rows=None) -> torch.Tensor:
    """Preserves row norms of modified weight matrices with strict meta-tensor safety guards."""
    if modified is None or original is None:
        return modified

    if modified.is_meta or original.is_meta:
        return modified
    
    if modified.device.type == "meta" or original.device.type == "meta":
        return modified

    if rows is not None and isinstance(rows, torch.Tensor):
        if rows.is_meta or rows.device.type == "meta":
            return modified
        if rows.dtype == torch.bool:
            try:
                if not rows.any().item():
                    return modified
            except Exception:
                return modified

    try:
        orig_norms = original.norm(dim=1, keepdim=True).clamp_min(1e-8)
        mod_norms = modified.norm(dim=1, keepdim=True).clamp_min(1e-8)
        return modified * (orig_norms / mod_norms)
    except Exception as e:
        logger.warning(f"Error in preserve_row_norms: {e}. Returning unmodified tensor.")
        return modified


class NormCalibrator:
    """Class wrapper to satisfy surgeon.py imports."""
    @staticmethod
    def preserve_row_norms(modified: torch.Tensor, original: torch.Tensor, rows=None) -> torch.Tensor:
        return preserve_row_norms(modified, original, rows)


def suggest_eval_batch_size(device: str = "cpu", free_vram_gb: float | None = None, model_params_b: float = 1.5) -> int:
    """Suggest a safe evaluation/forward batch size based on device and model size."""
    dev = (device or "cpu").lower()
    if dev == "cpu":
        return 2 if model_params_b <= 3.0 else 1
    if free_vram_gb is None:
        return 4 if model_params_b <= 3.0 else 2
    if free_vram_gb >= 16.0 and model_params_b <= 4.0:
        return 16
    if free_vram_gb >= 8.0 and model_params_b <= 4.0:
        return 8
    if free_vram_gb >= 6.0:
        return 4
    return 2
