import torch
import logging

logger = logging.getLogger("LatentAblate.Calibrator")

def preserve_row_norms(modified: torch.Tensor, original: torch.Tensor, rows=None) -> torch.Tensor:
    """Preserves row norms of modified weight matrices with strict meta-tensor safety guards."""
    if modified is None or original is None:
        return modified

    # GUARD: If any tensor is a meta tensor, bypass calibration safely
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
        # Standard row norm preservation logic
        orig_norms = original.norm(dim=1, keepdim=True).clamp_min(1e-8)
        mod_norms = modified.norm(dim=1, keepdim=True).clamp_min(1e-8)
        return modified * (orig_norms / mod_norms)
    except Exception as e:
        logger.warning(f"Error in preserve_row_norms: {e}. Returning unmodified tensor.")
        return modified
