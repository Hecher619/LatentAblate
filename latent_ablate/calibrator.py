import logging
import torch

logger = logging.getLogger("LatentAblate.Calibrator")


def estimate_free_vram_bytes(device: torch.device | str | None = None) -> int | None:
    """Return free GPU memory in bytes when available, or None on CPU-only runs."""
    if device is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    if dev.type == "cuda" and torch.cuda.is_available():
        try:
            idx = dev.index if dev.index is not None else torch.cuda.current_device()
            free_bytes, _total_bytes = torch.cuda.mem_get_info(idx)
            return int(free_bytes)
        except Exception:
            return None
    return None


def recommend_batch_size(
    model,
    device: torch.device | str | None = None,
    num_prompts: int = 16,
    default_batch_size: int = 8,
    max_batch_size: int = 32,
) -> int:
    """
    Pick a safe batch size based on free VRAM and model hidden dimension so
    small GPUs back off automatically and large-VRAM GPUs batch cleanly.
    """
    n_p = max(1, int(num_prompts))
    base_bs = max(1, min(int(default_batch_size), int(max_batch_size), n_p))
    free_bytes = estimate_free_vram_bytes(device)
    if free_bytes is None:
        return base_bs

    hidden = 1024
    try:
        cfg = getattr(model, "config", None)
        hidden = int(getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None) or 1024)
    except Exception:
        pass

    free_gib = free_bytes / (1024 ** 3)
    if free_gib < 2.0 or hidden >= 4096:
        chosen = 2
    elif free_gib < 5.0 or hidden >= 2560:
        chosen = 4
    elif free_gib < 12.0:
        chosen = 8
    else:
        chosen = min(max_batch_size, 16)

    return max(1, min(chosen, n_p))


class HardwareCalibrator:
    """Startup profiler for batch size and memory headroom."""

    def __init__(self, model, device=None):
        self.model = model
        self.device = device

    def calibrate_batch_size(self, num_prompts: int, default_batch_size: int = 8) -> int:
        bs = recommend_batch_size(
            self.model,
            device=self.device,
            num_prompts=num_prompts,
            default_batch_size=default_batch_size,
        )
        free_b = estimate_free_vram_bytes(self.device)
        if free_b is not None:
            logger.debug(f"Calibrated batch_size={bs} with {free_b / (1024**3):.2f} GiB free VRAM.")
        return bs
