import logging
import torch
from latent_ablate.arch import eligible_linears, resolve_layers

logger = logging.getLogger("LatentAblate.Snapshot")


class LayerWeightSnapshot:
    """Captures and restores original linear weights across targeted transformer layers."""

    def __init__(self):
        self._snapshots: dict[int, dict[str, torch.Tensor]] = {}

    def captured_layers(self) -> list[int]:
        return sorted(self._snapshots.keys())

    def capture(self, model, layer_indices: list[int], store_on_cpu: bool = True, overwrite: bool = False):
        layers = resolve_layers(model)
        added = 0
        for idx in layer_indices:
            if idx < 0 or idx >= len(layers):
                continue
            if idx in self._snapshots and not overwrite:
                continue
            saved = {}
            for name, weight in eligible_linears(layers[idx]):
                if getattr(weight, "is_meta", False):
                    continue
                t = weight.detach().clone()
                saved[name] = t.cpu() if store_on_cpu else t
            self._snapshots[idx] = saved
            added += 1
        logger.info(f"Captured weight snapshot for {added} new layer(s) (total cached: {len(self._snapshots)}).")

    def restore(self, model, layer_indices: list[int] | None = None):
        layers = resolve_layers(model)
        indices = layer_indices if layer_indices is not None else list(self._snapshots.keys())
        restored_count = 0
        for idx in indices:
            saved = self._snapshots.get(idx)
            if not saved or idx < 0 or idx >= len(layers):
                continue
            for name, weight in eligible_linears(layers[idx]):
                if name in saved and not getattr(weight, "is_meta", False):
                    with torch.no_grad():
                        weight.copy_(saved[name].to(device=weight.device, dtype=weight.dtype))
            restored_count += 1
        logger.info(f"Restored weight snapshot for {restored_count} layer(s).")
