import logging
import torch

from latent_ablate.arch import eligible_linears, resolve_layers
from latent_ablate.erasure import _is_meta

logger = logging.getLogger("LatentAblate.Snapshot")


class LayerWeightSnapshot:
    """Captures and restores 2D linear layer weights for selected transformer layers."""

    def __init__(self):
        self.snapshots: dict[int, dict[str, torch.Tensor]] = {}
        self.stored_on_cpu: bool = True

    def clear(self) -> None:
        self.snapshots.clear()

    def capture(
        self,
        model,
        layer_indices: list[int] | tuple[int, ...] | set[int] | None = None,
        store_on_cpu: bool = True,
    ) -> dict[int, dict[str, torch.Tensor]]:
        self.snapshots.clear()
        self.stored_on_cpu = bool(store_on_cpu)
        layers = resolve_layers(model)
        if not layers:
            return self.snapshots

        if layer_indices is None:
            selected = list(range(len(layers)))
        else:
            selected = sorted({int(i) for i in layer_indices if 0 <= int(i) < len(layers)})

        for idx in selected:
            layer_map: dict[str, torch.Tensor] = {}
            for name, weight in eligible_linears(layers[idx]):
                if not isinstance(weight, torch.Tensor) or _is_meta(weight) or weight.ndim != 2:
                    continue
                cloned = weight.detach().clone()
                if self.stored_on_cpu and cloned.device.type != "cpu":
                    cloned = cloned.cpu()
                layer_map[name] = cloned
            if layer_map:
                self.snapshots[idx] = layer_map

        logger.debug(f"Captured weight snapshot for {len(self.snapshots)} layer(s) (on_cpu={self.stored_on_cpu}).")
        return self.snapshots

    def restore(
        self,
        model,
        layer_indices: list[int] | tuple[int, ...] | set[int] | None = None,
    ) -> int:
        if not self.snapshots:
            return 0

        layers = resolve_layers(model)
        if not layers:
            return 0

        if layer_indices is None:
            target_indices = sorted(self.snapshots.keys())
        else:
            target_indices = [int(i) for i in layer_indices if int(i) in self.snapshots and 0 <= int(i) < len(layers)]

        restored_layers = 0
        with torch.no_grad():
            for idx in target_indices:
                saved_map = self.snapshots.get(idx)
                if not saved_map:
                    continue
                layer_touched = False
                for name, weight in eligible_linears(layers[idx]):
                    if not isinstance(weight, torch.Tensor) or _is_meta(weight) or weight.ndim != 2:
                        continue
                    saved = saved_map.get(name)
                    if saved is None or _is_meta(saved) or saved.shape != weight.shape:
                        continue
                    weight.copy_(saved.to(device=weight.device, dtype=weight.dtype))
                    layer_touched = True
                if layer_touched:
                    restored_layers += 1

        return restored_layers
