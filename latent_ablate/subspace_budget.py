import logging
import math
import torch

from latent_ablate.erasure import _is_meta, conflict_scaled_alpha, conflict_score

logger = logging.getLogger("LatentAblate.SubspaceBudget")


class SubspaceBudgetAllocator:
    """Allocates bounded per-layer steering budgets based on conflict and capability overlap."""

    def __init__(
        self,
        max_total_budget: float = 12.0,
        min_layer_scale: float = 0.15,
        max_layer_scale: float = 1.50,
    ):
        self.max_total_budget = max(0.1, float(max_total_budget))
        self.min_layer_scale = max(0.0, float(min_layer_scale))
        self.max_layer_scale = max(self.min_layer_scale, float(max_layer_scale))

    def allocate_layer_scales(
        self,
        layer_indices: list[int],
        base_alpha: float,
        layer_weights: dict[int, float] | None = None,
        refusal_directions: dict[int, torch.Tensor] | None = None,
        capability_bases: dict[int, torch.Tensor] | None = None,
    ) -> dict[int, float]:
        if not layer_indices:
            return {}

        alpha_f = float(base_alpha)
        if math.isnan(alpha_f) or math.isinf(alpha_f) or alpha_f <= 0.0:
            return {int(idx): 0.0 for idx in layer_indices}

        weights_map = layer_weights or {}
        dirs_map = refusal_directions or {}
        caps_map = capability_bases or {}

        raw_scales: dict[int, float] = {}
        for raw_idx in layer_indices:
            idx = int(raw_idx)
            k_w = float(weights_map.get(idx, 1.0))
            if math.isnan(k_w) or math.isinf(k_w):
                k_w = 1.0
            k_w = max(0.0, k_w)

            r_vec = dirs_map.get(idx)
            c_bas = caps_map.get(idx)
            c_score = 0.0
            if (
                r_vec is not None
                and c_bas is not None
                and not _is_meta(r_vec)
                and not _is_meta(c_bas)
                and r_vec.numel() > 0
                and c_bas.numel() > 0
            ):
                primary_r = r_vec[0] if r_vec.ndim >= 2 else r_vec
                raw_c = conflict_score(primary_r, c_bas)
                if math.isfinite(raw_c):
                    c_score = max(0.0, float(raw_c))

            eff = conflict_scaled_alpha(alpha_f * k_w, c_score)
            if math.isnan(eff) or math.isinf(eff):
                eff = 0.0
            clamped = max(self.min_layer_scale * alpha_f, min(self.max_layer_scale * alpha_f, eff))
            raw_scales[idx] = clamped

        total_allocated = sum(raw_scales.values())
        if total_allocated > self.max_total_budget and total_allocated > 0.0:
            ratio = self.max_total_budget / total_allocated
            for idx in raw_scales:
                raw_scales[idx] = round(raw_scales[idx] * ratio, 6)
        else:
            for idx in raw_scales:
                raw_scales[idx] = round(raw_scales[idx], 6)

        logger.debug(
            f"Allocated subspace scales across {len(raw_scales)} layer(s) "
            f"(total={sum(raw_scales.values()):.4f}, cap={self.max_total_budget:.4f})."
        )
        return raw_scales
