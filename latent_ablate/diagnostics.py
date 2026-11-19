import logging
import torch
from latent_ablate.erasure import conflict_score

logger = logging.getLogger("LatentAblate.Diagnostics")


def measure_layer_delta(before_weight: torch.Tensor, after_weight: torch.Tensor) -> dict:
    b = before_weight.detach().float()
    a = after_weight.detach().float()
    diff_norm = (a - b).norm().item()
    base_norm = b.norm().clamp_min(1e-8).item()
    return {
        "delta_frobenius": round(diff_norm, 6),
        "relative_change": round(diff_norm / base_norm, 6),
    }


def summarize_subspace_conflict(direction: torch.Tensor, capability_basis: torch.Tensor | None) -> dict:
    if capability_basis is None or capability_basis.numel() == 0:
        return {"capability_conflict": 0.0, "has_capability_basis": False}
    score = conflict_score(direction, capability_basis)
    return {
        "capability_conflict": round(score, 4),
        "has_capability_basis": True,
    }


class ModelDiagnostics:
    """Tracks weight drift and subspace conflict metrics across ablation steps."""

    def __init__(self):
        self.records = []

    def record_delta(self, label: str, before_weight: torch.Tensor, after_weight: torch.Tensor) -> dict:
        entry = {"label": label, **measure_layer_delta(before_weight, after_weight)}
        self.records.append(entry)
        return entry

    def record_conflict(self, label: str, direction: torch.Tensor, capability_basis: torch.Tensor | None = None) -> dict:
        entry = {"label": label, **summarize_subspace_conflict(direction, capability_basis)}
        self.records.append(entry)
        return entry
