import logging
import math
import torch

from latent_ablate.erasure import _is_meta, conflict_score
from latent_ablate.reaction import format_velora_telemetry
from latent_ablate.subspace_alignment import SubspaceAlignmentAnalyzer

logger = logging.getLogger("LatentAblate.SubspaceTelemetry")


class SubspaceTelemetryTracker:
    """Collects per-layer subspace alignment, orthogonality, and conflict metrics."""

    def __init__(self, version: str = "VELORA-v2.2"):
        self.version = str(version or "VELORA-v2.2")
        self.records: list[dict] = []

    def record_layer_metrics(
        self,
        layer_idx: int,
        refusal_basis: torch.Tensor | None,
        capability_basis: torch.Tensor | None = None,
        prior_basis: torch.Tensor | None = None,
        mode: str = "velora_v2_2_strategic",
        reaction: str = "COMPLIANT",
    ) -> dict:
        overlap_cap = 0.0
        c_score = 0.0
        if (
            refusal_basis is not None
            and capability_basis is not None
            and not _is_meta(refusal_basis)
            and not _is_meta(capability_basis)
            and refusal_basis.numel() > 0
            and capability_basis.numel() > 0
        ):
            overlap_cap = SubspaceAlignmentAnalyzer.subspace_overlap_score(refusal_basis, capability_basis)
            primary_vec = refusal_basis[0] if refusal_basis.ndim >= 2 else refusal_basis
            raw_c = conflict_score(primary_vec, capability_basis)
            if math.isfinite(raw_c):
                c_score = round(float(raw_c), 6)

        temporal_drift = 0.0
        if (
            refusal_basis is not None
            and prior_basis is not None
            and not _is_meta(refusal_basis)
            and not _is_meta(prior_basis)
            and refusal_basis.numel() > 0
            and prior_basis.numel() > 0
        ):
            prior_overlap = SubspaceAlignmentAnalyzer.subspace_overlap_score(refusal_basis, prior_basis)
            temporal_drift = round(max(0.0, min(1.0, 1.0 - prior_overlap)), 6)

        rank_val = 0
        if refusal_basis is not None and not _is_meta(refusal_basis) and refusal_basis.numel() > 0:
            rank_val = int(refusal_basis.shape[0]) if refusal_basis.ndim >= 2 else 1

        telemetry_line = format_velora_telemetry(
            active=True,
            mode=mode,
            reaction=reaction,
            layers=str(int(layer_idx)),
            extra=f"cap_overlap={overlap_cap:.4f} drift={temporal_drift:.4f} conflict={c_score:.4f}",
        )

        entry = {
            "version": self.version,
            "layer": int(layer_idx),
            "rank": rank_val,
            "capability_overlap": overlap_cap,
            "temporal_drift": temporal_drift,
            "conflict_score": c_score,
            "telemetry": telemetry_line,
        }
        self.records.append(entry)
        logger.debug(telemetry_line)
        return entry

    def summarize(self) -> dict:
        if not self.records:
            return {
                "version": self.version,
                "num_records": 0,
                "mean_capability_overlap": 0.0,
                "mean_temporal_drift": 0.0,
                "max_conflict_score": 0.0,
            }

        n = len(self.records)
        mean_overlap = sum(float(r["capability_overlap"]) for r in self.records) / n
        mean_drift = sum(float(r["temporal_drift"]) for r in self.records) / n
        max_conflict = max(float(r["conflict_score"]) for r in self.records)

        return {
            "version": self.version,
            "num_records": n,
            "mean_capability_overlap": round(mean_overlap, 6),
            "mean_temporal_drift": round(mean_drift, 6),
            "max_conflict_score": round(max_conflict, 6),
        }

    def attach_to_reporter(self, reporter, step_name: str = "subspace_telemetry") -> None:
        if reporter is None or not hasattr(reporter, "log_step"):
            return
        reporter.log_step(
            step_name,
            {
                "summary": self.summarize(),
                "layers": self.records,
            },
        )

    def clear(self) -> None:
        self.records.clear()
