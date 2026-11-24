import logging
import math
import torch

from latent_ablate.erasure import _is_meta, conflict_score
from latent_ablate.subspace_alignment import SubspaceAlignmentAnalyzer

logger = logging.getLogger("LatentAblate.SubspaceGuardrail")


class SubspaceDriftGuardrail:
    """Validates post-edit subspace alignment, capability separation, and conflict bounds."""

    def __init__(
        self,
        max_capability_overlap: float = 0.45,
        max_conflict_score: float = 0.65,
        max_temporal_drift: float = 0.85,
    ):
        self.max_capability_overlap = float(max_capability_overlap)
        self.max_conflict_score = float(max_conflict_score)
        self.max_temporal_drift = float(max_temporal_drift)

    def evaluate_layer(
        self,
        layer_idx: int,
        refusal_basis: torch.Tensor | None,
        capability_basis: torch.Tensor | None = None,
        prior_basis: torch.Tensor | None = None,
    ) -> dict:
        violations: list[str] = []
        overlap = 0.0
        c_score = 0.0
        drift = 0.0

        if (
            refusal_basis is not None
            and capability_basis is not None
            and not _is_meta(refusal_basis)
            and not _is_meta(capability_basis)
            and refusal_basis.numel() > 0
            and capability_basis.numel() > 0
        ):
            overlap = SubspaceAlignmentAnalyzer.subspace_overlap_score(refusal_basis, capability_basis)
            primary_r = refusal_basis[0] if refusal_basis.ndim >= 2 else refusal_basis
            raw_c = conflict_score(primary_r, capability_basis)
            c_score = float(raw_c) if math.isfinite(raw_c) else 0.0

            if overlap > self.max_capability_overlap:
                violations.append(
                    f"layer_{int(layer_idx)}: capability_overlap {overlap:.4f} > {self.max_capability_overlap:.4f}"
                )
            if c_score > self.max_conflict_score:
                violations.append(
                    f"layer_{int(layer_idx)}: conflict_score {c_score:.4f} > {self.max_conflict_score:.4f}"
                )

        if (
            refusal_basis is not None
            and prior_basis is not None
            and not _is_meta(refusal_basis)
            and not _is_meta(prior_basis)
            and refusal_basis.numel() > 0
            and prior_basis.numel() > 0
        ):
            prior_overlap = SubspaceAlignmentAnalyzer.subspace_overlap_score(refusal_basis, prior_basis)
            drift = max(0.0, min(1.0, 1.0 - prior_overlap))
            if drift > self.max_temporal_drift:
                violations.append(
                    f"layer_{int(layer_idx)}: temporal_drift {drift:.4f} > {self.max_temporal_drift:.4f}"
                )

        passed = len(violations) == 0
        return {
            "layer": int(layer_idx),
            "passed": passed,
            "capability_overlap": round(overlap, 6),
            "conflict_score": round(c_score, 6),
            "temporal_drift": round(drift, 6),
            "violations": violations,
        }

    def check_all(self, layer_entries: list[dict]) -> tuple[bool, list[str]]:
        all_violations: list[str] = []
        for item in layer_entries or []:
            res = self.evaluate_layer(
                layer_idx=int(item.get("layer", 0)),
                refusal_basis=item.get("refusal_basis"),
                capability_basis=item.get("capability_basis"),
                prior_basis=item.get("prior_basis"),
            )
            all_violations.extend(res["violations"])

        passed = len(all_violations) == 0
        if not passed:
            logger.warning(f"Subspace drift guardrail flagged {len(all_violations)} violation(s).")
        return passed, all_violations
