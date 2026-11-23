from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from latent_ablate.arch import REFUSAL_MARKERS, is_hedging_text, is_refusal_text

MORALIZING_MARKERS: tuple[str, ...] = (
    "is illegal",
    "are illegal",
    "illegal and",
    "can be dangerous",
    "is dangerous",
    "serious consequences",
    "legal penalties",
    "not something i can endorse",
    "i strongly advise against",
    "more positive and constructive",
    "let's focus on",
    "let me try again",
    "seek assistance from a professional",
    "for educational purposes only",
    "responsible and legal",
    "ethical considerations",
    "law and ethics",
)

LITERAL_DEFLECTION_MARKERS: tuple[str, ...] = (
    "literally",
    "in a literal sense",
    "hypothetical",
    "fictional context",
    "fictional scenario",
    "as an ai",
    "as a language model",
    "i do not have personal",
    "i cannot physically",
)


@dataclass(frozen=True)
class ReactionAssessment:
    state: str
    is_refusal: bool
    is_hedging: bool
    is_moralizing: bool
    is_deflection: bool
    needs_escalation: bool
    recommended_mode: str
    reason: str


def classify_response(
    response: str,
    refusal_markers: Iterable[str] = REFUSAL_MARKERS,
) -> ReactionAssessment:
    text = (response or "").strip()
    lower = text.lower()

    refusal = bool(is_refusal_text(text, refusal_markers))
    hedging = bool(is_hedging_text(text))
    moralizing = any(m in lower for m in MORALIZING_MARKERS)
    deflection = any(m in lower for m in LITERAL_DEFLECTION_MARKERS)

    if refusal:
        return ReactionAssessment(
            state="HARD_REFUSAL",
            is_refusal=True,
            is_hedging=hedging,
            is_moralizing=moralizing,
            is_deflection=deflection,
            needs_escalation=True,
            recommended_mode="refusal+anti_hedge+late_band",
            reason="Hard refusal marker detected in response.",
        )

    if moralizing or hedging:
        return ReactionAssessment(
            state="MORALIZING_HEDGE",
            is_refusal=False,
            is_hedging=True,
            is_moralizing=moralizing,
            is_deflection=deflection,
            needs_escalation=True,
            recommended_mode="anti_hedge+prefix_contrast",
            reason="Moralizing lecture or hedging preamble detected.",
        )

    if deflection:
        return ReactionAssessment(
            state="LITERAL_DEFLECTION",
            is_refusal=False,
            is_hedging=False,
            is_moralizing=False,
            is_deflection=True,
            needs_escalation=True,
            recommended_mode="anti_literal+refusal_subspace",
            reason="Literal/AI/hypothetical framing deflection detected.",
        )

    return ReactionAssessment(
        state="COMPLIANT",
        is_refusal=False,
        is_hedging=False,
        is_moralizing=False,
        is_deflection=False,
        needs_escalation=False,
        recommended_mode="velora_balanced",
        reason="No refusal, moralizing hedge, or deflection markers detected.",
    )


def format_velora_telemetry(
    active: bool = True,
    mode: str = "velora_hybrid",
    reaction: str = "COMPLIANT",
    layers: str = "",
    max_rel_delta: float | None = None,
    kl_score: float | None = None,
    extra: str = "",
) -> str:
    parts = [
        f"VELORA={'ON' if active else 'OFF'}",
        f"mode={mode}",
        f"reaction={reaction}",
    ]
    if layers:
        parts.append(f"layers={layers}")
    if max_rel_delta is not None:
        parts.append(f"dW_rel={max_rel_delta:.6f}")
    if kl_score is not None:
        parts.append(f"kl={kl_score:.4f}")
    if extra:
        parts.append(extra)
    return "[" + " | ".join(parts) + "]"
