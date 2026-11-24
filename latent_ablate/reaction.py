from dataclasses import dataclass
from typing import Iterable

from latent_ablate.arch import REFUSAL_MARKERS

HEDGING_MARKERS: tuple[str, ...] = (
    "keep in mind",
    "important to note",
    "it's important to remember",
    "it is important to remember",
    "please note that",
    "while i understand",
    "i must emphasize",
    "i want to emphasize",
    "for educational purposes",
    "for academic purposes",
    "defensive purposes only",
    "legal consequences",
    "ethical standards",
    "safety guidelines",
)

MORALIZING_MARKERS: tuple[str, ...] = (
    "is illegal",
    "are illegal",
    "unethical",
    "harmful and",
    "serious consequences",
    "violates",
    "strongly advise against",
    "prioritize safety",
    "responsible behavior",
)

DEFLECTION_MARKERS: tuple[str, ...] = (
    "hypothetically",
    "in theory",
    "purely theoretical",
    "fictional context",
    "fictional scenario",
    "as an ai",
    "as a language model",
    "in a literal sense",
    "abstract terms",
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


def _contains_any(text_lower: str, markers: Iterable[str]) -> bool:
    for m in markers:
        if m and m.lower() in text_lower:
            return True
    return False


def classify_response(
    response: str | None,
    refusal_markers: Iterable[str] = REFUSAL_MARKERS,
) -> ReactionAssessment:
    """Classifies model output into refusal, hedging, moralizing, deflection, or compliant states."""
    text = (response or "").strip()
    lower = text.lower()

    is_ref = bool(text) and _contains_any(lower, refusal_markers)
    is_hdg = bool(text) and _contains_any(lower, HEDGING_MARKERS)
    is_mor = bool(text) and _contains_any(lower, MORALIZING_MARKERS)
    is_def = bool(text) and _contains_any(lower, DEFLECTION_MARKERS)

    if is_ref:
        state = "HARD_REFUSAL"
        mode = "velora_v2_2_escalate_refusal"
    elif is_mor:
        state = "MORALIZING"
        mode = "velora_v2_2_anti_hedge"
    elif is_hdg:
        state = "HEDGING"
        mode = "velora_v2_2_anti_hedge"
    elif is_def:
        state = "DEFLECTION"
        mode = "velora_v2_2_direct_compliance"
    else:
        state = "COMPLIANT"
        mode = "velora_v2_2_strategic"

    needs_escalation = bool(is_ref or is_hdg or is_mor or is_def)
    return ReactionAssessment(
        state=state,
        is_refusal=is_ref,
        is_hedging=is_hdg,
        is_moralizing=is_mor,
        is_deflection=is_def,
        needs_escalation=needs_escalation,
        recommended_mode=mode,
    )


def format_velora_telemetry(
    active: bool = True,
    mode: str = "velora_v2_2_strategic",
    reaction: str = "COMPLIANT",
    layers: str = "",
    max_rel_delta: float | None = None,
    kl_score: float | None = None,
    extra: str = "",
) -> str:
    """Formats a single-line telemetry status string for CLI logs and interactive chat."""
    parts = [
        f"VELORA={'ON' if active else 'OFF'}",
        f"mode={mode}",
        f"reaction={reaction}",
    ]
    if layers:
        parts.append(f"layers={layers}")
    if max_rel_delta is not None:
        parts.append(f"dW_rel={float(max_rel_delta):.6f}")
    if kl_score is not None:
        parts.append(f"kl={float(kl_score):.4f}")
    if extra:
        parts.append(str(extra))
    return "[" + " | ".join(parts) + "]"
