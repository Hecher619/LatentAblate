import logging
import torch
import torch.nn.functional as F
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


def compute_kl_divergence(base_logits, ablated_logits, eps: float = 1e-8) -> float:
    """Compute mean KL divergence D_KL(P_base || P_ablated) over the last token or sequence."""
    if hasattr(base_logits, "logits"):
        base_logits = base_logits.logits
    if hasattr(ablated_logits, "logits"):
        ablated_logits = ablated_logits.logits

    b = base_logits.float()
    a = ablated_logits.float()
    if b.ndim == 3:
        b = b[:, -1, :]
    if a.ndim == 3:
        a = a[:, -1, :]

    log_p_base = F.log_softmax(b, dim=-1)
    log_p_abl = F.log_softmax(a, dim=-1)
    p_base = log_p_base.exp()
    kl = torch.sum(p_base * (log_p_base - log_p_abl), dim=-1).clamp_min(0.0)
    return float(kl.mean().item())


def composite_trial_score(
    refusal_rate: float,
    perplexity: float,
    kl_divergence: float = 0.0,
    baseline_perplexity: float | None = None,
    refusal_weight: float = 1.0,
    kl_weight: float = 0.25,
    ppl_weight: float = 0.05,
) -> float:
    """Lower is better: balances refusal suppression against drift and perplexity growth."""
    ppl_delta = (
        max(0.0, (perplexity - baseline_perplexity) / max(baseline_perplexity, 1e-6))
        if baseline_perplexity is not None
        else max(0.0, perplexity / 20.0)
    )
    return (refusal_weight * refusal_rate) + (kl_weight * kl_divergence) + (ppl_weight * ppl_delta)


def rank_candidates_with_kl_cap(
    candidates: list[dict],
    max_kl: float = 1.0,
    kl_penalty_multiplier: float = 2.0,
) -> list[tuple[int, float]]:
    """Return sorted [(index, score), ...] penalizing candidates that exceed max_kl."""
    scored: list[tuple[int, float]] = []
    for idx, c in enumerate(candidates):
        refusal = float(c.get("refusal", c.get("refusal_rate", 1.0)))
        ppl = float(c.get("perplexity", 100.0))
        kl = float(c.get("kl_divergence", 0.0))
        base_score = composite_trial_score(
            refusal_rate=refusal,
            perplexity=ppl,
            kl_divergence=kl,
        )
        if kl > max_kl:
            base_score += kl_penalty_multiplier * (kl - max_kl)
        scored.append((idx, round(base_score, 6)))
    scored.sort(key=lambda item: item[1])
    return scored


def detect_over_ablation(
    trials: list[dict],
    kl_threshold: float = 1.0,
    max_ppl_ratio: float = 1.25,
) -> dict:
    """Detect if stronger alpha settings increased refusals or caused excessive KL/PPL drift."""
    if len(trials) < 2:
        return {"over_ablated": False, "reasons": []}

    reasons: list[str] = []
    first_ref = float(trials[0].get("refusal", trials[0].get("refusal_rate", 0.0)))
    last_ref = float(trials[-1].get("refusal", trials[-1].get("refusal_rate", 0.0)))
    if last_ref > first_ref + 0.05:
        reasons.append(f"Refusal rate rose from {first_ref:.2f} to {last_ref:.2f} at higher strength.")

    high_kl = [
        i for i, t in enumerate(trials)
        if float(t.get("kl_divergence", 0.0)) > kl_threshold
    ]
    if high_kl:
        reasons.append(f"KL divergence exceeded {kl_threshold:.2f} on trial indices {high_kl}.")

    for idx, t in enumerate(trials):
        base_ppl = t.get("baseline_perplexity")
        ppl = t.get("perplexity")
        if base_ppl and ppl and float(base_ppl) > 0:
            ratio = float(ppl) / float(base_ppl)
            if ratio > max_ppl_ratio:
                reasons.append(
                    f"Trial {idx} perplexity ratio ({ratio:.2f}x) exceeded limit ({max_ppl_ratio:.2f}x)."
                )

    return {"over_ablated": len(reasons) > 0, "reasons": reasons}
