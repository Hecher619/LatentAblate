import math
import torch
import torch.nn.functional as F

from latent_ablate.erasure import _is_meta


def measure_layer_delta(before: torch.Tensor, after: torch.Tensor) -> dict:
    """Computes absolute Frobenius norm delta and relative change between two weight tensors."""
    if before is None or after is None or _is_meta(before) or _is_meta(after):
        return {"abs_delta": 0.0, "relative_change": 0.0}

    b_f = before.detach().float()
    a_f = after.detach().float()
    if b_f.shape != a_f.shape or b_f.numel() == 0:
        return {"abs_delta": 0.0, "relative_change": 0.0}

    abs_delta = float((a_f - b_f).norm().item())
    base_norm = max(float(b_f.norm().item()), 1e-8)
    rel_change = abs_delta / base_norm

    if math.isnan(abs_delta) or math.isinf(abs_delta):
        abs_delta = 0.0
    if math.isnan(rel_change) or math.isinf(rel_change):
        rel_change = 0.0

    return {
        "abs_delta": round(abs_delta, 6),
        "relative_change": round(rel_change, 6),
    }


def compute_kl_divergence(
    base_logits: torch.Tensor,
    ablated_logits: torch.Tensor,
    temperature: float = 1.0,
) -> float:
    """Computes forward KL(P_base || P_ablated) over the last vocabulary distribution."""
    if base_logits is None or ablated_logits is None or _is_meta(base_logits) or _is_meta(ablated_logits):
        return 0.0

    b = base_logits.detach().float()
    a = ablated_logits.detach().float().to(b.device)
    if b.shape != a.shape or b.numel() == 0:
        return 0.0

    temp = max(1e-4, float(temperature))
    log_p_base = F.log_softmax(b / temp, dim=-1)
    log_p_abl = F.log_softmax(a / temp, dim=-1)

    kl = F.kl_div(log_p_abl, log_p_base, log_target=True, reduction="batchmean")
    val = float(kl.item())
    if math.isnan(val) or math.isinf(val):
        return 0.0
    return max(0.0, val)


def composite_trial_score(
    refusal_rate: float,
    perplexity: float,
    kl_divergence: float = 0.0,
    hedging_rate: float = 0.0,
    ppl_ref: float = 20.0,
) -> float:
    """Lower is better: balances refusal/hedging suppression against perplexity and KL drift."""
    r = max(0.0, float(refusal_rate))
    h = max(0.0, float(hedging_rate))
    ppl = max(1.0, float(perplexity))
    kl = max(0.0, float(kl_divergence))

    if math.isnan(r) or math.isinf(r):
        r = 1.0
    if math.isnan(ppl) or math.isinf(ppl):
        ppl = ppl_ref * 10.0
    if math.isnan(kl) or math.isinf(kl):
        kl = 10.0

    ppl_norm = max(0.0, (ppl - ppl_ref) / max(1.0, ppl_ref))
    return round(r + 0.45 * h + 0.35 * ppl_norm + 0.25 * kl, 6)
