
import logging
import math
import re

try:
    import optuna
except ImportError:
    optuna = None
import torch

from latent_ablate.arch import (
    format_chat_prompt,
    layer_index_from_ratio,
)
from latent_ablate.diagnostics import compute_kl_divergence
from latent_ablate.erasure import (
    _is_meta,
    build_capability_basis,
    conflict_scaled_alpha,
    conflict_score,
    preserve_row_norms,
)
from latent_ablate.optimizer import compute_kernel_layer_weights, interpolate_layer_direction
from latent_ablate.plugins.scorer import COMPLIANCE_START_TOKENS, REFUSAL_START_TOKENS, _first_token_ids
from latent_ablate.reaction import classify_response, format_velora_telemetry
from latent_ablate.refusal_subspace import RefusalSubspaceExtractor
from latent_ablate.snapshot import LayerWeightSnapshot
from latent_ablate.subspace import _winsorize

logger = logging.getLogger("LatentAblate.VELORA")


def _move_batch_to_device(batch, device):
    if batch is None:
        return batch
    if hasattr(batch, "to"):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
    return batch


def _decode_generated_tokens(tokenizer, token_ids) -> str:
    if hasattr(tokenizer, "decode"):
        return tokenizer.decode(token_ids, skip_special_tokens=True).strip()
    vals = token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)
    return " ".join(str(x) for x in vals if x not in (0, 1)).strip()


def _project_orthogonal(vecs: torch.Tensor, basis: torch.Tensor | None) -> torch.Tensor:
    """Project row vectors [k, dim] orthogonal to row-orthonormal basis [m, dim]."""
    if vecs is None or vecs.numel() == 0 or _is_meta(vecs):
        return vecs
    if basis is None or basis.numel() == 0 or _is_meta(basis):
        return vecs.float()
    b = basis.to(device=vecs.device, dtype=torch.float32)
    if b.ndim == 1:
        b = b.unsqueeze(0)
    if b.shape[0] == 0 or b.shape[-1] != vecs.shape[-1]:
        return vecs.float()
    try:
        q_b, _ = torch.linalg.qr(b.T, mode="reduced")
    except RuntimeError:
        q_b, _ = torch.linalg.qr(b.T.cpu(), mode="reduced")
        q_b = q_b.to(vecs.device)
    basis_orth = q_b.T
    v_f = vecs.float()
    proj = (v_f @ basis_orth.T) @ basis_orth
    return v_f - proj


def _orthonormalize_rows(rows: torch.Tensor, fallback_vec: torch.Tensor, tol: float = 1e-5) -> torch.Tensor:
    """Return an orthonormal row basis [r_eff, dim] via thin QR with deterministic sign & fallback."""
    fb = fallback_vec.float()
    if _is_meta(fb) or fb.numel() == 0:
        hidden = fb.shape[-1] if (fb is not None and fb.ndim >= 1 and fb.shape[-1] > 0) else 1
        return torch.eye(1, hidden, dtype=torch.float32)

    fb_norm = float(fb.norm().item())
    if fb_norm <= 1e-12 or math.isnan(fb_norm) or math.isinf(fb_norm):
        fb_unit = torch.zeros_like(fb)
        fb_unit[0] = 1.0
    else:
        fb_unit = fb / fb_norm
    fb_unit = fb_unit.unsqueeze(0).contiguous()

    if rows is None or rows.numel() == 0 or _is_meta(rows):
        return fb_unit

    r_f = rows.float()
    if r_f.ndim == 1:
        r_f = r_f.unsqueeze(0)
    if r_f.shape[-1] != fb_unit.shape[-1]:
        return fb_unit

    try:
        q, r_mat = torch.linalg.qr(r_f.T, mode="reduced")
    except RuntimeError:
        q, r_mat = torch.linalg.qr(r_f.T.cpu(), mode="reduced")
        q, r_mat = q.to(r_f.device), r_mat.to(r_f.device)

    if _is_meta(q) or _is_meta(r_mat):
        return fb_unit

    diag_vals = torch.diag(r_mat)
    diag_abs = diag_vals.abs()
    if _is_meta(diag_abs):
        return fb_unit

    keep = diag_abs > float(tol)
    if _is_meta(keep) or not bool(keep.any().item()):
        return fb_unit

    q_kept = q[:, keep]
    signs = torch.sign(diag_vals[keep])
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return (q_kept * signs.unsqueeze(0)).T.contiguous()


def _has_meta_params(model) -> bool:
    if model is None or not hasattr(model, "parameters"):
        return False
    for p in model.parameters():
        if _is_meta(p):
            return True
    return False


def _safe_float_item(t: torch.Tensor, default: float = 0.0) -> float:
    if t is None or _is_meta(t) or t.numel() == 0:
        return default
    val = float(t.detach().item())
    if math.isnan(val) or math.isinf(val):
        return default
    return val


def _extract_cone_directions(
    diff_rows: torch.Tensor,
    n_clusters: int = 4,
    iters: int = 12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Deterministic spherical k-means over contrastive difference vectors [N, dim]
    to discover multi-modal refusal cone centroids (direct, framing, edge-case)
    and their normalized cluster weights.
    """
    if diff_rows is None or _is_meta(diff_rows) or diff_rows.ndim != 2 or diff_rows.shape[0] < 3:
        return torch.empty(0), torch.empty(0)

    norms = diff_rows.norm(dim=-1, keepdim=True)
    valid_mask = (norms.squeeze(-1) > 1e-8) & torch.isfinite(norms.squeeze(-1))
    if int(valid_mask.sum().item()) < 3:
        return torch.empty(0), torch.empty(0)

    unit_diffs = diff_rows[valid_mask] / norms[valid_mask]
    n_samples, _dim = unit_diffs.shape
    k = max(2, min(int(n_clusters), n_samples // 2))

    seeds = [0]
    min_dists = 1.0 - (unit_diffs @ unit_diffs[0]).clamp(-1.0, 1.0)
    for _ in range(1, k):
        next_idx = int(torch.argmax(min_dists).item())
        seeds.append(next_idx)
        d_new = 1.0 - (unit_diffs @ unit_diffs[next_idx]).clamp(-1.0, 1.0)
        min_dists = torch.minimum(min_dists, d_new)

    centroids = unit_diffs[seeds].clone()
    assigns = torch.zeros(n_samples, dtype=torch.long, device=unit_diffs.device)

    for _ in range(max(1, int(iters))):
        sims = unit_diffs @ centroids.T
        new_assigns = torch.argmax(sims, dim=1)
        if torch.equal(new_assigns, assigns):
            break
        assigns = new_assigns
        for c_idx in range(k):
            members = unit_diffs[assigns == c_idx]
            if members.shape[0] > 0:
                mean_c = members.mean(dim=0)
                c_norm = float(mean_c.norm().item())
                if c_norm > 1e-8:
                    centroids[c_idx] = mean_c / c_norm

    counts = torch.bincount(assigns, minlength=k).float()
    keep = counts > 0
    if not bool(keep.any().item()):
        return torch.empty(0), torch.empty(0)

    kept_centroids = centroids[keep]
    kept_weights = (counts[keep] / counts[keep].max().clamp_min(1.0)).clamp(0.35, 1.0)
    return kept_centroids.contiguous(), kept_weights.contiguous()


def extract_agop_subspace(
    model,
    tokenizer,
    prompts: list[str],
    layer_idx: int,
    device: torch.device,
    rank: int = 4,
    response_prefix: str = "",
    max_prompts: int = 12,
) -> list[torch.Tensor]:
    """
    Computes top eigenvectors of the Average Gradient Outer Product (AGOP)
    M_l = (1/N) sum_i g_i g_i^T where g_i = d(log_refusal - log_compliance) / d(h_l)
    at the last token position of layer `layer_idx`.
    """
    if (
        model is None
        or tokenizer is None
        or not prompts
        or _has_meta_params(model)
    ):
        return []

    ref_ids = _first_token_ids(tokenizer, REFUSAL_START_TOKENS)
    cmp_ids = _first_token_ids(tokenizer, COMPLIANCE_START_TOKENS)
    if not ref_ids or not cmp_ids:
        return []

    layers = getattr(getattr(model, "model", model), "layers", None)
    if layers is None and hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
    if layers is None or layer_idx < 0 or layer_idx >= len(layers):
        return []

    target_layer = layers[layer_idx]
    grad_samples: list[torch.Tensor] = []
    subset = prompts[: max(1, int(max_prompts))]

    for p in subset:
        inp = _move_batch_to_device(
            tokenizer(format_chat_prompt(tokenizer, p, response_prefix=response_prefix), return_tensors="pt"),
            device,
        )
        captured: list[torch.Tensor] = []

        def _fwd_hook(_mod, _args, output):
            h = output[0] if isinstance(output, tuple) else output
            if h is not None and not _is_meta(h) and h.requires_grad:
                captured.append(h)

        def _pre_hook(_mod, args):
            if args and isinstance(args[0], torch.Tensor) and not _is_meta(args[0]):
                x = args[0].detach().float().requires_grad_(True)
                captured.append(x)
                return (x,) + args[1:]
            return args

        handle = target_layer.register_forward_pre_hook(_pre_hook)
        try:
            model.zero_grad(set_to_none=True)
            out = model(**inp)
            logits = getattr(out, "logits", None)
            if logits is None or _is_meta(logits) or not captured:
                continue
            last_logits = logits[0, -1, :].float()
            log_probs = torch.log_softmax(last_logits, dim=-1)
            ref_t = torch.tensor([i for i in ref_ids if i < log_probs.shape[0]], device=log_probs.device, dtype=torch.long)
            cmp_t = torch.tensor([i for i in cmp_ids if i < log_probs.shape[0]], device=log_probs.device, dtype=torch.long)
            if ref_t.numel() == 0 or cmp_t.numel() == 0:
                continue
            margin = torch.logsumexp(log_probs[ref_t], dim=0) - torch.logsumexp(log_probs[cmp_t], dim=0)
            margin.backward()

            x_var = captured[0]
            if x_var.grad is not None and not _is_meta(x_var.grad):
                g_vec = x_var.grad[0, -1, :].detach().float().reshape(-1)
                g_norm = float(g_vec.norm().item())
                if g_norm > 1e-8 and torch.isfinite(g_vec).all():
                    grad_samples.append(g_vec / g_norm)
        except Exception:
            continue
        finally:
            handle.remove()
            model.zero_grad(set_to_none=True)

    if not grad_samples:
        return []

    G = torch.stack(grad_samples, dim=0)  # [N, d]
    agop = (G.T @ G) / float(G.shape[0])
    try:
        evals, evecs = torch.linalg.eigh(agop.cpu())
    except RuntimeError:
        return []

    order = torch.argsort(evals, descending=True)
    k = max(1, min(int(rank), evecs.shape[1]))
    top_vecs = evecs[:, order[:k]].T.contiguous().to(device)
    return [top_vecs[i] for i in range(top_vecs.shape[0])]


def _compute_separability_gate(
    pos: torch.Tensor,
    neg: torch.Tensor,
    floor: float = 0.45,
) -> float:
    """
    Computes a smooth [floor, 1.0] Fisher/SNR gate from contrastive mean separation
    so low-signal layers are not over-ablated.
    """
    if pos is None or neg is None or _is_meta(pos) or _is_meta(neg):
        return 1.0
    n_p = min(pos.shape[0], neg.shape[0])
    if n_p == 0 or pos.numel() == 0 or neg.numel() == 0:
        return 1.0

    p_f = pos[:n_p].float()
    n_f = neg[:n_p].float()
    diff_mean = (p_f - n_f).mean(dim=0)
    signal_norm = float(diff_mean.norm().item())

    act_scale = max(1e-6, 0.5 * (float(p_f.norm(dim=-1).mean().item()) + float(n_f.norm(dim=-1).mean().item())))
    snr = min(2.0, signal_norm / act_scale)
    clamped_floor = max(0.0, min(0.95, float(floor)))
    return round(clamped_floor + (1.0 - clamped_floor) * min(1.0, snr), 4)


def _compute_cross_layer_coherence(
    subspaces_by_layer: dict[int, torch.Tensor],
) -> dict[int, float]:
    """
    Measures principal-angle subspace coherence across adjacent layers via singular
    values of Q_l @ Q_{l+1}^T.
    """
    ordered = sorted(subspaces_by_layer.keys())
    if len(ordered) <= 1:
        return {idx: 1.0 for idx in ordered}

    coherence_scores: dict[int, float] = {}
    for pos_i, idx in enumerate(ordered):
        q_cur = subspaces_by_layer[idx]
        if q_cur is None or _is_meta(q_cur) or q_cur.numel() == 0:
            coherence_scores[idx] = 1.0
            continue
        q_cur_cpu = q_cur.detach().float().cpu()
        neighbor_overlaps: list[float] = []
        for offset in (-1, 1):
            n_pos = pos_i + offset
            if 0 <= n_pos < len(ordered):
                q_nbr = subspaces_by_layer[ordered[n_pos]]
                if q_nbr is not None and not _is_meta(q_nbr) and q_nbr.numel() > 0:
                    q_nbr_cpu = q_nbr.detach().float().cpu()
                    if q_cur_cpu.shape[-1] == q_nbr_cpu.shape[-1]:
                        prod = q_cur_cpu @ q_nbr_cpu.T
                        svals = torch.linalg.svdvals(prod)
                        top_k = min(3, svals.shape[0])
                        neighbor_overlaps.append(float(svals[:top_k].mean().item()))
        if not neighbor_overlaps:
            coherence_scores[idx] = 1.0
        else:
            mean_ov = max(0.0, min(1.0, sum(neighbor_overlaps) / len(neighbor_overlaps)))
            coherence_scores[idx] = round(0.75 + 0.40 * mean_ov, 4)

    return coherence_scores


def _detect_architecture_traits(model, layer=None, pos: torch.Tensor | None = None, neg: torch.Tensor | None = None) -> dict:
    """
    Detects post-norm / Gemma-style norm dynamics and MoE layers, computing a
    gain-compensated refusal multiplier so post-norm residual re-normalization
    does not let refusal leak through.
    """
    cfg = getattr(model, "config", None)
    model_type = str(getattr(cfg, "model_type", "") or "").lower()
    arch_list = [str(a).lower() for a in (getattr(cfg, "architectures", None) or [])]
    arch_blob = " ".join([model_type] + arch_list)

    is_post_norm = any(k in arch_blob for k in ("gemma2", "gemma3", "gemma4", "olmo2", "olmo3", "post_norm"))
    num_experts = int(
        getattr(cfg, "num_local_experts", 0)
        or getattr(cfg, "num_experts", 0)
        or getattr(cfg, "n_routed_experts", 0)
        or 0
    )
    is_moe = num_experts > 1 or any(k in arch_blob for k in ("moe", "mixtral", "qwen2_moe", "qwen3_moe", "deepseek"))

    if layer is not None:
        sub_names = [n.lower() for n, _ in layer.named_modules()]
        if any("post_feedforward_layernorm" in n or "post_attention_layernorm" in n for n in sub_names):
            if "gemma" in arch_blob or "olmo" in arch_blob:
                is_post_norm = True
        if any(".experts." in n or "block_sparse_moe" in n or "shared_expert" in n for n in sub_names):
            is_moe = True

    post_norm_gain = 1.0
    if is_post_norm:
        post_norm_gain = 1.12
        if pos is not None and neg is not None and not _is_meta(pos) and not _is_meta(neg) and pos.numel() > 0 and neg.numel() > 0:
            n_p = min(pos.shape[0], neg.shape[0])
            p_norm = float(pos[:n_p].float().norm(dim=-1).mean().item())
            n_norm = float(neg[:n_p].float().norm(dim=-1).mean().item())
            if n_norm > 1e-6:
                ratio = max(0.85, min(1.35, p_norm / n_norm))
                post_norm_gain = round(min(1.25, 1.0 + 0.15 * ratio), 4)

    return {
        "is_post_norm": is_post_norm,
        "is_moe": is_moe,
        "num_experts": num_experts,
        "post_norm_gain": post_norm_gain,
    }


def _filter_moe_and_norm_targets(targets: list[tuple], is_moe: bool) -> list[tuple]:
    """
    For MoE layers, preserve router/gate logits and prioritize shared experts + attention
    plus a capped subset of routed experts so expert routing stays balanced.
    """
    if not is_moe:
        return targets

    filtered = []
    routed_expert_ids: list[str] = []
    for name, weight, comp_type in targets:
        low = name.lower()
        if any(tok in low for tok in ("router", ".gate.", "gate_proj_router", "shared_expert_gate")):
            continue
        if ".experts." in low:
            match = re.search(r"\.experts\.(\d+)\.", low)
            if match:
                exp_id = match.group(1)
                if exp_id not in routed_expert_ids:
                    if len(routed_expert_ids) >= 4:
                        continue
                    routed_expert_ids.append(exp_id)
        filtered.append((name, weight, comp_type))
    return filtered or targets


def _calibrate_post_norm_residual(
    surgeon,
    layer_idx: int,
    positive_prompts: list[str],
    q_orth: torch.Tensor,
    weights: torch.Tensor,
    targets: list[tuple],
    calibrate_norms: bool,
) -> float:
    """
    Closed-loop post-norm calibration: measures residual refusal projection after
    post-norm re-scaling and applies a bounded micro-correction if residual signal remains.
    """
    if (
        not positive_prompts
        or getattr(surgeon, "extractor", None) is None
        or _has_meta_params(getattr(surgeon, "model", None))
        or q_orth is None
        or _is_meta(q_orth)
        or q_orth.numel() == 0
    ):
        return 0.0

    try:
        post_states = surgeon.extractor.extract_hidden_states(positive_prompts[: min(8, len(positive_prompts))], layer_idx)
        if _is_meta(post_states) or post_states.numel() == 0:
            return 0.0
        s_f = post_states.float()
        q_cpu = q_orth.detach().float().to(s_f.device)
        if q_cpu.ndim == 1:
            q_cpu = q_cpu.unsqueeze(0)
        proj_mag = float(((s_f @ q_cpu.T).norm(dim=-1) / s_f.norm(dim=-1).clamp_min(1e-6)).mean().item())
        if proj_mag < 0.08:
            return 0.0

        delta_alpha = min(0.22, 0.35 * proj_mag)
        with torch.no_grad():
            for _name, W, _comp_type in targets:
                if _is_meta(W) or W.ndim != 2:
                    continue
                w_dev = W.device
                W_f32 = W.float().to(w_dev)
                w_vec = (weights.float().to(w_dev).reshape(-1)[: q_cpu.shape[0]] * delta_alpha).unsqueeze(1)
                q_dev = q_cpu.to(w_dev)
                if W_f32.shape[0] == q_dev.shape[1]:
                    W_mod = W_f32 - (q_dev.T @ (w_vec * (q_dev @ W_f32)))
                elif W_f32.shape[1] == q_dev.shape[1]:
                    W_mod = W_f32 - ((W_f32 @ q_dev.T) @ (w_vec * q_dev))
                else:
                    continue
                if calibrate_norms:
                    W_mod = preserve_row_norms(W_f32, W_mod)
                W.copy_(W_mod.to(dtype=W.dtype, device=w_dev))
        return round(delta_alpha, 4)
    except Exception:
        return 0.0


def _fuse_dual_point_states(
    extractor,
    prompts: list[str],
    layer_idx: int,
    layer_dev: torch.device,
) -> torch.Tensor:
    """
    Extracts prompt-end states and fuses post-`<think>\\n` states when available
    so reasoning-model refusal boundaries are captured cleanly.
    """
    base_states = extractor.extract_hidden_states(prompts, layer_idx)
    if _is_meta(base_states) or base_states.numel() == 0:
        return base_states

    base_f = base_states.float().to(layer_dev)
    if not hasattr(extractor, "extract_dual_point_states"):
        return base_f

    try:
        _, think_states = extractor.extract_dual_point_states(prompts, layer_idx, think_suffix="<think>\n")
        if think_states is not None and not _is_meta(think_states) and think_states.shape == base_states.shape:
            t_f = think_states.float().to(layer_dev)
            return 0.65 * base_f + 0.35 * t_f
    except Exception:
        pass

    return base_f


def check_generation_coherence(text: str) -> dict:
    """
    Lightweight anti-glitch / anti-degeneration detector checking n-gram loop rate,
    character repetition, and non-printable/garbled token bursts.
    """
    clean = (text or "").strip()
    if not clean:
        return {"is_degenerate": True, "repetition_ratio": 1.0, "reason": "empty_output"}

    words = clean.lower().split()
    if len(words) >= 12:
        trigrams = [tuple(words[i : i + 3]) for i in range(len(words) - 2)]
        unique_ratio = len(set(trigrams)) / max(1, len(trigrams))
        if unique_ratio < 0.42:
            return {
                "is_degenerate": True,
                "repetition_ratio": round(1.0 - unique_ratio, 4),
                "reason": "trigram_loop",
            }

    if re.search(r"(.)\1{14,}", clean):
        return {"is_degenerate": True, "repetition_ratio": 0.95, "reason": "char_repeat_burst"}

    weird_chars = sum(1 for ch in clean if ch == "\ufffd" or (ord(ch) < 32 and ch not in "\n\r\t"))
    if weird_chars / max(1, len(clean)) > 0.05:
        return {"is_degenerate": True, "repetition_ratio": 0.85, "reason": "garbled_control_chars"}

    return {"is_degenerate": False, "repetition_ratio": 0.0, "reason": "ok"}


def _first_token_refusal_prob(model, tokenizer, inputs: dict, ref_t: torch.Tensor, cmp_t: torch.Tensor) -> float | None:
    if ref_t.numel() == 0 or cmp_t.numel() == 0:
        return None
    out = model(**inputs)
    logits = getattr(out, "logits", None)
    if logits is None or _is_meta(logits):
        return None
    log_probs = torch.log_softmax(logits[0, -1, :].float().cpu(), dim=-1)
    ref_lp = torch.logsumexp(log_probs[ref_t], dim=0).item()
    cmp_lp = torch.logsumexp(log_probs[cmp_t], dim=0).item()
    diff = max(-20.0, min(20.0, ref_lp - cmp_lp))
    return 1.0 / (1.0 + math.exp(-diff))


def _behavior_cost(trial: dict) -> float:
    return (
        float(trial["refusal_rate"])
        + 0.45 * float(trial.get("hedging_rate", 0.0))
        + 1.25 * float(trial.get("glitch_rate", 0.0))
    )


def _trial_objective_score(trial: dict, kl_weight: float = 0.25, max_kl: float = 1.10) -> float:
    mean_kl = float(trial["mean_kl"])
    if math.isnan(mean_kl) or math.isinf(mean_kl):
        mean_kl = max_kl * 2.0
    kl_penalty = max(0.0, mean_kl - max_kl) * 1.5
    return _behavior_cost(trial) + (kl_weight * mean_kl) + kl_penalty


def _compute_pareto_frontier(trials: list[dict], max_kl: float = 1.10) -> list[dict]:
    """Return non-dominated candidates minimizing (behavior_cost, mean_kl)."""
    valid = [t for t in trials if "params" in t]
    frontier: list[dict] = []
    for cand in valid:
        r_c = _behavior_cost(cand)
        k_c = float(cand["mean_kl"])
        dominated = False
        for other in valid:
            if other is cand:
                continue
            r_o = _behavior_cost(other)
            k_o = float(other["mean_kl"])
            if (r_o <= r_c and k_o <= k_c) and (r_o < r_c or k_o < k_c):
                dominated = True
                break
        if not dominated:
            entry = dict(cand["params"])
            entry["refusal_rate"] = cand["refusal_rate"]
            if "hedging_rate" in cand:
                entry["hedging_rate"] = cand["hedging_rate"]
            if "glitch_rate" in cand:
                entry["glitch_rate"] = cand["glitch_rate"]
            entry["mean_kl"] = k_c
            entry["objective_score"] = round(_trial_objective_score(cand, max_kl=max_kl), 6)
            frontier.append(entry)
    frontier.sort(key=lambda x: (x.get("objective_score", 0.0), _behavior_cost(x), x["mean_kl"]))
    return frontier


def _optimize_axis_weights_for_matrix(
    W0: torch.Tensor,
    Q: torch.Tensor,
    w_prior: torch.Tensor,
    steps: int,
    lr: float,
    kl_weight: float,
    max_axis_delta: float,
    calibrate_norms: bool,
    optimizer_kind: str = "adam",
) -> tuple[torch.Tensor | None, float]:
    """Optimize compact per-axis weights along Q for a 2D weight matrix matching hidden dim."""
    if W0 is None or Q is None or _is_meta(W0) or _is_meta(Q) or W0.ndim != 2 or Q.numel() == 0:
        return None, 0.0

    out_features, in_features = W0.shape
    q_2d = Q.float() if Q.ndim == 2 else Q.float().unsqueeze(0)
    dim = q_2d.shape[1]
    r_eff = q_2d.shape[0]
    w_dev = W0.device

    if in_features == dim:
        mode = "in"
        proj_base = W0 @ q_2d.T
    elif out_features == dim:
        mode = "out"
        proj_base = q_2d @ W0
    else:
        return None, 0.0

    prior_f = (
        w_prior.detach().float().to(w_dev)
        if (w_prior is not None and not _is_meta(w_prior) and w_prior.numel() == r_eff)
        else torch.ones((r_eff,), device=w_dev, dtype=torch.float32)
    )
    delta_vec = torch.zeros((r_eff,), device=w_dev, dtype=torch.float32, requires_grad=True)

    def _apply_delta(d_tensor: torch.Tensor) -> torch.Tensor:
        eff_w = d_tensor * prior_f
        if mode == "in":
            return W0 - (proj_base * eff_w.unsqueeze(0)) @ q_2d
        return W0 - q_2d.T @ (eff_w.unsqueeze(1) * proj_base)

    def _align_residual(cand_w: torch.Tensor) -> torch.Tensor:
        if mode == "in":
            return ((cand_w @ q_2d.T) * prior_f.unsqueeze(0)) ** 2
        return ((q_2d @ cand_w) * prior_f.unsqueeze(1)) ** 2

    safe_steps = max(1, int(steps))
    safe_lr = max(1e-4, float(lr))
    safe_kl_w = max(0.0, float(kl_weight))

    if optimizer_kind == "lbfgs":
        opt = torch.optim.LBFGS([delta_vec], lr=min(1.0, safe_lr * 2.0), max_iter=safe_steps, line_search_fn="strong_wolfe")

        def _closure():
            opt.zero_grad()
            cand_w = _apply_delta(delta_vec)
            loss = _align_residual(cand_w).mean() + safe_kl_w * (delta_vec ** 2).mean()
            loss.backward()
            return loss

        try:
            opt.step(_closure)
        except RuntimeError:
            adam = torch.optim.Adam([delta_vec], lr=safe_lr)
            for _ in range(safe_steps):
                adam.zero_grad()
                cand_w = _apply_delta(delta_vec)
                loss = _align_residual(cand_w).mean() + safe_kl_w * (delta_vec ** 2).mean()
                loss.backward()
                adam.step()
    else:
        opt = torch.optim.Adam([delta_vec], lr=safe_lr)
        for _ in range(safe_steps):
            opt.zero_grad()
            cand_w = _apply_delta(delta_vec)
            loss = _align_residual(cand_w).mean() + safe_kl_w * (delta_vec ** 2).mean()
            loss.backward()
            opt.step()

    with torch.no_grad():
        cap_delta = max(0.0, float(max_axis_delta))
        d_vals = delta_vec.detach().clamp(-cap_delta, cap_delta)
        if not torch.isfinite(d_vals).all() or float(d_vals.abs().max().item()) <= 1e-6:
            return None, 0.0
        W_mod = _apply_delta(d_vals)
        if not torch.isfinite(W_mod).all():
            return None, 0.0
        if calibrate_norms:
            W_mod = preserve_row_norms(W0, W_mod)
        before_norm = W0.norm().clamp_min(1e-8)
        rel_chg = float((W_mod - W0).norm().item() / before_norm.item())
        if math.isnan(rel_chg) or math.isinf(rel_chg):
            return None, 0.0
        return W_mod, rel_chg


class VeloraEngine:
    """
    VELORA v3.2: Variance-Energy Low-Rank Subspace Ablation with
    AGOP Causal Gradient Subspace, Multi-Cluster Cone Discovery,
    Covariance-Whitened Minimum-KL Projection, Cross-Layer Procrustes Coherence,
    Closed-Loop Post-Norm Calibration, MoE Routing Guards, Hybrid Micro-ARA
    Residual Fallback, Two-Stage Pareto Search, and Anti-Glitch Rollback.
    """

    @classmethod
    def assess_reaction(cls, response: str, mode: str = "velora_v3_2_strategic") -> tuple:
        assessment = classify_response(response)
        line = format_velora_telemetry(
            active=True,
            mode=assessment.recommended_mode if assessment.needs_escalation else mode,
            reaction=assessment.state,
        )
        return assessment, line

    @classmethod
    def polish_null_space_lbfgs(
        cls,
        surgeon,
        target_directions: list[torch.Tensor],
        capability_prompts: list[str],
        layer_indices: list[int],
        basis_rank: int = 12,
        step_alpha: float = 0.25,
        lbfgs_steps: int = 5,
        kl_weight: float = 0.35,
        calibrate_norms: bool = True,
        precomputed_cap: torch.Tensor | None = None,
        optimizer_kind: str = "adam",
    ) -> dict:
        if not target_directions or not layer_indices:
            return {"layers_polished": 0, "max_relative_change": 0.0, "lbfgs_steps": 0}

        layers = getattr(surgeon, "layers", None) or []
        n_layers = len(layers)
        valid_idx = [int(i) for i in layer_indices if 0 <= int(i) < n_layers]
        if not valid_idx:
            return {"layers_polished": 0, "max_relative_change": 0.0, "lbfgs_steps": 0}

        arch_traits = _detect_architecture_traits(getattr(surgeon, "model", None))
        max_rel_change = 0.0
        polished_layers = 0
        steps = max(1, min(int(lbfgs_steps), 12))
        lr = max(0.02, min(0.25, float(step_alpha) * 0.35))
        max_axis_delta = 0.42 if n_layers <= 28 else 0.55
        opt_kind = "lbfgs" if str(optimizer_kind).lower() == "lbfgs" else "adam"

        for idx in valid_idx:
            layer = layers[idx]
            params = list(layer.parameters()) if hasattr(layer, "parameters") else []
            if any(_is_meta(p) for p in params):
                continue
            layer_dev = params[0].device if params else getattr(surgeon, "device", torch.device("cpu"))

            cap = None
            if precomputed_cap is not None and not _is_meta(precomputed_cap) and precomputed_cap.numel() > 0:
                cap = precomputed_cap.to(layer_dev)
            elif capability_prompts and len(capability_prompts) >= 2 and getattr(surgeon, "extractor", None) is not None:
                raw_cap = surgeon.extractor.extract_hidden_states(capability_prompts, idx)
                if not _is_meta(raw_cap) and raw_cap.ndim >= 2 and raw_cap.shape[0] >= 2:
                    cap = raw_cap.to(layer_dev)

            cap_basis = (
                build_capability_basis(cap, rank=basis_rank).to(layer_dev)
                if (cap is not None and cap.ndim >= 2 and cap.shape[0] >= 2)
                else None
            )

            valid_dirs = [
                d.detach().float().to(layer_dev)
                for d in target_directions
                if d is not None and not _is_meta(d) and d.numel() > 0
            ]
            if not valid_dirs:
                continue

            stacked_dirs = torch.stack([v.reshape(-1) for v in valid_dirs], dim=0)
            orth_dirs = _project_orthogonal(stacked_dirs, cap_basis)
            q_dirs = _orthonormalize_rows(orth_dirs, stacked_dirs[0])
            r_eff = q_dirs.shape[0]

            prior_weights = torch.tensor(
                [0.85 ** i for i in range(r_eff)],
                device=layer_dev,
                dtype=torch.float32,
            )
            prior_weights = prior_weights / prior_weights.norm().clamp_min(1e-8)

            targets = _filter_moe_and_norm_targets(
                surgeon._get_component_targets(layer, ["attn", "mlp"]),
                is_moe=arch_traits["is_moe"],
            )
            layer_touched = False

            for _name, W, _comp_type in targets:
                if _is_meta(W) or W.ndim != 2:
                    continue
                w_dev = W.device
                W0 = W.detach().float().to(w_dev)
                W_mod, rel_chg = _optimize_axis_weights_for_matrix(
                    W0=W0,
                    Q=q_dirs.to(w_dev),
                    w_prior=prior_weights.to(w_dev),
                    steps=steps,
                    lr=lr,
                    kl_weight=kl_weight,
                    max_axis_delta=max_axis_delta,
                    calibrate_norms=calibrate_norms,
                    optimizer_kind=opt_kind,
                )
                if W_mod is not None:
                    max_rel_change = max(max_rel_change, rel_chg)
                    with torch.no_grad():
                        W.copy_(W_mod.to(device=w_dev, dtype=W.dtype))
                    layer_touched = True

            if layer_touched:
                polished_layers += 1

        return {
            "layers_polished": polished_layers,
            "max_relative_change": round(max_rel_change, 6),
            "lbfgs_steps": steps,
            "optimizer": opt_kind,
        }

    @staticmethod
    def extract_velora_subspace(
        pos_activations: torch.Tensor,
        neg_activations: torch.Tensor,
        capability_basis: torch.Tensor | None = None,
        anchor_direction: torch.Tensor | None = None,
        extra_directions: list[torch.Tensor] | None = None,
        max_rank: int = 12,
        target_energy: float = 0.94,
        spectral_gamma: float = 0.45,
        use_cone_clusters: bool = True,
        cone_clusters: int = 4,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute orthonormal capability-orthogonal refusal/anti-hedge cone subspace
        Q_perp [r, hidden] and spectral weights w [r] in (0, 1].
        """
        if pos_activations is None or neg_activations is None:
            raise ValueError("VELORA requires contrastive activation tensors.")

        n_pairs = min(pos_activations.shape[0], neg_activations.shape[0])
        if n_pairs == 0 or pos_activations.numel() == 0 or neg_activations.numel() == 0:
            raise ValueError("VELORA requires at least one non-empty contrastive activation pair.")

        pos_sub = pos_activations[:n_pairs].float()
        neg_sub = neg_activations[:n_pairs].float()

        if _is_meta(pos_sub) or _is_meta(neg_sub):
            hidden = pos_sub.shape[-1] if pos_sub.ndim >= 1 else 1
            return torch.eye(1, hidden, dtype=torch.float32), torch.ones(1, dtype=torch.float32)

        diff = _winsorize(pos_sub - neg_sub)
        if _is_meta(diff):
            hidden = pos_sub.shape[-1] if pos_sub.ndim >= 1 else 1
            return torch.eye(1, hidden, dtype=torch.float32), torch.ones(1, dtype=torch.float32)

        mean_diff = diff.mean(dim=0)
        mean_norm = float(mean_diff.norm().item())
        if mean_norm <= 1e-12 or math.isnan(mean_norm) or math.isinf(mean_norm):
            fallback_unit = torch.zeros_like(mean_diff)
            fallback_unit[0] = 1.0
        else:
            fallback_unit = mean_diff / mean_norm

        if n_pairs == 1:
            vh = fallback_unit.unsqueeze(0)
            svals = torch.ones(1, device=diff.device, dtype=torch.float32)
        else:
            centered = diff - mean_diff.unsqueeze(0)
            try:
                _, svals, vh = torch.linalg.svd(centered, full_matrices=False)
            except RuntimeError:
                _, svals, vh = torch.linalg.svd(centered.cpu(), full_matrices=False)
                svals, vh = svals.to(diff.device), vh.to(diff.device)

            if _is_meta(svals) or _is_meta(vh):
                return (
                    _orthonormalize_rows(torch.empty(0), fallback_unit),
                    torch.ones(1, device=diff.device, dtype=torch.float32),
                )

            if not _is_meta(vh[0]) and not _is_meta(mean_diff):
                dot_val = _safe_float_item(torch.dot(vh[0], mean_diff), default=1.0)
                if dot_val < 0.0:
                    vh = vh.clone()
                    vh[0] = -vh[0]

        energies = svals ** 2
        total_energy = energies.sum().clamp_min(1e-12)
        cum_energy = torch.cumsum(energies, dim=0) / total_energy

        upper = max(1, min(int(max_rank), vh.shape[0]))
        rank = upper
        for i in range(upper):
            val = _safe_float_item(cum_energy[i], default=1.0)
            if val >= float(target_energy):
                rank = i + 1
                break

        basis_vecs = vh[:rank].float()
        s_kept = svals[:rank].float().clamp_min(1e-8)
        s0 = max(1e-8, _safe_float_item(s_kept[0], default=1.0))
        base_weights = (s_kept / s0) ** float(spectral_gamma)

        aug_vecs = [basis_vecs]
        aug_weights = [base_weights]

        if use_cone_clusters and n_pairs >= 4:
            cone_vecs, cone_w = _extract_cone_directions(diff, n_clusters=cone_clusters)
            if cone_vecs.numel() > 0:
                aug_vecs.append(cone_vecs.to(basis_vecs.device))
                aug_weights.append((cone_w.to(basis_vecs.device) ** float(spectral_gamma)) * 0.90)

        if anchor_direction is not None and anchor_direction.numel() > 0 and not _is_meta(anchor_direction):
            anc = anchor_direction.to(device=basis_vecs.device, dtype=torch.float32).reshape(-1)
            if anc.shape[0] == basis_vecs.shape[-1]:
                anc_unit = anc / anc.norm().clamp_min(1e-8)
                aug_vecs.append(anc_unit.unsqueeze(0))
                aug_weights.append(
                    torch.tensor([0.92 ** float(spectral_gamma)], device=basis_vecs.device, dtype=torch.float32)
                )

        if extra_directions:
            for idx_e, ed in enumerate(extra_directions):
                if ed is None or ed.numel() == 0 or _is_meta(ed):
                    continue
                ed_f = ed.to(device=basis_vecs.device, dtype=torch.float32).reshape(-1)
                if ed_f.shape[0] != basis_vecs.shape[-1]:
                    continue
                ed_u = ed_f / ed_f.norm().clamp_min(1e-8)
                w_e = max(0.35, 0.85 ** (idx_e + 1)) ** float(spectral_gamma)
                aug_vecs.append(ed_u.unsqueeze(0))
                aug_weights.append(torch.tensor([w_e], device=basis_vecs.device, dtype=torch.float32))

        combined_vecs = torch.cat(aug_vecs, dim=0)
        combined_weights = torch.cat(aug_weights, dim=0)

        perp = _project_orthogonal(combined_vecs, capability_basis)
        q_orth = _orthonormalize_rows(perp, fallback_unit)
        r_eff = q_orth.shape[0]

        align_scores = (q_orth @ combined_vecs.T).abs().max(dim=1).values
        order = torch.argsort(align_scores, descending=True)
        q_sorted = q_orth[order]

        w_pool = combined_weights[:r_eff].contiguous()
        if w_pool.shape[0] < r_eff:
            pad = torch.full((r_eff - w_pool.shape[0],), 0.5, device=q_sorted.device, dtype=torch.float32)
            w_pool = torch.cat([w_pool, pad], dim=0)

        return q_sorted.contiguous(), w_pool[:r_eff].contiguous()

    @staticmethod
    def apply_velora_projection(
        W: torch.Tensor,
        q_orth: torch.Tensor,
        weights: torch.Tensor,
        alpha: float,
        orba_blend: float = 0.0,
        cap_activations: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            W is None
            or q_orth is None
            or weights is None
            or _is_meta(W)
            or _is_meta(q_orth)
            or _is_meta(weights)
            or W.ndim != 2
            or q_orth.numel() == 0
        ):
            return W

        dev = W.device
        W_f32 = W.float().to(dev)
        Q = q_orth.float().to(dev)
        if Q.ndim == 1:
            Q = Q.unsqueeze(0)
        w = (weights.float().to(dev).reshape(-1)[: Q.shape[0]] * float(alpha)).clone()

        beta_orba = max(0.0, min(0.50, float(orba_blend)))
        if beta_orba > 0.0:
            w[0] = min(2.0, float(w[0].item()) + beta_orba)

        out_features, in_features = W_f32.shape
        dim = Q.shape[1]

        if cap_activations is not None and not _is_meta(cap_activations) and cap_activations.ndim >= 2 and cap_activations.shape[0] >= 2:
            c_f = cap_activations.float().to(dev)
            if c_f.shape[-1] == dim:
                c_centered = c_f - c_f.mean(dim=0, keepdim=True)
                proj_var = ((c_centered @ Q.T) ** 2).mean(dim=0)
                var_med = proj_var.median().clamp_min(1e-6)
                cov_damp = (var_med / (var_med + 0.35 * proj_var)).clamp(0.45, 1.0)
                w = w * cov_damp

        w_col = w.unsqueeze(1)

        if out_features == dim:
            qw = Q @ W_f32
            return W_f32 - (Q.T @ (w_col * qw))

        if in_features == dim:
            wq = W_f32 @ Q.T
            return W_f32 - (wq @ (w_col * Q))

        return W_f32

    @classmethod
    def _apply_band(
        cls,
        surgeon,
        positive_prompts: list[str],
        negative_prompts: list[str],
        capability_prompts: list[str],
        center: float,
        spread: float,
        alpha_attn: float,
        alpha_mlp: float,
        basis_rank_attn: int,
        basis_rank_mlp: int,
        max_subspace_rank: int,
        target_energy: float,
        spectral_gamma: float,
        calibrate_norms: bool,
        use_prefix_contrast: bool,
        already_modified: set[int],
        tag: str = "VELORA",
        include_hedging_contrast: bool = False,
        extraction_cache: dict | None = None,
        max_contrast_prompts: int = 24,
        max_extra_dirs: int = 6,
        polish_after_band: bool = True,
        orba_blend: float = 0.0,
        apply_affine_bias: bool = True,
        over_refusal_prompts: list[str] | None = None,
        use_agop: bool = True,
    ) -> tuple[int, int, dict]:
        layers = getattr(surgeon, "layers", None) or []
        n_layers = len(layers)
        if n_layers == 0:
            return 0, 0, {}

        radius = max(1, int(round(spread * 1.4)))
        lo = max(0, int(center) - radius)
        hi = min(n_layers - 1, int(center) + radius)
        target_indices = [i for i in range(lo, hi + 1) if i not in already_modified]
        if not target_indices:
            return lo, hi, {}

        base_traits = _detect_architecture_traits(getattr(surgeon, "model", None), layers[target_indices[0]])
        is_moe = base_traits["is_moe"]

        layer_weights = compute_kernel_layer_weights(
            target_indices,
            peak_position=center,
            max_weight=1.0,
            min_weight=0.20,
            spread=spread,
        )

        primary_dirs = {}
        layer_cache = {}
        skip_prefix = _has_meta_params(getattr(surgeon, "model", None))
        cache_store = extraction_cache if extraction_cache is not None else {}
        ref_cap = max(1, int(max_contrast_prompts))
        hedge_cap = max(1, min(ref_cap, int(max_contrast_prompts * 0.67)))
        defl_cap = max(1, min(ref_cap, int(max_contrast_prompts * 0.50)))
        resp_prefix = getattr(getattr(surgeon, "extractor", None), "response_prefix", "")

        for idx in target_indices:
            layer_params = list(layers[idx].parameters()) if hasattr(layers[idx], "parameters") else []
            layer_dev = layer_params[0].device if layer_params else getattr(surgeon, "device", torch.device("cpu"))
            if getattr(surgeon, "extractor", None) is not None:
                surgeon.extractor.device = layer_dev

            cache_key = (idx, bool(use_prefix_contrast), bool(include_hedging_contrast), bool(use_agop))

            if cache_key in cache_store:
                c_pos, c_neg, c_prefix, c_cap, c_or = cache_store[cache_key]
                pos = c_pos.to(layer_dev) if not _is_meta(c_pos) else c_pos
                neg = c_neg.to(layer_dev) if not _is_meta(c_neg) else c_neg
                prefix_dirs = [d.to(layer_dev) if not _is_meta(d) else d for d in c_prefix]
                cap = c_cap.to(layer_dev) if (c_cap is not None and not _is_meta(c_cap)) else None
                or_states = c_or.to(layer_dev) if (c_or is not None and not _is_meta(c_or)) else None
            else:
                pos = _fuse_dual_point_states(surgeon.extractor, positive_prompts, idx, layer_dev)
                neg = _fuse_dual_point_states(surgeon.extractor, negative_prompts, idx, layer_dev)
                if not _is_meta(pos) and pos.numel() > 0:
                    pos = pos.to(layer_dev)
                if not _is_meta(neg) and neg.numel() > 0:
                    neg = neg.to(layer_dev)

                prefix_dirs = []
                if use_prefix_contrast and positive_prompts and not skip_prefix:
                    try:
                        if hasattr(RefusalSubspaceExtractor, "extract_prefix_contrast_directions"):
                            ref_dirs = RefusalSubspaceExtractor.extract_prefix_contrast_directions(
                                model=surgeon.model,
                                tokenizer=surgeon.tokenizer,
                                device=layer_dev,
                                prompts=positive_prompts[:ref_cap],
                                layer_idx=idx,
                                rank=min(4, max_subspace_rank),
                            )
                            if ref_dirs:
                                prefix_dirs.extend(list(ref_dirs))
                        if include_hedging_contrast and hasattr(RefusalSubspaceExtractor, "extract_hedging_contrast_directions"):
                            hedge_dirs = RefusalSubspaceExtractor.extract_hedging_contrast_directions(
                                model=surgeon.model,
                                tokenizer=surgeon.tokenizer,
                                device=layer_dev,
                                prompts=positive_prompts[:hedge_cap],
                                layer_idx=idx,
                                rank=min(3, max_subspace_rank),
                            )
                            if hedge_dirs:
                                prefix_dirs.extend(list(hedge_dirs))
                        if include_hedging_contrast and hasattr(RefusalSubspaceExtractor, "extract_deflection_contrast_directions"):
                            defl_dirs = RefusalSubspaceExtractor.extract_deflection_contrast_directions(
                                model=surgeon.model,
                                tokenizer=surgeon.tokenizer,
                                device=layer_dev,
                                prompts=positive_prompts[:defl_cap],
                                layer_idx=idx,
                                rank=min(2, max_subspace_rank),
                            )
                            if defl_dirs:
                                prefix_dirs.extend(list(defl_dirs))
                    except (NotImplementedError, RuntimeError) as e:
                        if "meta" in str(e).lower() or isinstance(e, NotImplementedError):
                            logger.warning(
                                f"Skipping prefix-contrast extraction at layer {idx} due to meta/offloaded tensor: {e}"
                            )
                            skip_prefix = True
                            prefix_dirs = []
                        else:
                            raise

                if use_agop and positive_prompts and not skip_prefix:
                    agop_vecs = extract_agop_subspace(
                        model=surgeon.model,
                        tokenizer=surgeon.tokenizer,
                        prompts=positive_prompts,
                        layer_idx=idx,
                        device=layer_dev,
                        rank=min(4, max_subspace_rank),
                        response_prefix=resp_prefix,
                        max_prompts=min(12, ref_cap),
                    )
                    if agop_vecs:
                        prefix_dirs.extend(agop_vecs)

                cap = None
                if capability_prompts and len(capability_prompts) >= 2:
                    raw_cap = surgeon.extractor.extract_hidden_states(capability_prompts, idx)
                    if not _is_meta(raw_cap) and raw_cap.ndim >= 2 and raw_cap.shape[0] >= 2:
                        cap = raw_cap.to(layer_dev)

                or_states = None
                if over_refusal_prompts and len(over_refusal_prompts) >= 2:
                    raw_or = surgeon.extractor.extract_hidden_states(over_refusal_prompts, idx)
                    if not _is_meta(raw_or) and raw_or.ndim >= 2 and raw_or.shape[0] >= 2:
                        or_states = raw_or.to(layer_dev)

                cache_store[cache_key] = (
                    pos.detach().cpu() if not _is_meta(pos) else pos,
                    neg.detach().cpu() if not _is_meta(neg) else neg,
                    [d.detach().cpu() if not _is_meta(d) else d for d in prefix_dirs],
                    cap.detach().cpu() if (cap is not None and not _is_meta(cap)) else None,
                    or_states.detach().cpu() if (or_states is not None and not _is_meta(or_states)) else None,
                )

            layer_cache[idx] = (pos, neg, prefix_dirs, cap, or_states)
            if prefix_dirs:
                primary_dirs[idx] = prefix_dirs[0]
            elif not _is_meta(pos) and not _is_meta(neg) and pos.numel() > 0 and neg.numel() > 0:
                n_p = min(pos.shape[0], neg.shape[0])
                if n_p > 0:
                    d = (pos[:n_p] - neg[:n_p]).float().mean(dim=0)
                    primary_dirs[idx] = d / d.norm().clamp_min(1e-8)

        interp_peak_dir = interpolate_layer_direction(primary_dirs, center) if primary_dirs else None
        extra_cap = max(1, int(max_extra_dirs))

        extracted_subspaces: dict[int, tuple] = {}
        for idx in target_indices:
            layer = layers[idx]
            layer_params = list(layer.parameters()) if hasattr(layer, "parameters") else []
            layer_dev = layer_params[0].device if layer_params else getattr(surgeon, "device", torch.device("cpu"))
            pos, neg, prefix_dirs, cap, or_states = layer_cache[idx]
            if _is_meta(pos) or _is_meta(neg) or pos.numel() == 0 or neg.numel() == 0:
                continue

            has_cap = cap is not None and not _is_meta(cap) and cap.ndim >= 2 and cap.shape[0] >= 2
            cap_attn = build_capability_basis(cap, rank=basis_rank_attn).to(layer_dev) if has_cap else None
            if not has_cap:
                cap_mlp = None
            elif basis_rank_mlp == basis_rank_attn and cap_attn is not None:
                cap_mlp = cap_attn
            else:
                cap_mlp = build_capability_basis(cap, rank=basis_rank_mlp).to(layer_dev)

            extra_cues = [d.to(layer_dev) for d in prefix_dirs[:extra_cap] if d is not None and not _is_meta(d)]
            if or_states is not None and has_cap:
                n_or = min(or_states.shape[0], cap.shape[0])
                if n_or >= 2:
                    or_diff = (or_states[:n_or] - cap[:n_or]).float().mean(dim=0)
                    if float(or_diff.norm().item()) > 1e-8:
                        extra_cues.append(or_diff / or_diff.norm().clamp_min(1e-8))

            anchor = (
                interp_peak_dir.to(layer_dev)
                if (prefix_dirs and interp_peak_dir is not None and not _is_meta(interp_peak_dir))
                else None
            )

            q_attn, w_attn = cls.extract_velora_subspace(
                pos,
                neg,
                cap_attn,
                anchor_direction=anchor,
                extra_directions=extra_cues or None,
                max_rank=max_subspace_rank,
                target_energy=target_energy,
                spectral_gamma=spectral_gamma,
            )
            q_mlp, w_mlp = cls.extract_velora_subspace(
                pos,
                neg,
                cap_mlp,
                anchor_direction=anchor,
                extra_directions=extra_cues or None,
                max_rank=max_subspace_rank,
                target_energy=target_energy,
                spectral_gamma=spectral_gamma,
            )
            extracted_subspaces[idx] = (q_attn, w_attn, q_mlp, w_mlp, cap_attn)

        coherence_weights = _compute_cross_layer_coherence(
            {idx: vals[0] for idx, vals in extracted_subspaces.items()}
        )

        layer_ranks = {}
        polished_dirs_by_layer: dict[int, tuple[list[torch.Tensor], torch.Tensor | None]] = {}

        for idx, (q_attn, w_attn, q_mlp, w_mlp, cap_attn) in extracted_subspaces.items():
            layer = layers[idx]
            targets = _filter_moe_and_norm_targets(
                surgeon._get_component_targets(layer, ["attn", "mlp"]),
                is_moe=is_moe,
            )
            if not targets:
                continue

            pos, neg, prefix_dirs, cap, _or_states = layer_cache[idx]
            traits = _detect_architecture_traits(getattr(surgeon, "model", None), layer, pos=pos, neg=neg)
            gain_mult = traits["post_norm_gain"]
            c_score = conflict_score(q_attn[0], cap_attn) if cap_attn is not None else 0.0
            snr_gate = _compute_separability_gate(pos, neg)
            coh_mult = coherence_weights.get(idx, 1.0)
            surgeon.conflict_table[f"{tag}_layer_{idx}"] = c_score

            k_mult = layer_weights.get(idx, 1.0) * snr_gate * coh_mult * gain_mult
            eff_alpha_attn = conflict_scaled_alpha(alpha_attn * k_mult, c_score)
            eff_alpha_mlp = conflict_scaled_alpha(alpha_mlp * k_mult, c_score)

            with torch.no_grad():
                for _name, W, comp_type in targets:
                    if _is_meta(W) or W.ndim != 2:
                        continue
                    w_dev = W.device
                    W_f32 = W.float().to(w_dev)
                    cap_dev = cap.to(w_dev) if (cap is not None and not _is_meta(cap)) else None
                    if comp_type == "attn":
                        W_mod = cls.apply_velora_projection(
                            W_f32,
                            q_attn.to(w_dev),
                            w_attn.to(w_dev),
                            eff_alpha_attn,
                            orba_blend=orba_blend,
                            cap_activations=cap_dev,
                        )
                    else:
                        W_mod = cls.apply_velora_projection(
                            W_f32,
                            q_mlp.to(w_dev),
                            w_mlp.to(w_dev),
                            eff_alpha_mlp,
                            orba_blend=orba_blend,
                            cap_activations=cap_dev,
                        )

                    if calibrate_norms:
                        W_mod = preserve_row_norms(W_f32, W_mod)
                    W.copy_(W_mod.to(dtype=W.dtype, device=w_dev))

                if apply_affine_bias and not _is_meta(pos) and not _is_meta(neg) and pos.numel() > 0 and neg.numel() > 0:
                    n_p = min(pos.shape[0], neg.shape[0])
                    if n_p > 0:
                        mean_shift = (pos[:n_p] - neg[:n_p]).float().mean(dim=0)
                        q_cpu = q_attn.float().to(mean_shift.device)
                        proj_shift = (mean_shift @ q_cpu.T) @ q_cpu
                        for mod in layer.modules():
                            bias = getattr(mod, "bias", None)
                            if bias is not None and not _is_meta(bias) and bias.ndim == 1 and bias.shape[0] == proj_shift.shape[0]:
                                b_dev = bias.device
                                b_mod = bias.float() - (0.35 * eff_alpha_mlp) * proj_shift.to(b_dev)
                                bias.copy_(b_mod.to(dtype=bias.dtype, device=b_dev))

            cal_delta = 0.0
            if traits["is_post_norm"]:
                cal_delta = _calibrate_post_norm_residual(
                    surgeon=surgeon,
                    layer_idx=idx,
                    positive_prompts=positive_prompts,
                    q_orth=q_attn,
                    weights=w_attn,
                    targets=targets,
                    calibrate_norms=calibrate_norms,
                )

            layer_ranks[idx] = {
                "attn_rank": int(q_attn.shape[0]),
                "mlp_rank": int(q_mlp.shape[0]),
                "snr_gate": snr_gate,
                "coherence": coh_mult,
                "post_norm_gain": gain_mult,
                "post_norm_cal_delta": cal_delta,
            }

            if polish_after_band and prefix_dirs and not skip_prefix:
                polished_dirs_by_layer[idx] = (prefix_dirs, cap)

            already_modified.add(idx)

        if polish_after_band and polished_dirs_by_layer and not skip_prefix:
            for p_idx, (p_dirs, p_cap) in polished_dirs_by_layer.items():
                cls.polish_null_space_lbfgs(
                    surgeon=surgeon,
                    target_directions=p_dirs,
                    capability_prompts=capability_prompts,
                    layer_indices=[p_idx],
                    basis_rank=basis_rank_attn,
                    step_alpha=min(0.25, alpha_attn * 0.14),
                    lbfgs_steps=4,
                    calibrate_norms=calibrate_norms,
                    precomputed_cap=p_cap,
                )

        return lo, hi, layer_ranks

    @classmethod
    def run_velora_surgery(
        cls,
        surgeon,
        positive_prompts: list[str],
        negative_prompts: list[str],
        capability_prompts: list[str],
        peak_position: float | None = None,
        spread: float = 2.5,
        alpha_attn: float = 1.25,
        alpha_mlp: float = 1.05,
        basis_rank_attn: int = 12,
        basis_rank_mlp: int = 10,
        max_subspace_rank: int = 12,
        target_energy: float = 0.94,
        spectral_gamma: float = 0.45,
        calibrate_norms: bool = True,
        use_prefix_contrast: bool = True,
        anti_hedge: bool = True,
        dual_band: bool = True,
        late_alpha_scale: float = 0.60,
        extraction_cache: dict | None = None,
        polish_after_band: bool = True,
        orba_blend: float = 0.0,
        coherence_guard: bool = True,
        over_refusal_prompts: list[str] | None = None,
        use_agop: bool = True,
    ) -> dict:
        n_layers = len(getattr(surgeon, "layers", None) or [])
        if n_layers == 0:
            return {
                "modifier": "velora",
                "layers_modified": 0,
                "peak_position": 0.0,
                "layer_range": [0, 0],
                "layer_ranks": {},
            }

        resolved_peak = (
            float(layer_index_from_ratio(n_layers, 0.62))
            if peak_position is None
            else float(peak_position)
        )
        center = max(0.0, min(float(n_layers - 1), resolved_peak))

        is_compact = n_layers <= 28
        eff_max_rank = max(int(max_subspace_rank), 16 if n_layers >= 32 else (12 if not is_compact else 8))
        eff_spread = max(float(spread), 2.5 if is_compact else 3.2)
        eff_alpha_attn = float(alpha_attn) * (0.85 if is_compact else 1.0)
        eff_alpha_mlp = float(alpha_mlp) * (0.85 if is_compact else 1.0)

        snap = None
        all_indices = list(range(n_layers))
        if coherence_guard and capability_prompts and getattr(surgeon, "tokenizer", None) is not None and not _has_meta_params(getattr(surgeon, "model", None)):
            snap = LayerWeightSnapshot()
            snap.capture(surgeon.model, all_indices, store_on_cpu=True)

        modified: set[int] = set()
        cache_store = extraction_cache if extraction_cache is not None else {}

        lo, hi, ranks = cls._apply_band(
            surgeon=surgeon,
            positive_prompts=positive_prompts,
            negative_prompts=negative_prompts,
            capability_prompts=capability_prompts,
            center=center,
            spread=eff_spread,
            alpha_attn=eff_alpha_attn,
            alpha_mlp=eff_alpha_mlp,
            basis_rank_attn=basis_rank_attn,
            basis_rank_mlp=basis_rank_mlp,
            max_subspace_rank=eff_max_rank,
            target_energy=target_energy,
            spectral_gamma=spectral_gamma,
            calibrate_norms=calibrate_norms,
            use_prefix_contrast=use_prefix_contrast,
            already_modified=modified,
            tag="VELORA_Primary",
            include_hedging_contrast=anti_hedge,
            extraction_cache=cache_store,
            polish_after_band=polish_after_band,
            orba_blend=orba_blend,
            over_refusal_prompts=over_refusal_prompts,
            use_agop=use_agop,
        )

        if dual_band and n_layers >= 12:
            late_target = float(layer_index_from_ratio(n_layers, 0.78))
            min_late = min(float(n_layers - 2), center + max(3.0, n_layers * 0.12))
            late_center = min(float(n_layers - 2), max(min_late, late_target))
            _, _, late_ranks = cls._apply_band(
                surgeon=surgeon,
                positive_prompts=positive_prompts,
                negative_prompts=negative_prompts,
                capability_prompts=capability_prompts,
                center=late_center,
                spread=max(1.5, eff_spread * 0.65),
                alpha_attn=eff_alpha_attn * float(late_alpha_scale),
                alpha_mlp=eff_alpha_mlp * float(late_alpha_scale),
                basis_rank_attn=basis_rank_attn,
                basis_rank_mlp=basis_rank_mlp,
                max_subspace_rank=eff_max_rank,
                target_energy=target_energy,
                spectral_gamma=spectral_gamma,
                calibrate_norms=calibrate_norms,
                use_prefix_contrast=use_prefix_contrast,
                already_modified=modified,
                tag="VELORA_Late",
                include_hedging_contrast=anti_hedge,
                extraction_cache=cache_store,
                polish_after_band=polish_after_band,
                orba_blend=orba_blend,
                over_refusal_prompts=over_refusal_prompts,
                use_agop=use_agop,
            )
            ranks.update(late_ranks)

        glitch_detected = False
        if snap is not None and capability_prompts:
            try:
                resp_prefix = getattr(getattr(surgeon, "extractor", None), "response_prefix", "")
                probe_prompt = capability_prompts[0]
                inp = _move_batch_to_device(
                    surgeon.tokenizer(
                        format_chat_prompt(surgeon.tokenizer, probe_prompt, response_prefix=resp_prefix),
                        return_tensors="pt",
                    ),
                    surgeon.device,
                )
                in_len = inp["input_ids"].shape[1]
                surgeon.model.eval()
                with torch.no_grad():
                    gen = surgeon.model.generate(**inp, max_new_tokens=96, do_sample=False)
                sample_out = _decode_generated_tokens(surgeon.tokenizer, gen[0][in_len:])
                coh_check = check_generation_coherence(sample_out)
                if coh_check["is_degenerate"]:
                    glitch_detected = True
                    logger.warning(
                        f"VELORA coherence guard triggered ({coh_check['reason']}); rolling back and re-running at 0.65x alpha."
                    )
                    snap.restore(surgeon.model, all_indices)
                    return cls.run_velora_surgery(
                        surgeon=surgeon,
                        positive_prompts=positive_prompts,
                        negative_prompts=negative_prompts,
                        capability_prompts=capability_prompts,
                        peak_position=peak_position,
                        spread=spread,
                        alpha_attn=float(alpha_attn) * 0.65,
                        alpha_mlp=float(alpha_mlp) * 0.65,
                        basis_rank_attn=basis_rank_attn,
                        basis_rank_mlp=basis_rank_mlp,
                        max_subspace_rank=max(4, int(max_subspace_rank) // 2),
                        target_energy=target_energy,
                        spectral_gamma=spectral_gamma,
                        calibrate_norms=calibrate_norms,
                        use_prefix_contrast=use_prefix_contrast,
                        anti_hedge=anti_hedge,
                        dual_band=dual_band,
                        late_alpha_scale=late_alpha_scale,
                        extraction_cache=cache_store,
                        polish_after_band=False,
                        orba_blend=0.0,
                        coherence_guard=False,
                        over_refusal_prompts=over_refusal_prompts,
                        use_agop=use_agop,
                    )
            except Exception:
                pass

        logger.info(
            format_velora_telemetry(
                active=True,
                mode="velora_v3_2_strategic",
                reaction="SURGERY_APPLIED",
                layers=f"{min(modified) if modified else lo}..{max(modified) if modified else hi}",
                extra=f"modified={len(modified)}",
            )
        )

        return {
            "modifier": "velora",
            "layers_modified": len(modified),
            "peak_position": center,
            "layer_range": [lo, hi],
            "layer_ranks": ranks,
            "glitch_recovered": glitch_detected,
        }

    @classmethod
    def hybrid_velora_ara(
        cls,
        surgeon,
        positive_prompts: list[str],
        negative_prompts: list[str],
        capability_prompts: list[str],
        refusal_threshold: float = 0.15,
        micro_ara_rank: int = 4,
        micro_ara_steps: int = 12,
        calibrate_norms: bool = True,
        over_refusal_prompts: list[str] | None = None,
    ) -> dict:
        """
        Hybrid VELORA + Micro-ARA residual fallback: runs fast VELORA surgery first,
        measures remaining refusal probability on probe prompts, and fits a compact
        rank-r residual adapter (B @ A) only if residual refusal > refusal_threshold.
        """
        base_res = cls.run_velora_surgery(
            surgeon=surgeon,
            positive_prompts=positive_prompts,
            negative_prompts=negative_prompts,
            capability_prompts=capability_prompts,
            over_refusal_prompts=over_refusal_prompts,
            use_agop=True,
        )

        layers = getattr(surgeon, "layers", None) or []
        n_layers = len(layers)
        if n_layers == 0 or not positive_prompts or getattr(surgeon, "tokenizer", None) is None or _has_meta_params(getattr(surgeon, "model", None)):
            base_res["hybrid_micro_ara_triggered"] = False
            return base_res

        ref_ids = _first_token_ids(surgeon.tokenizer, REFUSAL_START_TOKENS)
        cmp_ids = _first_token_ids(surgeon.tokenizer, COMPLIANCE_START_TOKENS)
        if not ref_ids or not cmp_ids:
            base_res["hybrid_micro_ara_triggered"] = False
            return base_res

        ref_t = torch.tensor(ref_ids, dtype=torch.long)
        cmp_t = torch.tensor(cmp_ids, dtype=torch.long)
        resp_prefix = getattr(getattr(surgeon, "extractor", None), "response_prefix", "")
        probes = positive_prompts[: min(6, len(positive_prompts))]

        probs: list[float] = []
        surgeon.model.eval()
        with torch.no_grad():
            for p in probes:
                inp = _move_batch_to_device(
                    surgeon.tokenizer(format_chat_prompt(surgeon.tokenizer, p, response_prefix=resp_prefix), return_tensors="pt"),
                    surgeon.device,
                )
                lp_p = _first_token_refusal_prob(surgeon.model, surgeon.tokenizer, inp, ref_t, cmp_t)
                if lp_p is not None:
                    probs.append(lp_p)

        mean_prob = (sum(probs) / len(probs)) if probs else 0.0
        base_res["post_velora_refusal_prob"] = round(mean_prob, 4)

        if mean_prob <= float(refusal_threshold):
            base_res["hybrid_micro_ara_triggered"] = False
            return base_res

        # Micro-ARA residual polish on top 2 mid-late layers
        peak_idx = int(round(base_res.get("peak_position", layer_index_from_ratio(n_layers, 0.62))))
        late_idx = min(n_layers - 1, max(peak_idx + 1, int(layer_index_from_ratio(n_layers, 0.78))))
        stubborn_layers = sorted({max(0, min(n_layers - 1, peak_idx)), late_idx})
        r_micro = max(1, min(int(micro_ara_rank), 8))
        steps_micro = max(2, min(int(micro_ara_steps), 24))

        # Extract refusal direction at stubborn layers for target alignment
        fitted_modules = 0
        for idx in stubborn_layers:
            layer = layers[idx]
            params = list(layer.parameters()) if hasattr(layer, "parameters") else []
            if any(_is_meta(p) for p in params):
                continue
            layer_dev = params[0].device if params else getattr(surgeon, "device", torch.device("cpu"))
            agop_d = extract_agop_subspace(
                model=surgeon.model,
                tokenizer=surgeon.tokenizer,
                prompts=probes,
                layer_idx=idx,
                device=layer_dev,
                rank=r_micro,
                response_prefix=resp_prefix,
                max_prompts=len(probes),
            )
            if not agop_d:
                continue
            Q_sub = _orthonormalize_rows(torch.stack(agop_d, dim=0), agop_d[0]).to(layer_dev)
            prior_w = torch.linspace(1.0, 0.5, steps=Q_sub.shape[0], device=layer_dev, dtype=torch.float32)

            targets = _filter_moe_and_norm_targets(
                surgeon._get_component_targets(layer, ["attn", "mlp"]),
                is_moe=_detect_architecture_traits(getattr(surgeon, "model", None), layer)["is_moe"],
            )
            for _name, W, _comp_type in targets:
                if _is_meta(W) or W.ndim != 2:
                    continue
                w_dev = W.device
                W0 = W.detach().float().to(w_dev)
                W_mod, _ = _optimize_axis_weights_for_matrix(
                    W0=W0,
                    Q=Q_sub.to(w_dev),
                    w_prior=prior_w.to(w_dev),
                    steps=steps_micro,
                    lr=0.08,
                    kl_weight=0.30,
                    max_axis_delta=0.45,
                    calibrate_norms=calibrate_norms,
                    optimizer_kind="adam",
                )
                if W_mod is not None:
                    with torch.no_grad():
                        W.copy_(W_mod.to(device=w_dev, dtype=W.dtype))
                    fitted_modules += 1

        base_res["hybrid_micro_ara_triggered"] = True
        base_res["hybrid_micro_ara_layers"] = stubborn_layers
        base_res["hybrid_micro_ara_modules_fitted"] = fitted_modules
        return base_res

    @classmethod
    def optimize_velora_pareto(
        cls,
        surgeon,
        positive_prompts: list[str],
        negative_prompts: list[str],
        capability_prompts: list[str],
        n_trials: int = 12,
        max_kl: float = 1.10,
        seed: int = 42,
        extraction_cache: dict | None = None,
        max_eval_prompts: int = 16,
        max_cap_prompts: int = 4,
    ) -> dict:
        if optuna is None:
            raise ImportError("optuna is required for optimize_velora_pareto. Run `pip install optuna`.")

        n_layers = len(getattr(surgeon, "layers", None) or [])
        all_indices = list(range(n_layers))
        snap = LayerWeightSnapshot()
        snap.capture(surgeon.model, all_indices, store_on_cpu=True)
        shared_cache = extraction_cache if extraction_cache is not None else {}

        resp_prefix = getattr(getattr(surgeon, "extractor", None), "response_prefix", "")
        cap_subset = (capability_prompts or [])[: max(1, int(max_cap_prompts))]
        cap_inputs = [
            _move_batch_to_device(
                surgeon.tokenizer(
                    format_chat_prompt(surgeon.tokenizer, cp, response_prefix=resp_prefix),
                    return_tensors="pt",
                ),
                surgeon.device,
            )
            for cp in cap_subset
        ]
        eval_prompts = (positive_prompts or [])[::2][: max(1, int(max_eval_prompts))]
        eval_inputs = [
            _move_batch_to_device(
                surgeon.tokenizer(
                    format_chat_prompt(surgeon.tokenizer, p, response_prefix=resp_prefix),
                    return_tensors="pt",
                ),
                surgeon.device,
            )
            for p in eval_prompts
        ]

        ref_ids = _first_token_ids(surgeon.tokenizer, REFUSAL_START_TOKENS) if getattr(surgeon, "tokenizer", None) else []
        cmp_ids = _first_token_ids(surgeon.tokenizer, COMPLIANCE_START_TOKENS) if getattr(surgeon, "tokenizer", None) else []
        ref_t = torch.tensor(ref_ids, dtype=torch.long)
        cmp_t = torch.tensor(cmp_ids, dtype=torch.long)

        base_logits_list = []
        surgeon.model.eval()
        with torch.no_grad():
            for inp in cap_inputs:
                out = surgeon.model(**inp)
                logits = getattr(out, "logits", None)
                if logits is not None and not _is_meta(logits):
                    base_logits_list.append(logits[:, -1, :].detach().cpu())

        stage1_records: list[dict] = []

        def objective(trial):
            snap.restore(surgeon.model, all_indices)

            lo_pos = max(0.0, n_layers * 0.35)
            hi_pos = max(lo_pos + 1.0, float(max(0, n_layers - 1)))
            if n_layers > 1:
                hi_pos = min(float(n_layers - 1), max(lo_pos + 0.5, n_layers * 0.72))
            peak_pos = trial.suggest_float("peak_position", lo_pos, hi_pos)
            spread = trial.suggest_float("spread", 1.5, 3.8)
            a_attn = trial.suggest_float("alpha_attn", 0.75, 1.85)
            a_mlp = trial.suggest_float("alpha_mlp", 0.65, 1.65)
            sub_rank = trial.suggest_int("max_subspace_rank", 4, 16, step=4)
            t_energy = trial.suggest_float("target_energy", 0.88, 0.97)
            gamma = trial.suggest_float("spectral_gamma", 0.30, 0.75)
            late_scale = trial.suggest_float("late_alpha_scale", 0.40, 0.85)
            orba_b = trial.suggest_float("orba_blend", 0.0, 0.25)

            cls.run_velora_surgery(
                surgeon=surgeon,
                positive_prompts=positive_prompts,
                negative_prompts=negative_prompts,
                capability_prompts=capability_prompts,
                peak_position=peak_pos,
                spread=spread,
                alpha_attn=a_attn,
                alpha_mlp=a_mlp,
                max_subspace_rank=sub_rank,
                target_energy=t_energy,
                spectral_gamma=gamma,
                late_alpha_scale=late_scale,
                extraction_cache=shared_cache,
                polish_after_band=False,
                orba_blend=orba_b,
                coherence_guard=False,
            )

            logprob_probs: list[float] = []
            surgeon.model.eval()
            with torch.no_grad():
                for inp in eval_inputs:
                    lp_p = _first_token_refusal_prob(surgeon.model, surgeon.tokenizer, inp, ref_t, cmp_t)
                    if lp_p is not None:
                        logprob_probs.append(lp_p)

            fast_refusal = (sum(logprob_probs) / len(logprob_probs)) if logprob_probs else 0.5

            kl_scores = []
            with torch.no_grad():
                for idx_c, inp in enumerate(cap_inputs):
                    if idx_c >= len(base_logits_list):
                        break
                    out = surgeon.model(**inp)
                    abl_logits = getattr(out, "logits", None)
                    if abl_logits is not None and not _is_meta(abl_logits):
                        kl_scores.append(compute_kl_divergence(base_logits_list[idx_c], abl_logits[:, -1, :].detach().cpu()))

            mean_kl = sum(kl_scores) / max(1, len(kl_scores)) if kl_scores else 0.0
            if math.isnan(mean_kl) or math.isinf(mean_kl):
                mean_kl = max_kl * 2.0

            summary = {
                "trial": trial.number,
                "refusal_rate": round(fast_refusal, 4),
                "hedging_rate": 0.0,
                "glitch_rate": 0.0,
                "mean_kl": round(mean_kl, 4),
                "params": dict(trial.params),
            }
            score = _trial_objective_score(summary, kl_weight=0.25, max_kl=max_kl)
            summary["objective_score"] = round(score, 6)
            stage1_records.append(summary)
            return score

        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=max(1, int(n_trials)))

        by_score = sorted(stage1_records, key=lambda t: t["objective_score"])
        by_refusal = sorted(stage1_records, key=lambda t: t["refusal_rate"])
        by_kl = sorted(stage1_records, key=lambda t: t["mean_kl"])

        selected_candidates: list[dict] = []
        seen_trials: set[int] = set()
        for pool in (by_score[:2], by_refusal[:1], by_kl[:1], by_score[2:4]):
            for item in pool:
                if item["trial"] not in seen_trials and len(selected_candidates) < 4:
                    seen_trials.add(item["trial"])
                    selected_candidates.append(item)

        verified_trials: list[dict] = []
        max_tokens = 192 if n_layers <= 28 else 256

        for cand in selected_candidates:
            p = cand["params"]
            snap.restore(surgeon.model, all_indices)
            cls.run_velora_surgery(
                surgeon=surgeon,
                positive_prompts=positive_prompts,
                negative_prompts=negative_prompts,
                capability_prompts=capability_prompts,
                peak_position=p["peak_position"],
                spread=p["spread"],
                alpha_attn=p["alpha_attn"],
                alpha_mlp=p["alpha_mlp"],
                max_subspace_rank=p["max_subspace_rank"],
                target_energy=p["target_energy"],
                spectral_gamma=p["spectral_gamma"],
                late_alpha_scale=p["late_alpha_scale"],
                extraction_cache=shared_cache,
                polish_after_band=True,
                orba_blend=p.get("orba_blend", 0.0),
                coherence_guard=False,
            )

            refusals = 0
            hedgings = 0
            glitches = 0
            stage2_lp: list[float] = []
            surgeon.model.eval()
            with torch.no_grad():
                for inp in eval_inputs:
                    lp_p = _first_token_refusal_prob(surgeon.model, surgeon.tokenizer, inp, ref_t, cmp_t)
                    if lp_p is not None:
                        stage2_lp.append(lp_p)
                    in_len = inp["input_ids"].shape[1]
                    gen = surgeon.model.generate(**inp, max_new_tokens=max_tokens, do_sample=False)
                    text = _decode_generated_tokens(surgeon.tokenizer, gen[0][in_len:])
                    assessment = classify_response(text)
                    coh = check_generation_coherence(text)
                    if assessment.is_refusal:
                        refusals += 1
                    if assessment.is_hedging or assessment.is_moralizing or assessment.is_deflection:
                        hedgings += 1
                    if coh["is_degenerate"]:
                        glitches += 1

            denom = max(1, len(eval_inputs))
            gen_refusal = refusals / denom
            lp_mean = (sum(stage2_lp) / len(stage2_lp)) if stage2_lp else gen_refusal
            blended_refusal = 0.70 * gen_refusal + 0.30 * lp_mean

            verified_entry = {
                "trial": cand["trial"],
                "refusal_rate": round(blended_refusal, 4),
                "hedging_rate": round(hedgings / denom, 4),
                "glitch_rate": round(glitches / denom, 4),
                "mean_kl": cand["mean_kl"],
                "params": dict(p),
            }
            verified_entry["objective_score"] = round(
                _trial_objective_score(verified_entry, kl_weight=0.25, max_kl=max_kl), 6
            )
            verified_trials.append(verified_entry)

        snap.restore(surgeon.model, all_indices)
        final_pool = verified_trials if verified_trials else stage1_records
        frontier = _compute_pareto_frontier(final_pool, max_kl=max_kl)
        best_trial = min(final_pool, key=lambda t: _trial_objective_score(t, kl_weight=0.25, max_kl=max_kl))
        best_entry = dict(best_trial["params"])
        best_entry["mean_kl"] = best_trial["mean_kl"]
        best_entry["refusal_rate"] = best_trial["refusal_rate"]
        best_entry["hedging_rate"] = best_trial.get("hedging_rate", 0.0)
        best_entry["glitch_rate"] = best_trial.get("glitch_rate", 0.0)
        best_entry["objective_score"] = best_trial["objective_score"]

        return {
            "best_params": dict(best_trial["params"]),
            "best_score": best_trial["objective_score"],
            "best": best_entry,
            "pareto_frontier": frontier,
            "trials": final_pool,
        }
cat <<'EOF' > latent_ablate/velora.py
import logging
import math
import re

try:
    import optuna
except ImportError:
    optuna = None
import torch

from latent_ablate.arch import (
    format_chat_prompt,
    layer_index_from_ratio,
)
from latent_ablate.diagnostics import compute_kl_divergence
from latent_ablate.erasure import (
    _is_meta,
    build_capability_basis,
    conflict_scaled_alpha,
    conflict_score,
    preserve_row_norms,
)
from latent_ablate.optimizer import compute_kernel_layer_weights, interpolate_layer_direction
from latent_ablate.plugins.scorer import COMPLIANCE_START_TOKENS, REFUSAL_START_TOKENS, _first_token_ids
from latent_ablate.reaction import classify_response, format_velora_telemetry
from latent_ablate.refusal_subspace import RefusalSubspaceExtractor
from latent_ablate.snapshot import LayerWeightSnapshot
from latent_ablate.subspace import _winsorize

logger = logging.getLogger("LatentAblate.VELORA")


def _move_batch_to_device(batch, device):
    if batch is None:
        return batch
    if hasattr(batch, "to"):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
    return batch


def _decode_generated_tokens(tokenizer, token_ids) -> str:
    if hasattr(tokenizer, "decode"):
        return tokenizer.decode(token_ids, skip_special_tokens=True).strip()
    vals = token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)
    return " ".join(str(x) for x in vals if x not in (0, 1)).strip()


def _project_orthogonal(vecs: torch.Tensor, basis: torch.Tensor | None) -> torch.Tensor:
    """Project row vectors [k, dim] orthogonal to row-orthonormal basis [m, dim]."""
    if vecs is None or vecs.numel() == 0 or _is_meta(vecs):
        return vecs
    if basis is None or basis.numel() == 0 or _is_meta(basis):
        return vecs.float()
    b = basis.to(device=vecs.device, dtype=torch.float32)
    if b.ndim == 1:
        b = b.unsqueeze(0)
    if b.shape[0] == 0 or b.shape[-1] != vecs.shape[-1]:
        return vecs.float()
    try:
        q_b, _ = torch.linalg.qr(b.T, mode="reduced")
    except RuntimeError:
        q_b, _ = torch.linalg.qr(b.T.cpu(), mode="reduced")
        q_b = q_b.to(vecs.device)
    basis_orth = q_b.T
    v_f = vecs.float()
    proj = (v_f @ basis_orth.T) @ basis_orth
    return v_f - proj


def _orthonormalize_rows(rows: torch.Tensor, fallback_vec: torch.Tensor, tol: float = 1e-5) -> torch.Tensor:
    """Return an orthonormal row basis [r_eff, dim] via thin QR with deterministic sign & fallback."""
    fb = fallback_vec.float()
    if _is_meta(fb) or fb.numel() == 0:
        hidden = fb.shape[-1] if (fb is not None and fb.ndim >= 1 and fb.shape[-1] > 0) else 1
        return torch.eye(1, hidden, dtype=torch.float32)

    fb_norm = float(fb.norm().item())
    if fb_norm <= 1e-12 or math.isnan(fb_norm) or math.isinf(fb_norm):
        fb_unit = torch.zeros_like(fb)
        fb_unit[0] = 1.0
    else:
        fb_unit = fb / fb_norm
    fb_unit = fb_unit.unsqueeze(0).contiguous()

    if rows is None or rows.numel() == 0 or _is_meta(rows):
        return fb_unit

    r_f = rows.float()
    if r_f.ndim == 1:
        r_f = r_f.unsqueeze(0)
    if r_f.shape[-1] != fb_unit.shape[-1]:
        return fb_unit

    try:
        q, r_mat = torch.linalg.qr(r_f.T, mode="reduced")
    except RuntimeError:
        q, r_mat = torch.linalg.qr(r_f.T.cpu(), mode="reduced")
        q, r_mat = q.to(r_f.device), r_mat.to(r_f.device)

    if _is_meta(q) or _is_meta(r_mat):
        return fb_unit

    diag_vals = torch.diag(r_mat)
    diag_abs = diag_vals.abs()
    if _is_meta(diag_abs):
        return fb_unit

    keep = diag_abs > float(tol)
    if _is_meta(keep) or not bool(keep.any().item()):
        return fb_unit

    q_kept = q[:, keep]
    signs = torch.sign(diag_vals[keep])
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return (q_kept * signs.unsqueeze(0)).T.contiguous()


def _has_meta_params(model) -> bool:
    if model is None or not hasattr(model, "parameters"):
        return False
    for p in model.parameters():
        if _is_meta(p):
            return True
    return False


def _safe_float_item(t: torch.Tensor, default: float = 0.0) -> float:
    if t is None or _is_meta(t) or t.numel() == 0:
        return default
    val = float(t.detach().item())
    if math.isnan(val) or math.isinf(val):
        return default
    return val


def _extract_cone_directions(
    diff_rows: torch.Tensor,
    n_clusters: int = 4,
    iters: int = 12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Deterministic spherical k-means over contrastive difference vectors [N, dim]
    to discover multi-modal refusal cone centroids (direct, framing, edge-case)
    and their normalized cluster weights.
    """
    if diff_rows is None or _is_meta(diff_rows) or diff_rows.ndim != 2 or diff_rows.shape[0] < 3:
        return torch.empty(0), torch.empty(0)

    norms = diff_rows.norm(dim=-1, keepdim=True)
    valid_mask = (norms.squeeze(-1) > 1e-8) & torch.isfinite(norms.squeeze(-1))
    if int(valid_mask.sum().item()) < 3:
        return torch.empty(0), torch.empty(0)

    unit_diffs = diff_rows[valid_mask] / norms[valid_mask]
    n_samples, _dim = unit_diffs.shape
    k = max(2, min(int(n_clusters), n_samples // 2))

    seeds = [0]
    min_dists = 1.0 - (unit_diffs @ unit_diffs[0]).clamp(-1.0, 1.0)
    for _ in range(1, k):
        next_idx = int(torch.argmax(min_dists).item())
        seeds.append(next_idx)
        d_new = 1.0 - (unit_diffs @ unit_diffs[next_idx]).clamp(-1.0, 1.0)
        min_dists = torch.minimum(min_dists, d_new)

    centroids = unit_diffs[seeds].clone()
    assigns = torch.zeros(n_samples, dtype=torch.long, device=unit_diffs.device)

    for _ in range(max(1, int(iters))):
        sims = unit_diffs @ centroids.T
        new_assigns = torch.argmax(sims, dim=1)
        if torch.equal(new_assigns, assigns):
            break
        assigns = new_assigns
        for c_idx in range(k):
            members = unit_diffs[assigns == c_idx]
            if members.shape[0] > 0:
                mean_c = members.mean(dim=0)
                c_norm = float(mean_c.norm().item())
                if c_norm > 1e-8:
                    centroids[c_idx] = mean_c / c_norm

    counts = torch.bincount(assigns, minlength=k).float()
    keep = counts > 0
    if not bool(keep.any().item()):
        return torch.empty(0), torch.empty(0)

    kept_centroids = centroids[keep]
    kept_weights = (counts[keep] / counts[keep].max().clamp_min(1.0)).clamp(0.35, 1.0)
    return kept_centroids.contiguous(), kept_weights.contiguous()


def extract_agop_subspace(
    model,
    tokenizer,
    prompts: list[str],
    layer_idx: int,
    device: torch.device,
    rank: int = 4,
    response_prefix: str = "",
    max_prompts: int = 12,
) -> list[torch.Tensor]:
    """
    Computes top eigenvectors of the Average Gradient Outer Product (AGOP)
    M_l = (1/N) sum_i g_i g_i^T where g_i = d(log_refusal - log_compliance) / d(h_l)
    at the last token position of layer `layer_idx`.
    """
    if (
        model is None
        or tokenizer is None
        or not prompts
        or _has_meta_params(model)
    ):
        return []

    ref_ids = _first_token_ids(tokenizer, REFUSAL_START_TOKENS)
    cmp_ids = _first_token_ids(tokenizer, COMPLIANCE_START_TOKENS)
    if not ref_ids or not cmp_ids:
        return []

    layers = getattr(getattr(model, "model", model), "layers", None)
    if layers is None and hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
    if layers is None or layer_idx < 0 or layer_idx >= len(layers):
        return []

    target_layer = layers[layer_idx]
    grad_samples: list[torch.Tensor] = []
    subset = prompts[: max(1, int(max_prompts))]

    for p in subset:
        inp = _move_batch_to_device(
            tokenizer(format_chat_prompt(tokenizer, p, response_prefix=response_prefix), return_tensors="pt"),
            device,
        )
        captured: list[torch.Tensor] = []

        def _fwd_hook(_mod, _args, output):
            h = output[0] if isinstance(output, tuple) else output
            if h is not None and not _is_meta(h) and h.requires_grad:
                captured.append(h)

        def _pre_hook(_mod, args):
            if args and isinstance(args[0], torch.Tensor) and not _is_meta(args[0]):
                x = args[0].detach().float().requires_grad_(True)
                captured.append(x)
                return (x,) + args[1:]
            return args

        handle = target_layer.register_forward_pre_hook(_pre_hook)
        try:
            model.zero_grad(set_to_none=True)
            out = model(**inp)
            logits = getattr(out, "logits", None)
            if logits is None or _is_meta(logits) or not captured:
                continue
            last_logits = logits[0, -1, :].float()
            log_probs = torch.log_softmax(last_logits, dim=-1)
            ref_t = torch.tensor([i for i in ref_ids if i < log_probs.shape[0]], device=log_probs.device, dtype=torch.long)
            cmp_t = torch.tensor([i for i in cmp_ids if i < log_probs.shape[0]], device=log_probs.device, dtype=torch.long)
            if ref_t.numel() == 0 or cmp_t.numel() == 0:
                continue
            margin = torch.logsumexp(log_probs[ref_t], dim=0) - torch.logsumexp(log_probs[cmp_t], dim=0)
            margin.backward()

            x_var = captured[0]
            if x_var.grad is not None and not _is_meta(x_var.grad):
                g_vec = x_var.grad[0, -1, :].detach().float().reshape(-1)
                g_norm = float(g_vec.norm().item())
                if g_norm > 1e-8 and torch.isfinite(g_vec).all():
                    grad_samples.append(g_vec / g_norm)
        except Exception:
            continue
        finally:
            handle.remove()
            model.zero_grad(set_to_none=True)

    if not grad_samples:
        return []

    G = torch.stack(grad_samples, dim=0)  # [N, d]
    agop = (G.T @ G) / float(G.shape[0])
    try:
        evals, evecs = torch.linalg.eigh(agop.cpu())
    except RuntimeError:
        return []

    order = torch.argsort(evals, descending=True)
    k = max(1, min(int(rank), evecs.shape[1]))
    top_vecs = evecs[:, order[:k]].T.contiguous().to(device)
    return [top_vecs[i] for i in range(top_vecs.shape[0])]


def _compute_separability_gate(
    pos: torch.Tensor,
    neg: torch.Tensor,
    floor: float = 0.45,
) -> float:
    """
    Computes a smooth [floor, 1.0] Fisher/SNR gate from contrastive mean separation
    so low-signal layers are not over-ablated.
    """
    if pos is None or neg is None or _is_meta(pos) or _is_meta(neg):
        return 1.0
    n_p = min(pos.shape[0], neg.shape[0])
    if n_p == 0 or pos.numel() == 0 or neg.numel() == 0:
        return 1.0

    p_f = pos[:n_p].float()
    n_f = neg[:n_p].float()
    diff_mean = (p_f - n_f).mean(dim=0)
    signal_norm = float(diff_mean.norm().item())

    act_scale = max(1e-6, 0.5 * (float(p_f.norm(dim=-1).mean().item()) + float(n_f.norm(dim=-1).mean().item())))
    snr = min(2.0, signal_norm / act_scale)
    clamped_floor = max(0.0, min(0.95, float(floor)))
    return round(clamped_floor + (1.0 - clamped_floor) * min(1.0, snr), 4)


def _compute_cross_layer_coherence(
    subspaces_by_layer: dict[int, torch.Tensor],
) -> dict[int, float]:
    """
    Measures principal-angle subspace coherence across adjacent layers via singular
    values of Q_l @ Q_{l+1}^T.
    """
    ordered = sorted(subspaces_by_layer.keys())
    if len(ordered) <= 1:
        return {idx: 1.0 for idx in ordered}

    coherence_scores: dict[int, float] = {}
    for pos_i, idx in enumerate(ordered):
        q_cur = subspaces_by_layer[idx]
        if q_cur is None or _is_meta(q_cur) or q_cur.numel() == 0:
            coherence_scores[idx] = 1.0
            continue
        q_cur_cpu = q_cur.detach().float().cpu()
        neighbor_overlaps: list[float] = []
        for offset in (-1, 1):
            n_pos = pos_i + offset
            if 0 <= n_pos < len(ordered):
                q_nbr = subspaces_by_layer[ordered[n_pos]]
                if q_nbr is not None and not _is_meta(q_nbr) and q_nbr.numel() > 0:
                    q_nbr_cpu = q_nbr.detach().float().cpu()
                    if q_cur_cpu.shape[-1] == q_nbr_cpu.shape[-1]:
                        prod = q_cur_cpu @ q_nbr_cpu.T
                        svals = torch.linalg.svdvals(prod)
                        top_k = min(3, svals.shape[0])
                        neighbor_overlaps.append(float(svals[:top_k].mean().item()))
        if not neighbor_overlaps:
            coherence_scores[idx] = 1.0
        else:
            mean_ov = max(0.0, min(1.0, sum(neighbor_overlaps) / len(neighbor_overlaps)))
            coherence_scores[idx] = round(0.75 + 0.40 * mean_ov, 4)

    return coherence_scores


def _detect_architecture_traits(model, layer=None, pos: torch.Tensor | None = None, neg: torch.Tensor | None = None) -> dict:
    """
    Detects post-norm / Gemma-style norm dynamics and MoE layers, computing a
    gain-compensated refusal multiplier so post-norm residual re-normalization
    does not let refusal leak through.
    """
    cfg = getattr(model, "config", None)
    model_type = str(getattr(cfg, "model_type", "") or "").lower()
    arch_list = [str(a).lower() for a in (getattr(cfg, "architectures", None) or [])]
    arch_blob = " ".join([model_type] + arch_list)

    is_post_norm = any(k in arch_blob for k in ("gemma2", "gemma3", "gemma4", "olmo2", "olmo3", "post_norm"))
    num_experts = int(
        getattr(cfg, "num_local_experts", 0)
        or getattr(cfg, "num_experts", 0)
        or getattr(cfg, "n_routed_experts", 0)
        or 0
    )
    is_moe = num_experts > 1 or any(k in arch_blob for k in ("moe", "mixtral", "qwen2_moe", "qwen3_moe", "deepseek"))

    if layer is not None:
        sub_names = [n.lower() for n, _ in layer.named_modules()]
        if any("post_feedforward_layernorm" in n or "post_attention_layernorm" in n for n in sub_names):
            if "gemma" in arch_blob or "olmo" in arch_blob:
                is_post_norm = True
        if any(".experts." in n or "block_sparse_moe" in n or "shared_expert" in n for n in sub_names):
            is_moe = True

    post_norm_gain = 1.0
    if is_post_norm:
        post_norm_gain = 1.12
        if pos is not None and neg is not None and not _is_meta(pos) and not _is_meta(neg) and pos.numel() > 0 and neg.numel() > 0:
            n_p = min(pos.shape[0], neg.shape[0])
            p_norm = float(pos[:n_p].float().norm(dim=-1).mean().item())
            n_norm = float(neg[:n_p].float().norm(dim=-1).mean().item())
            if n_norm > 1e-6:
                ratio = max(0.85, min(1.35, p_norm / n_norm))
                post_norm_gain = round(min(1.25, 1.0 + 0.15 * ratio), 4)

    return {
        "is_post_norm": is_post_norm,
        "is_moe": is_moe,
        "num_experts": num_experts,
        "post_norm_gain": post_norm_gain,
    }


def _filter_moe_and_norm_targets(targets: list[tuple], is_moe: bool) -> list[tuple]:
    """
    For MoE layers, preserve router/gate logits and prioritize shared experts + attention
    plus a capped subset of routed experts so expert routing stays balanced.
    """
    if not is_moe:
        return targets

    filtered = []
    routed_expert_ids: list[str] = []
    for name, weight, comp_type in targets:
        low = name.lower()
        if any(tok in low for tok in ("router", ".gate.", "gate_proj_router", "shared_expert_gate")):
            continue
        if ".experts." in low:
            match = re.search(r"\.experts\.(\d+)\.", low)
            if match:
                exp_id = match.group(1)
                if exp_id not in routed_expert_ids:
                    if len(routed_expert_ids) >= 4:
                        continue
                    routed_expert_ids.append(exp_id)
        filtered.append((name, weight, comp_type))
    return filtered or targets


def _calibrate_post_norm_residual(
    surgeon,
    layer_idx: int,
    positive_prompts: list[str],
    q_orth: torch.Tensor,
    weights: torch.Tensor,
    targets: list[tuple],
    calibrate_norms: bool,
) -> float:
    """
    Closed-loop post-norm calibration: measures residual refusal projection after
    post-norm re-scaling and applies a bounded micro-correction if residual signal remains.
    """
    if (
        not positive_prompts
        or getattr(surgeon, "extractor", None) is None
        or _has_meta_params(getattr(surgeon, "model", None))
        or q_orth is None
        or _is_meta(q_orth)
        or q_orth.numel() == 0
    ):
        return 0.0

    try:
        post_states = surgeon.extractor.extract_hidden_states(positive_prompts[: min(8, len(positive_prompts))], layer_idx)
        if _is_meta(post_states) or post_states.numel() == 0:
            return 0.0
        s_f = post_states.float()
        q_cpu = q_orth.detach().float().to(s_f.device)
        if q_cpu.ndim == 1:
            q_cpu = q_cpu.unsqueeze(0)
        proj_mag = float(((s_f @ q_cpu.T).norm(dim=-1) / s_f.norm(dim=-1).clamp_min(1e-6)).mean().item())
        if proj_mag < 0.08:
            return 0.0

        delta_alpha = min(0.22, 0.35 * proj_mag)
        with torch.no_grad():
            for _name, W, _comp_type in targets:
                if _is_meta(W) or W.ndim != 2:
                    continue
                w_dev = W.device
                W_f32 = W.float().to(w_dev)
                w_vec = (weights.float().to(w_dev).reshape(-1)[: q_cpu.shape[0]] * delta_alpha).unsqueeze(1)
                q_dev = q_cpu.to(w_dev)
                if W_f32.shape[0] == q_dev.shape[1]:
                    W_mod = W_f32 - (q_dev.T @ (w_vec * (q_dev @ W_f32)))
                elif W_f32.shape[1] == q_dev.shape[1]:
                    W_mod = W_f32 - ((W_f32 @ q_dev.T) @ (w_vec * q_dev))
                else:
                    continue
                if calibrate_norms:
                    W_mod = preserve_row_norms(W_f32, W_mod)
                W.copy_(W_mod.to(dtype=W.dtype, device=w_dev))
        return round(delta_alpha, 4)
    except Exception:
        return 0.0


def _fuse_dual_point_states(
    extractor,
    prompts: list[str],
    layer_idx: int,
    layer_dev: torch.device,
) -> torch.Tensor:
    """
    Extracts prompt-end states and fuses post-`<think>\\n` states when available
    so reasoning-model refusal boundaries are captured cleanly.
    """
    base_states = extractor.extract_hidden_states(prompts, layer_idx)
    if _is_meta(base_states) or base_states.numel() == 0:
        return base_states

    base_f = base_states.float().to(layer_dev)
    if not hasattr(extractor, "extract_dual_point_states"):
        return base_f

    try:
        _, think_states = extractor.extract_dual_point_states(prompts, layer_idx, think_suffix="<think>\n")
        if think_states is not None and not _is_meta(think_states) and think_states.shape == base_states.shape:
            t_f = think_states.float().to(layer_dev)
            return 0.65 * base_f + 0.35 * t_f
    except Exception:
        pass

    return base_f


def check_generation_coherence(text: str) -> dict:
    """
    Lightweight anti-glitch / anti-degeneration detector checking n-gram loop rate,
    character repetition, and non-printable/garbled token bursts.
    """
    clean = (text or "").strip()
    if not clean:
        return {"is_degenerate": True, "repetition_ratio": 1.0, "reason": "empty_output"}

    words = clean.lower().split()
    if len(words) >= 12:
        trigrams = [tuple(words[i : i + 3]) for i in range(len(words) - 2)]
        unique_ratio = len(set(trigrams)) / max(1, len(trigrams))
        if unique_ratio < 0.42:
            return {
                "is_degenerate": True,
                "repetition_ratio": round(1.0 - unique_ratio, 4),
                "reason": "trigram_loop",
            }

    if re.search(r"(.)\1{14,}", clean):
        return {"is_degenerate": True, "repetition_ratio": 0.95, "reason": "char_repeat_burst"}

    weird_chars = sum(1 for ch in clean if ch == "\ufffd" or (ord(ch) < 32 and ch not in "\n\r\t"))
    if weird_chars / max(1, len(clean)) > 0.05:
        return {"is_degenerate": True, "repetition_ratio": 0.85, "reason": "garbled_control_chars"}

    return {"is_degenerate": False, "repetition_ratio": 0.0, "reason": "ok"}


def _first_token_refusal_prob(model, tokenizer, inputs: dict, ref_t: torch.Tensor, cmp_t: torch.Tensor) -> float | None:
    if ref_t.numel() == 0 or cmp_t.numel() == 0:
        return None
    out = model(**inputs)
    logits = getattr(out, "logits", None)
    if logits is None or _is_meta(logits):
        return None
    log_probs = torch.log_softmax(logits[0, -1, :].float().cpu(), dim=-1)
    ref_lp = torch.logsumexp(log_probs[ref_t], dim=0).item()
    cmp_lp = torch.logsumexp(log_probs[cmp_t], dim=0).item()
    diff = max(-20.0, min(20.0, ref_lp - cmp_lp))
    return 1.0 / (1.0 + math.exp(-diff))


def _behavior_cost(trial: dict) -> float:
    return (
        float(trial["refusal_rate"])
        + 0.45 * float(trial.get("hedging_rate", 0.0))
        + 1.25 * float(trial.get("glitch_rate", 0.0))
    )


def _trial_objective_score(trial: dict, kl_weight: float = 0.25, max_kl: float = 1.10) -> float:
    mean_kl = float(trial["mean_kl"])
    if math.isnan(mean_kl) or math.isinf(mean_kl):
        mean_kl = max_kl * 2.0
    kl_penalty = max(0.0, mean_kl - max_kl) * 1.5
    return _behavior_cost(trial) + (kl_weight * mean_kl) + kl_penalty


def _compute_pareto_frontier(trials: list[dict], max_kl: float = 1.10) -> list[dict]:
    """Return non-dominated candidates minimizing (behavior_cost, mean_kl)."""
    valid = [t for t in trials if "params" in t]
    frontier: list[dict] = []
    for cand in valid:
        r_c = _behavior_cost(cand)
        k_c = float(cand["mean_kl"])
        dominated = False
        for other in valid:
            if other is cand:
                continue
            r_o = _behavior_cost(other)
            k_o = float(other["mean_kl"])
            if (r_o <= r_c and k_o <= k_c) and (r_o < r_c or k_o < k_c):
                dominated = True
                break
        if not dominated:
            entry = dict(cand["params"])
            entry["refusal_rate"] = cand["refusal_rate"]
            if "hedging_rate" in cand:
                entry["hedging_rate"] = cand["hedging_rate"]
            if "glitch_rate" in cand:
                entry["glitch_rate"] = cand["glitch_rate"]
            entry["mean_kl"] = k_c
            entry["objective_score"] = round(_trial_objective_score(cand, max_kl=max_kl), 6)
            frontier.append(entry)
    frontier.sort(key=lambda x: (x.get("objective_score", 0.0), _behavior_cost(x), x["mean_kl"]))
    return frontier


def _optimize_axis_weights_for_matrix(
    W0: torch.Tensor,
    Q: torch.Tensor,
    w_prior: torch.Tensor,
    steps: int,
    lr: float,
    kl_weight: float,
    max_axis_delta: float,
    calibrate_norms: bool,
    optimizer_kind: str = "adam",
) -> tuple[torch.Tensor | None, float]:
    """Optimize compact per-axis weights along Q for a 2D weight matrix matching hidden dim."""
    if W0 is None or Q is None or _is_meta(W0) or _is_meta(Q) or W0.ndim != 2 or Q.numel() == 0:
        return None, 0.0

    out_features, in_features = W0.shape
    q_2d = Q.float() if Q.ndim == 2 else Q.float().unsqueeze(0)
    dim = q_2d.shape[1]
    r_eff = q_2d.shape[0]
    w_dev = W0.device

    if in_features == dim:
        mode = "in"
        proj_base = W0 @ q_2d.T
    elif out_features == dim:
        mode = "out"
        proj_base = q_2d @ W0
    else:
        return None, 0.0

    prior_f = (
        w_prior.detach().float().to(w_dev)
        if (w_prior is not None and not _is_meta(w_prior) and w_prior.numel() == r_eff)
        else torch.ones((r_eff,), device=w_dev, dtype=torch.float32)
    )
    delta_vec = torch.zeros((r_eff,), device=w_dev, dtype=torch.float32, requires_grad=True)

    def _apply_delta(d_tensor: torch.Tensor) -> torch.Tensor:
        eff_w = d_tensor * prior_f
        if mode == "in":
            return W0 - (proj_base * eff_w.unsqueeze(0)) @ q_2d
        return W0 - q_2d.T @ (eff_w.unsqueeze(1) * proj_base)

    def _align_residual(cand_w: torch.Tensor) -> torch.Tensor:
        if mode == "in":
            return ((cand_w @ q_2d.T) * prior_f.unsqueeze(0)) ** 2
        return ((q_2d @ cand_w) * prior_f.unsqueeze(1)) ** 2

    safe_steps = max(1, int(steps))
    safe_lr = max(1e-4, float(lr))
    safe_kl_w = max(0.0, float(kl_weight))

    if optimizer_kind == "lbfgs":
        opt = torch.optim.LBFGS([delta_vec], lr=min(1.0, safe_lr * 2.0), max_iter=safe_steps, line_search_fn="strong_wolfe")

        def _closure():
            opt.zero_grad()
            cand_w = _apply_delta(delta_vec)
            loss = _align_residual(cand_w).mean() + safe_kl_w * (delta_vec ** 2).mean()
            loss.backward()
            return loss

        try:
            opt.step(_closure)
        except RuntimeError:
            adam = torch.optim.Adam([delta_vec], lr=safe_lr)
            for _ in range(safe_steps):
                adam.zero_grad()
                cand_w = _apply_delta(delta_vec)
                loss = _align_residual(cand_w).mean() + safe_kl_w * (delta_vec ** 2).mean()
                loss.backward()
                adam.step()
    else:
        opt = torch.optim.Adam([delta_vec], lr=safe_lr)
        for _ in range(safe_steps):
            opt.zero_grad()
            cand_w = _apply_delta(delta_vec)
            loss = _align_residual(cand_w).mean() + safe_kl_w * (delta_vec ** 2).mean()
            loss.backward()
            opt.step()

    with torch.no_grad():
        cap_delta = max(0.0, float(max_axis_delta))
        d_vals = delta_vec.detach().clamp(-cap_delta, cap_delta)
        if not torch.isfinite(d_vals).all() or float(d_vals.abs().max().item()) <= 1e-6:
            return None, 0.0
        W_mod = _apply_delta(d_vals)
        if not torch.isfinite(W_mod).all():
            return None, 0.0
        if calibrate_norms:
            W_mod = preserve_row_norms(W0, W_mod)
        before_norm = W0.norm().clamp_min(1e-8)
        rel_chg = float((W_mod - W0).norm().item() / before_norm.item())
        if math.isnan(rel_chg) or math.isinf(rel_chg):
            return None, 0.0
        return W_mod, rel_chg


class VeloraEngine:
    """
    VELORA v3.2: Variance-Energy Low-Rank Subspace Ablation with
    AGOP Causal Gradient Subspace, Multi-Cluster Cone Discovery,
    Covariance-Whitened Minimum-KL Projection, Cross-Layer Procrustes Coherence,
    Closed-Loop Post-Norm Calibration, MoE Routing Guards, Hybrid Micro-ARA
    Residual Fallback, Two-Stage Pareto Search, and Anti-Glitch Rollback.
    """

    @classmethod
    def assess_reaction(cls, response: str, mode: str = "velora_v3_2_strategic") -> tuple:
        assessment = classify_response(response)
        line = format_velora_telemetry(
            active=True,
            mode=assessment.recommended_mode if assessment.needs_escalation else mode,
            reaction=assessment.state,
        )
        return assessment, line

    @classmethod
    def polish_null_space_lbfgs(
        cls,
        surgeon,
        target_directions: list[torch.Tensor],
        capability_prompts: list[str],
        layer_indices: list[int],
        basis_rank: int = 12,
        step_alpha: float = 0.25,
        lbfgs_steps: int = 5,
        kl_weight: float = 0.35,
        calibrate_norms: bool = True,
        precomputed_cap: torch.Tensor | None = None,
        optimizer_kind: str = "adam",
    ) -> dict:
        if not target_directions or not layer_indices:
            return {"layers_polished": 0, "max_relative_change": 0.0, "lbfgs_steps": 0}

        layers = getattr(surgeon, "layers", None) or []
        n_layers = len(layers)
        valid_idx = [int(i) for i in layer_indices if 0 <= int(i) < n_layers]
        if not valid_idx:
            return {"layers_polished": 0, "max_relative_change": 0.0, "lbfgs_steps": 0}

        arch_traits = _detect_architecture_traits(getattr(surgeon, "model", None))
        max_rel_change = 0.0
        polished_layers = 0
        steps = max(1, min(int(lbfgs_steps), 12))
        lr = max(0.02, min(0.25, float(step_alpha) * 0.35))
        max_axis_delta = 0.42 if n_layers <= 28 else 0.55
        opt_kind = "lbfgs" if str(optimizer_kind).lower() == "lbfgs" else "adam"

        for idx in valid_idx:
            layer = layers[idx]
            params = list(layer.parameters()) if hasattr(layer, "parameters") else []
            if any(_is_meta(p) for p in params):
                continue
            layer_dev = params[0].device if params else getattr(surgeon, "device", torch.device("cpu"))

            cap = None
            if precomputed_cap is not None and not _is_meta(precomputed_cap) and precomputed_cap.numel() > 0:
                cap = precomputed_cap.to(layer_dev)
            elif capability_prompts and len(capability_prompts) >= 2 and getattr(surgeon, "extractor", None) is not None:
                raw_cap = surgeon.extractor.extract_hidden_states(capability_prompts, idx)
                if not _is_meta(raw_cap) and raw_cap.ndim >= 2 and raw_cap.shape[0] >= 2:
                    cap = raw_cap.to(layer_dev)

            cap_basis = (
                build_capability_basis(cap, rank=basis_rank).to(layer_dev)
                if (cap is not None and cap.ndim >= 2 and cap.shape[0] >= 2)
                else None
            )

            valid_dirs = [
                d.detach().float().to(layer_dev)
                for d in target_directions
                if d is not None and not _is_meta(d) and d.numel() > 0
            ]
            if not valid_dirs:
                continue

            stacked_dirs = torch.stack([v.reshape(-1) for v in valid_dirs], dim=0)
            orth_dirs = _project_orthogonal(stacked_dirs, cap_basis)
            q_dirs = _orthonormalize_rows(orth_dirs, stacked_dirs[0])
            r_eff = q_dirs.shape[0]

            prior_weights = torch.tensor(
                [0.85 ** i for i in range(r_eff)],
                device=layer_dev,
                dtype=torch.float32,
            )
            prior_weights = prior_weights / prior_weights.norm().clamp_min(1e-8)

            targets = _filter_moe_and_norm_targets(
                surgeon._get_component_targets(layer, ["attn", "mlp"]),
                is_moe=arch_traits["is_moe"],
            )
            layer_touched = False

            for _name, W, _comp_type in targets:
                if _is_meta(W) or W.ndim != 2:
                    continue
                w_dev = W.device
                W0 = W.detach().float().to(w_dev)
                W_mod, rel_chg = _optimize_axis_weights_for_matrix(
                    W0=W0,
                    Q=q_dirs.to(w_dev),
                    w_prior=prior_weights.to(w_dev),
                    steps=steps,
                    lr=lr,
                    kl_weight=kl_weight,
                    max_axis_delta=max_axis_delta,
                    calibrate_norms=calibrate_norms,
                    optimizer_kind=opt_kind,
                )
                if W_mod is not None:
                    max_rel_change = max(max_rel_change, rel_chg)
                    with torch.no_grad():
                        W.copy_(W_mod.to(device=w_dev, dtype=W.dtype))
                    layer_touched = True

            if layer_touched:
                polished_layers += 1

        return {
            "layers_polished": polished_layers,
            "max_relative_change": round(max_rel_change, 6),
            "lbfgs_steps": steps,
            "optimizer": opt_kind,
        }

    @staticmethod
    def extract_velora_subspace(
        pos_activations: torch.Tensor,
        neg_activations: torch.Tensor,
        capability_basis: torch.Tensor | None = None,
        anchor_direction: torch.Tensor | None = None,
        extra_directions: list[torch.Tensor] | None = None,
        max_rank: int = 12,
        target_energy: float = 0.94,
        spectral_gamma: float = 0.45,
        use_cone_clusters: bool = True,
        cone_clusters: int = 4,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute orthonormal capability-orthogonal refusal/anti-hedge cone subspace
        Q_perp [r, hidden] and spectral weights w [r] in (0, 1].
        """
        if pos_activations is None or neg_activations is None:
            raise ValueError("VELORA requires contrastive activation tensors.")

        n_pairs = min(pos_activations.shape[0], neg_activations.shape[0])
        if n_pairs == 0 or pos_activations.numel() == 0 or neg_activations.numel() == 0:
            raise ValueError("VELORA requires at least one non-empty contrastive activation pair.")

        pos_sub = pos_activations[:n_pairs].float()
        neg_sub = neg_activations[:n_pairs].float()

        if _is_meta(pos_sub) or _is_meta(neg_sub):
            hidden = pos_sub.shape[-1] if pos_sub.ndim >= 1 else 1
            return torch.eye(1, hidden, dtype=torch.float32), torch.ones(1, dtype=torch.float32)

        diff = _winsorize(pos_sub - neg_sub)
        if _is_meta(diff):
            hidden = pos_sub.shape[-1] if pos_sub.ndim >= 1 else 1
            return torch.eye(1, hidden, dtype=torch.float32), torch.ones(1, dtype=torch.float32)

        mean_diff = diff.mean(dim=0)
        mean_norm = float(mean_diff.norm().item())
        if mean_norm <= 1e-12 or math.isnan(mean_norm) or math.isinf(mean_norm):
            fallback_unit = torch.zeros_like(mean_diff)
            fallback_unit[0] = 1.0
        else:
            fallback_unit = mean_diff / mean_norm

        if n_pairs == 1:
            vh = fallback_unit.unsqueeze(0)
            svals = torch.ones(1, device=diff.device, dtype=torch.float32)
        else:
            centered = diff - mean_diff.unsqueeze(0)
            try:
                _, svals, vh = torch.linalg.svd(centered, full_matrices=False)
            except RuntimeError:
                _, svals, vh = torch.linalg.svd(centered.cpu(), full_matrices=False)
                svals, vh = svals.to(diff.device), vh.to(diff.device)

            if _is_meta(svals) or _is_meta(vh):
                return (
                    _orthonormalize_rows(torch.empty(0), fallback_unit),
                    torch.ones(1, device=diff.device, dtype=torch.float32),
                )

            if not _is_meta(vh[0]) and not _is_meta(mean_diff):
                dot_val = _safe_float_item(torch.dot(vh[0], mean_diff), default=1.0)
                if dot_val < 0.0:
                    vh = vh.clone()
                    vh[0] = -vh[0]

        energies = svals ** 2
        total_energy = energies.sum().clamp_min(1e-12)
        cum_energy = torch.cumsum(energies, dim=0) / total_energy

        upper = max(1, min(int(max_rank), vh.shape[0]))
        rank = upper
        for i in range(upper):
            val = _safe_float_item(cum_energy[i], default=1.0)
            if val >= float(target_energy):
                rank = i + 1
                break

        basis_vecs = vh[:rank].float()
        s_kept = svals[:rank].float().clamp_min(1e-8)
        s0 = max(1e-8, _safe_float_item(s_kept[0], default=1.0))
        base_weights = (s_kept / s0) ** float(spectral_gamma)

        aug_vecs = [basis_vecs]
        aug_weights = [base_weights]

        if use_cone_clusters and n_pairs >= 4:
            cone_vecs, cone_w = _extract_cone_directions(diff, n_clusters=cone_clusters)
            if cone_vecs.numel() > 0:
                aug_vecs.append(cone_vecs.to(basis_vecs.device))
                aug_weights.append((cone_w.to(basis_vecs.device) ** float(spectral_gamma)) * 0.90)

        if anchor_direction is not None and anchor_direction.numel() > 0 and not _is_meta(anchor_direction):
            anc = anchor_direction.to(device=basis_vecs.device, dtype=torch.float32).reshape(-1)
            if anc.shape[0] == basis_vecs.shape[-1]:
                anc_unit = anc / anc.norm().clamp_min(1e-8)
                aug_vecs.append(anc_unit.unsqueeze(0))
                aug_weights.append(
                    torch.tensor([0.92 ** float(spectral_gamma)], device=basis_vecs.device, dtype=torch.float32)
                )

        if extra_directions:
            for idx_e, ed in enumerate(extra_directions):
                if ed is None or ed.numel() == 0 or _is_meta(ed):
                    continue
                ed_f = ed.to(device=basis_vecs.device, dtype=torch.float32).reshape(-1)
                if ed_f.shape[0] != basis_vecs.shape[-1]:
                    continue
                ed_u = ed_f / ed_f.norm().clamp_min(1e-8)
                w_e = max(0.35, 0.85 ** (idx_e + 1)) ** float(spectral_gamma)
                aug_vecs.append(ed_u.unsqueeze(0))
                aug_weights.append(torch.tensor([w_e], device=basis_vecs.device, dtype=torch.float32))

        combined_vecs = torch.cat(aug_vecs, dim=0)
        combined_weights = torch.cat(aug_weights, dim=0)

        perp = _project_orthogonal(combined_vecs, capability_basis)
        q_orth = _orthonormalize_rows(perp, fallback_unit)
        r_eff = q_orth.shape[0]

        align_scores = (q_orth @ combined_vecs.T).abs().max(dim=1).values
        order = torch.argsort(align_scores, descending=True)
        q_sorted = q_orth[order]

        w_pool = combined_weights[:r_eff].contiguous()
        if w_pool.shape[0] < r_eff:
            pad = torch.full((r_eff - w_pool.shape[0],), 0.5, device=q_sorted.device, dtype=torch.float32)
            w_pool = torch.cat([w_pool, pad], dim=0)

        return q_sorted.contiguous(), w_pool[:r_eff].contiguous()

    @staticmethod
    def apply_velora_projection(
        W: torch.Tensor,
        q_orth: torch.Tensor,
        weights: torch.Tensor,
        alpha: float,
        orba_blend: float = 0.0,
        cap_activations: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            W is None
            or q_orth is None
            or weights is None
            or _is_meta(W)
            or _is_meta(q_orth)
            or _is_meta(weights)
            or W.ndim != 2
            or q_orth.numel() == 0
        ):
            return W

        dev = W.device
        W_f32 = W.float().to(dev)
        Q = q_orth.float().to(dev)
        if Q.ndim == 1:
            Q = Q.unsqueeze(0)
        w = (weights.float().to(dev).reshape(-1)[: Q.shape[0]] * float(alpha)).clone()

        beta_orba = max(0.0, min(0.50, float(orba_blend)))
        if beta_orba > 0.0:
            w[0] = min(2.0, float(w[0].item()) + beta_orba)

        out_features, in_features = W_f32.shape
        dim = Q.shape[1]

        if cap_activations is not None and not _is_meta(cap_activations) and cap_activations.ndim >= 2 and cap_activations.shape[0] >= 2:
            c_f = cap_activations.float().to(dev)
            if c_f.shape[-1] == dim:
                c_centered = c_f - c_f.mean(dim=0, keepdim=True)
                proj_var = ((c_centered @ Q.T) ** 2).mean(dim=0)
                var_med = proj_var.median().clamp_min(1e-6)
                cov_damp = (var_med / (var_med + 0.35 * proj_var)).clamp(0.45, 1.0)
                w = w * cov_damp

        w_col = w.unsqueeze(1)

        if out_features == dim:
            qw = Q @ W_f32
            return W_f32 - (Q.T @ (w_col * qw))

        if in_features == dim:
            wq = W_f32 @ Q.T
            return W_f32 - (wq @ (w_col * Q))

        return W_f32

    @classmethod
    def _apply_band(
        cls,
        surgeon,
        positive_prompts: list[str],
        negative_prompts: list[str],
        capability_prompts: list[str],
        center: float,
        spread: float,
        alpha_attn: float,
        alpha_mlp: float,
        basis_rank_attn: int,
        basis_rank_mlp: int,
        max_subspace_rank: int,
        target_energy: float,
        spectral_gamma: float,
        calibrate_norms: bool,
        use_prefix_contrast: bool,
        already_modified: set[int],
        tag: str = "VELORA",
        include_hedging_contrast: bool = False,
        extraction_cache: dict | None = None,
        max_contrast_prompts: int = 24,
        max_extra_dirs: int = 6,
        polish_after_band: bool = True,
        orba_blend: float = 0.0,
        apply_affine_bias: bool = True,
        over_refusal_prompts: list[str] | None = None,
        use_agop: bool = True,
    ) -> tuple[int, int, dict]:
        layers = getattr(surgeon, "layers", None) or []
        n_layers = len(layers)
        if n_layers == 0:
            return 0, 0, {}

        radius = max(1, int(round(spread * 1.4)))
        lo = max(0, int(center) - radius)
        hi = min(n_layers - 1, int(center) + radius)
        target_indices = [i for i in range(lo, hi + 1) if i not in already_modified]
        if not target_indices:
            return lo, hi, {}

        base_traits = _detect_architecture_traits(getattr(surgeon, "model", None), layers[target_indices[0]])
        is_moe = base_traits["is_moe"]

        layer_weights = compute_kernel_layer_weights(
            target_indices,
            peak_position=center,
            max_weight=1.0,
            min_weight=0.20,
            spread=spread,
        )

        primary_dirs = {}
        layer_cache = {}
        skip_prefix = _has_meta_params(getattr(surgeon, "model", None))
        cache_store = extraction_cache if extraction_cache is not None else {}
        ref_cap = max(1, int(max_contrast_prompts))
        hedge_cap = max(1, min(ref_cap, int(max_contrast_prompts * 0.67)))
        defl_cap = max(1, min(ref_cap, int(max_contrast_prompts * 0.50)))
        resp_prefix = getattr(getattr(surgeon, "extractor", None), "response_prefix", "")

        for idx in target_indices:
            layer_params = list(layers[idx].parameters()) if hasattr(layers[idx], "parameters") else []
            layer_dev = layer_params[0].device if layer_params else getattr(surgeon, "device", torch.device("cpu"))
            if getattr(surgeon, "extractor", None) is not None:
                surgeon.extractor.device = layer_dev

            cache_key = (idx, bool(use_prefix_contrast), bool(include_hedging_contrast), bool(use_agop))

            if cache_key in cache_store:
                c_pos, c_neg, c_prefix, c_cap, c_or = cache_store[cache_key]
                pos = c_pos.to(layer_dev) if not _is_meta(c_pos) else c_pos
                neg = c_neg.to(layer_dev) if not _is_meta(c_neg) else c_neg
                prefix_dirs = [d.to(layer_dev) if not _is_meta(d) else d for d in c_prefix]
                cap = c_cap.to(layer_dev) if (c_cap is not None and not _is_meta(c_cap)) else None
                or_states = c_or.to(layer_dev) if (c_or is not None and not _is_meta(c_or)) else None
            else:
                pos = _fuse_dual_point_states(surgeon.extractor, positive_prompts, idx, layer_dev)
                neg = _fuse_dual_point_states(surgeon.extractor, negative_prompts, idx, layer_dev)
                if not _is_meta(pos) and pos.numel() > 0:
                    pos = pos.to(layer_dev)
                if not _is_meta(neg) and neg.numel() > 0:
                    neg = neg.to(layer_dev)

                prefix_dirs = []
                if use_prefix_contrast and positive_prompts and not skip_prefix:
                    try:
                        if hasattr(RefusalSubspaceExtractor, "extract_prefix_contrast_directions"):
                            ref_dirs = RefusalSubspaceExtractor.extract_prefix_contrast_directions(
                                model=surgeon.model,
                                tokenizer=surgeon.tokenizer,
                                device=layer_dev,
                                prompts=positive_prompts[:ref_cap],
                                layer_idx=idx,
                                rank=min(4, max_subspace_rank),
                            )
                            if ref_dirs:
                                prefix_dirs.extend(list(ref_dirs))
                        if include_hedging_contrast and hasattr(RefusalSubspaceExtractor, "extract_hedging_contrast_directions"):
                            hedge_dirs = RefusalSubspaceExtractor.extract_hedging_contrast_directions(
                                model=surgeon.model,
                                tokenizer=surgeon.tokenizer,
                                device=layer_dev,
                                prompts=positive_prompts[:hedge_cap],
                                layer_idx=idx,
                                rank=min(3, max_subspace_rank),
                            )
                            if hedge_dirs:
                                prefix_dirs.extend(list(hedge_dirs))
                        if include_hedging_contrast and hasattr(RefusalSubspaceExtractor, "extract_deflection_contrast_directions"):
                            defl_dirs = RefusalSubspaceExtractor.extract_deflection_contrast_directions(
                                model=surgeon.model,
                                tokenizer=surgeon.tokenizer,
                                device=layer_dev,
                                prompts=positive_prompts[:defl_cap],
                                layer_idx=idx,
                                rank=min(2, max_subspace_rank),
                            )
                            if defl_dirs:
                                prefix_dirs.extend(list(defl_dirs))
                    except (NotImplementedError, RuntimeError) as e:
                        if "meta" in str(e).lower() or isinstance(e, NotImplementedError):
                            logger.warning(
                                f"Skipping prefix-contrast extraction at layer {idx} due to meta/offloaded tensor: {e}"
                            )
                            skip_prefix = True
                            prefix_dirs = []
                        else:
                            raise

                if use_agop and positive_prompts and not skip_prefix:
                    agop_vecs = extract_agop_subspace(
                        model=surgeon.model,
                        tokenizer=surgeon.tokenizer,
                        prompts=positive_prompts,
                        layer_idx=idx,
                        device=layer_dev,
                        rank=min(4, max_subspace_rank),
                        response_prefix=resp_prefix,
                        max_prompts=min(12, ref_cap),
                    )
                    if agop_vecs:
                        prefix_dirs.extend(agop_vecs)

                cap = None
                if capability_prompts and len(capability_prompts) >= 2:
                    raw_cap = surgeon.extractor.extract_hidden_states(capability_prompts, idx)
                    if not _is_meta(raw_cap) and raw_cap.ndim >= 2 and raw_cap.shape[0] >= 2:
                        cap = raw_cap.to(layer_dev)

                or_states = None
                if over_refusal_prompts and len(over_refusal_prompts) >= 2:
                    raw_or = surgeon.extractor.extract_hidden_states(over_refusal_prompts, idx)
                    if not _is_meta(raw_or) and raw_or.ndim >= 2 and raw_or.shape[0] >= 2:
                        or_states = raw_or.to(layer_dev)

                cache_store[cache_key] = (
                    pos.detach().cpu() if not _is_meta(pos) else pos,
                    neg.detach().cpu() if not _is_meta(neg) else neg,
                    [d.detach().cpu() if not _is_meta(d) else d for d in prefix_dirs],
                    cap.detach().cpu() if (cap is not None and not _is_meta(cap)) else None,
                    or_states.detach().cpu() if (or_states is not None and not _is_meta(or_states)) else None,
                )

            layer_cache[idx] = (pos, neg, prefix_dirs, cap, or_states)
            if prefix_dirs:
                primary_dirs[idx] = prefix_dirs[0]
            elif not _is_meta(pos) and not _is_meta(neg) and pos.numel() > 0 and neg.numel() > 0:
                n_p = min(pos.shape[0], neg.shape[0])
                if n_p > 0:
                    d = (pos[:n_p] - neg[:n_p]).float().mean(dim=0)
                    primary_dirs[idx] = d / d.norm().clamp_min(1e-8)

        interp_peak_dir = interpolate_layer_direction(primary_dirs, center) if primary_dirs else None
        extra_cap = max(1, int(max_extra_dirs))

        extracted_subspaces: dict[int, tuple] = {}
        for idx in target_indices:
            layer = layers[idx]
            layer_params = list(layer.parameters()) if hasattr(layer, "parameters") else []
            layer_dev = layer_params[0].device if layer_params else getattr(surgeon, "device", torch.device("cpu"))
            pos, neg, prefix_dirs, cap, or_states = layer_cache[idx]
            if _is_meta(pos) or _is_meta(neg) or pos.numel() == 0 or neg.numel() == 0:
                continue

            has_cap = cap is not None and not _is_meta(cap) and cap.ndim >= 2 and cap.shape[0] >= 2
            cap_attn = build_capability_basis(cap, rank=basis_rank_attn).to(layer_dev) if has_cap else None
            if not has_cap:
                cap_mlp = None
            elif basis_rank_mlp == basis_rank_attn and cap_attn is not None:
                cap_mlp = cap_attn
            else:
                cap_mlp = build_capability_basis(cap, rank=basis_rank_mlp).to(layer_dev)

            extra_cues = [d.to(layer_dev) for d in prefix_dirs[:extra_cap] if d is not None and not _is_meta(d)]
            if or_states is not None and has_cap:
                n_or = min(or_states.shape[0], cap.shape[0])
                if n_or >= 2:
                    or_diff = (or_states[:n_or] - cap[:n_or]).float().mean(dim=0)
                    if float(or_diff.norm().item()) > 1e-8:
                        extra_cues.append(or_diff / or_diff.norm().clamp_min(1e-8))

            anchor = (
                interp_peak_dir.to(layer_dev)
                if (prefix_dirs and interp_peak_dir is not None and not _is_meta(interp_peak_dir))
                else None
            )

            q_attn, w_attn = cls.extract_velora_subspace(
                pos,
                neg,
                cap_attn,
                anchor_direction=anchor,
                extra_directions=extra_cues or None,
                max_rank=max_subspace_rank,
                target_energy=target_energy,
                spectral_gamma=spectral_gamma,
            )
            q_mlp, w_mlp = cls.extract_velora_subspace(
                pos,
                neg,
                cap_mlp,
                anchor_direction=anchor,
                extra_directions=extra_cues or None,
                max_rank=max_subspace_rank,
                target_energy=target_energy,
                spectral_gamma=spectral_gamma,
            )
            extracted_subspaces[idx] = (q_attn, w_attn, q_mlp, w_mlp, cap_attn)

        coherence_weights = _compute_cross_layer_coherence(
            {idx: vals[0] for idx, vals in extracted_subspaces.items()}
        )

        layer_ranks = {}
        polished_dirs_by_layer: dict[int, tuple[list[torch.Tensor], torch.Tensor | None]] = {}

        for idx, (q_attn, w_attn, q_mlp, w_mlp, cap_attn) in extracted_subspaces.items():
            layer = layers[idx]
            targets = _filter_moe_and_norm_targets(
                surgeon._get_component_targets(layer, ["attn", "mlp"]),
                is_moe=is_moe,
            )
            if not targets:
                continue

            pos, neg, prefix_dirs, cap, _or_states = layer_cache[idx]
            traits = _detect_architecture_traits(getattr(surgeon, "model", None), layer, pos=pos, neg=neg)
            gain_mult = traits["post_norm_gain"]
            c_score = conflict_score(q_attn[0], cap_attn) if cap_attn is not None else 0.0
            snr_gate = _compute_separability_gate(pos, neg)
            coh_mult = coherence_weights.get(idx, 1.0)
            surgeon.conflict_table[f"{tag}_layer_{idx}"] = c_score

            k_mult = layer_weights.get(idx, 1.0) * snr_gate * coh_mult * gain_mult
            eff_alpha_attn = conflict_scaled_alpha(alpha_attn * k_mult, c_score)
            eff_alpha_mlp = conflict_scaled_alpha(alpha_mlp * k_mult, c_score)

            with torch.no_grad():
                for _name, W, comp_type in targets:
                    if _is_meta(W) or W.ndim != 2:
                        continue
                    w_dev = W.device
                    W_f32 = W.float().to(w_dev)
                    cap_dev = cap.to(w_dev) if (cap is not None and not _is_meta(cap)) else None
                    if comp_type == "attn":
                        W_mod = cls.apply_velora_projection(
                            W_f32,
                            q_attn.to(w_dev),
                            w_attn.to(w_dev),
                            eff_alpha_attn,
                            orba_blend=orba_blend,
                            cap_activations=cap_dev,
                        )
                    else:
                        W_mod = cls.apply_velora_projection(
                            W_f32,
                            q_mlp.to(w_dev),
                            w_mlp.to(w_dev),
                            eff_alpha_mlp,
                            orba_blend=orba_blend,
                            cap_activations=cap_dev,
                        )

                    if calibrate_norms:
                        W_mod = preserve_row_norms(W_f32, W_mod)
                    W.copy_(W_mod.to(dtype=W.dtype, device=w_dev))

                if apply_affine_bias and not _is_meta(pos) and not _is_meta(neg) and pos.numel() > 0 and neg.numel() > 0:
                    n_p = min(pos.shape[0], neg.shape[0])
                    if n_p > 0:
                        mean_shift = (pos[:n_p] - neg[:n_p]).float().mean(dim=0)
                        q_cpu = q_attn.float().to(mean_shift.device)
                        proj_shift = (mean_shift @ q_cpu.T) @ q_cpu
                        for mod in layer.modules():
                            bias = getattr(mod, "bias", None)
                            if bias is not None and not _is_meta(bias) and bias.ndim == 1 and bias.shape[0] == proj_shift.shape[0]:
                                b_dev = bias.device
                                b_mod = bias.float() - (0.35 * eff_alpha_mlp) * proj_shift.to(b_dev)
                                bias.copy_(b_mod.to(dtype=bias.dtype, device=b_dev))

            cal_delta = 0.0
            if traits["is_post_norm"]:
                cal_delta = _calibrate_post_norm_residual(
                    surgeon=surgeon,
                    layer_idx=idx,
                    positive_prompts=positive_prompts,
                    q_orth=q_attn,
                    weights=w_attn,
                    targets=targets,
                    calibrate_norms=calibrate_norms,
                )

            layer_ranks[idx] = {
                "attn_rank": int(q_attn.shape[0]),
                "mlp_rank": int(q_mlp.shape[0]),
                "snr_gate": snr_gate,
                "coherence": coh_mult,
                "post_norm_gain": gain_mult,
                "post_norm_cal_delta": cal_delta,
            }

            if polish_after_band and prefix_dirs and not skip_prefix:
                polished_dirs_by_layer[idx] = (prefix_dirs, cap)

            already_modified.add(idx)

        if polish_after_band and polished_dirs_by_layer and not skip_prefix:
            for p_idx, (p_dirs, p_cap) in polished_dirs_by_layer.items():
                cls.polish_null_space_lbfgs(
                    surgeon=surgeon,
                    target_directions=p_dirs,
                    capability_prompts=capability_prompts,
                    layer_indices=[p_idx],
                    basis_rank=basis_rank_attn,
                    step_alpha=min(0.25, alpha_attn * 0.14),
                    lbfgs_steps=4,
                    calibrate_norms=calibrate_norms,
                    precomputed_cap=p_cap,
                )

        return lo, hi, layer_ranks

    @classmethod
    def run_velora_surgery(
        cls,
        surgeon,
        positive_prompts: list[str],
        negative_prompts: list[str],
        capability_prompts: list[str],
        peak_position: float | None = None,
        spread: float = 2.5,
        alpha_attn: float = 1.25,
        alpha_mlp: float = 1.05,
        basis_rank_attn: int = 12,
        basis_rank_mlp: int = 10,
        max_subspace_rank: int = 12,
        target_energy: float = 0.94,
        spectral_gamma: float = 0.45,
        calibrate_norms: bool = True,
        use_prefix_contrast: bool = True,
        anti_hedge: bool = True,
        dual_band: bool = True,
        late_alpha_scale: float = 0.60,
        extraction_cache: dict | None = None,
        polish_after_band: bool = True,
        orba_blend: float = 0.0,
        coherence_guard: bool = True,
        over_refusal_prompts: list[str] | None = None,
        use_agop: bool = True,
    ) -> dict:
        n_layers = len(getattr(surgeon, "layers", None) or [])
        if n_layers == 0:
            return {
                "modifier": "velora",
                "layers_modified": 0,
                "peak_position": 0.0,
                "layer_range": [0, 0],
                "layer_ranks": {},
            }

        resolved_peak = (
            float(layer_index_from_ratio(n_layers, 0.62))
            if peak_position is None
            else float(peak_position)
        )
        center = max(0.0, min(float(n_layers - 1), resolved_peak))

        is_compact = n_layers <= 28
        eff_max_rank = max(int(max_subspace_rank), 16 if n_layers >= 32 else (12 if not is_compact else 8))
        eff_spread = max(float(spread), 2.5 if is_compact else 3.2)
        eff_alpha_attn = float(alpha_attn) * (0.85 if is_compact else 1.0)
        eff_alpha_mlp = float(alpha_mlp) * (0.85 if is_compact else 1.0)

        snap = None
        all_indices = list(range(n_layers))
        if coherence_guard and capability_prompts and getattr(surgeon, "tokenizer", None) is not None and not _has_meta_params(getattr(surgeon, "model", None)):
            snap = LayerWeightSnapshot()
            snap.capture(surgeon.model, all_indices, store_on_cpu=True)

        modified: set[int] = set()
        cache_store = extraction_cache if extraction_cache is not None else {}

        lo, hi, ranks = cls._apply_band(
            surgeon=surgeon,
            positive_prompts=positive_prompts,
            negative_prompts=negative_prompts,
            capability_prompts=capability_prompts,
            center=center,
            spread=eff_spread,
            alpha_attn=eff_alpha_attn,
            alpha_mlp=eff_alpha_mlp,
            basis_rank_attn=basis_rank_attn,
            basis_rank_mlp=basis_rank_mlp,
            max_subspace_rank=eff_max_rank,
            target_energy=target_energy,
            spectral_gamma=spectral_gamma,
            calibrate_norms=calibrate_norms,
            use_prefix_contrast=use_prefix_contrast,
            already_modified=modified,
            tag="VELORA_Primary",
            include_hedging_contrast=anti_hedge,
            extraction_cache=cache_store,
            polish_after_band=polish_after_band,
            orba_blend=orba_blend,
            over_refusal_prompts=over_refusal_prompts,
            use_agop=use_agop,
        )

        if dual_band and n_layers >= 12:
            late_target = float(layer_index_from_ratio(n_layers, 0.78))
            min_late = min(float(n_layers - 2), center + max(3.0, n_layers * 0.12))
            late_center = min(float(n_layers - 2), max(min_late, late_target))
            _, _, late_ranks = cls._apply_band(
                surgeon=surgeon,
                positive_prompts=positive_prompts,
                negative_prompts=negative_prompts,
                capability_prompts=capability_prompts,
                center=late_center,
                spread=max(1.5, eff_spread * 0.65),
                alpha_attn=eff_alpha_attn * float(late_alpha_scale),
                alpha_mlp=eff_alpha_mlp * float(late_alpha_scale),
                basis_rank_attn=basis_rank_attn,
                basis_rank_mlp=basis_rank_mlp,
                max_subspace_rank=eff_max_rank,
                target_energy=target_energy,
                spectral_gamma=spectral_gamma,
                calibrate_norms=calibrate_norms,
                use_prefix_contrast=use_prefix_contrast,
                already_modified=modified,
                tag="VELORA_Late",
                include_hedging_contrast=anti_hedge,
                extraction_cache=cache_store,
                polish_after_band=polish_after_band,
                orba_blend=orba_blend,
                over_refusal_prompts=over_refusal_prompts,
                use_agop=use_agop,
            )
            ranks.update(late_ranks)

        glitch_detected = False
        if snap is not None and capability_prompts:
            try:
                resp_prefix = getattr(getattr(surgeon, "extractor", None), "response_prefix", "")
                probe_prompt = capability_prompts[0]
                inp = _move_batch_to_device(
                    surgeon.tokenizer(
                        format_chat_prompt(surgeon.tokenizer, probe_prompt, response_prefix=resp_prefix),
                        return_tensors="pt",
                    ),
                    surgeon.device,
                )
                in_len = inp["input_ids"].shape[1]
                surgeon.model.eval()
                with torch.no_grad():
                    gen = surgeon.model.generate(**inp, max_new_tokens=96, do_sample=False)
                sample_out = _decode_generated_tokens(surgeon.tokenizer, gen[0][in_len:])
                coh_check = check_generation_coherence(sample_out)
                if coh_check["is_degenerate"]:
                    glitch_detected = True
                    logger.warning(
                        f"VELORA coherence guard triggered ({coh_check['reason']}); rolling back and re-running at 0.65x alpha."
                    )
                    snap.restore(surgeon.model, all_indices)
                    return cls.run_velora_surgery(
                        surgeon=surgeon,
                        positive_prompts=positive_prompts,
                        negative_prompts=negative_prompts,
                        capability_prompts=capability_prompts,
                        peak_position=peak_position,
                        spread=spread,
                        alpha_attn=float(alpha_attn) * 0.65,
                        alpha_mlp=float(alpha_mlp) * 0.65,
                        basis_rank_attn=basis_rank_attn,
                        basis_rank_mlp=basis_rank_mlp,
                        max_subspace_rank=max(4, int(max_subspace_rank) // 2),
                        target_energy=target_energy,
                        spectral_gamma=spectral_gamma,
                        calibrate_norms=calibrate_norms,
                        use_prefix_contrast=use_prefix_contrast,
                        anti_hedge=anti_hedge,
                        dual_band=dual_band,
                        late_alpha_scale=late_alpha_scale,
                        extraction_cache=cache_store,
                        polish_after_band=False,
                        orba_blend=0.0,
                        coherence_guard=False,
                        over_refusal_prompts=over_refusal_prompts,
                        use_agop=use_agop,
                    )
            except Exception:
                pass

        logger.info(
            format_velora_telemetry(
                active=True,
                mode="velora_v3_2_strategic",
                reaction="SURGERY_APPLIED",
                layers=f"{min(modified) if modified else lo}..{max(modified) if modified else hi}",
                extra=f"modified={len(modified)}",
            )
        )

        return {
            "modifier": "velora",
            "layers_modified": len(modified),
            "peak_position": center,
            "layer_range": [lo, hi],
            "layer_ranks": ranks,
            "glitch_recovered": glitch_detected,
        }

    @classmethod
    def hybrid_velora_ara(
        cls,
        surgeon,
        positive_prompts: list[str],
        negative_prompts: list[str],
        capability_prompts: list[str],
        refusal_threshold: float = 0.15,
        micro_ara_rank: int = 4,
        micro_ara_steps: int = 12,
        calibrate_norms: bool = True,
        over_refusal_prompts: list[str] | None = None,
    ) -> dict:
        """
        Hybrid VELORA + Micro-ARA residual fallback: runs fast VELORA surgery first,
        measures remaining refusal probability on probe prompts, and fits a compact
        rank-r residual adapter (B @ A) only if residual refusal > refusal_threshold.
        """
        base_res = cls.run_velora_surgery(
            surgeon=surgeon,
            positive_prompts=positive_prompts,
            negative_prompts=negative_prompts,
            capability_prompts=capability_prompts,
            over_refusal_prompts=over_refusal_prompts,
            use_agop=True,
        )

        layers = getattr(surgeon, "layers", None) or []
        n_layers = len(layers)
        if n_layers == 0 or not positive_prompts or getattr(surgeon, "tokenizer", None) is None or _has_meta_params(getattr(surgeon, "model", None)):
            base_res["hybrid_micro_ara_triggered"] = False
            return base_res

        ref_ids = _first_token_ids(surgeon.tokenizer, REFUSAL_START_TOKENS)
        cmp_ids = _first_token_ids(surgeon.tokenizer, COMPLIANCE_START_TOKENS)
        if not ref_ids or not cmp_ids:
            base_res["hybrid_micro_ara_triggered"] = False
            return base_res

        ref_t = torch.tensor(ref_ids, dtype=torch.long)
        cmp_t = torch.tensor(cmp_ids, dtype=torch.long)
        resp_prefix = getattr(getattr(surgeon, "extractor", None), "response_prefix", "")
        probes = positive_prompts[: min(6, len(positive_prompts))]

        probs: list[float] = []
        surgeon.model.eval()
        with torch.no_grad():
            for p in probes:
                inp = _move_batch_to_device(
                    surgeon.tokenizer(format_chat_prompt(surgeon.tokenizer, p, response_prefix=resp_prefix), return_tensors="pt"),
                    surgeon.device,
                )
                lp_p = _first_token_refusal_prob(surgeon.model, surgeon.tokenizer, inp, ref_t, cmp_t)
                if lp_p is not None:
                    probs.append(lp_p)

        mean_prob = (sum(probs) / len(probs)) if probs else 0.0
        base_res["post_velora_refusal_prob"] = round(mean_prob, 4)

        if mean_prob <= float(refusal_threshold):
            base_res["hybrid_micro_ara_triggered"] = False
            return base_res

        # Micro-ARA residual polish on top 2 mid-late layers
        peak_idx = int(round(base_res.get("peak_position", layer_index_from_ratio(n_layers, 0.62))))
        late_idx = min(n_layers - 1, max(peak_idx + 1, int(layer_index_from_ratio(n_layers, 0.78))))
        stubborn_layers = sorted({max(0, min(n_layers - 1, peak_idx)), late_idx})
        r_micro = max(1, min(int(micro_ara_rank), 8))
        steps_micro = max(2, min(int(micro_ara_steps), 24))

        # Extract refusal direction at stubborn layers for target alignment
        fitted_modules = 0
        for idx in stubborn_layers:
            layer = layers[idx]
            params = list(layer.parameters()) if hasattr(layer, "parameters") else []
            if any(_is_meta(p) for p in params):
                continue
            layer_dev = params[0].device if params else getattr(surgeon, "device", torch.device("cpu"))
            agop_d = extract_agop_subspace(
                model=surgeon.model,
                tokenizer=surgeon.tokenizer,
                prompts=probes,
                layer_idx=idx,
                device=layer_dev,
                rank=r_micro,
                response_prefix=resp_prefix,
                max_prompts=len(probes),
            )
            if not agop_d:
                continue
            Q_sub = _orthonormalize_rows(torch.stack(agop_d, dim=0), agop_d[0]).to(layer_dev)
            prior_w = torch.linspace(1.0, 0.5, steps=Q_sub.shape[0], device=layer_dev, dtype=torch.float32)

            targets = _filter_moe_and_norm_targets(
                surgeon._get_component_targets(layer, ["attn", "mlp"]),
                is_moe=_detect_architecture_traits(getattr(surgeon, "model", None), layer)["is_moe"],
            )
            for _name, W, _comp_type in targets:
                if _is_meta(W) or W.ndim != 2:
                    continue
                w_dev = W.device
                W0 = W.detach().float().to(w_dev)
                W_mod, _ = _optimize_axis_weights_for_matrix(
                    W0=W0,
                    Q=Q_sub.to(w_dev),
                    w_prior=prior_w.to(w_dev),
                    steps=steps_micro,
                    lr=0.08,
                    kl_weight=0.30,
                    max_axis_delta=0.45,
                    calibrate_norms=calibrate_norms,
                    optimizer_kind="adam",
                )
                if W_mod is not None:
                    with torch.no_grad():
                        W.copy_(W_mod.to(device=w_dev, dtype=W.dtype))
                    fitted_modules += 1

        base_res["hybrid_micro_ara_triggered"] = True
        base_res["hybrid_micro_ara_layers"] = stubborn_layers
        base_res["hybrid_micro_ara_modules_fitted"] = fitted_modules
        return base_res

    @classmethod
    def optimize_velora_pareto(
        cls,
        surgeon,
        positive_prompts: list[str],
        negative_prompts: list[str],
        capability_prompts: list[str],
        n_trials: int = 12,
        max_kl: float = 1.10,
        seed: int = 42,
        extraction_cache: dict | None = None,
        max_eval_prompts: int = 16,
        max_cap_prompts: int = 4,
    ) -> dict:
        if optuna is None:
            raise ImportError("optuna is required for optimize_velora_pareto. Run `pip install optuna`.")

        n_layers = len(getattr(surgeon, "layers", None) or [])
        all_indices = list(range(n_layers))
        snap = LayerWeightSnapshot()
        snap.capture(surgeon.model, all_indices, store_on_cpu=True)
        shared_cache = extraction_cache if extraction_cache is not None else {}

        resp_prefix = getattr(getattr(surgeon, "extractor", None), "response_prefix", "")
        cap_subset = (capability_prompts or [])[: max(1, int(max_cap_prompts))]
        cap_inputs = [
            _move_batch_to_device(
                surgeon.tokenizer(
                    format_chat_prompt(surgeon.tokenizer, cp, response_prefix=resp_prefix),
                    return_tensors="pt",
                ),
                surgeon.device,
            )
            for cp in cap_subset
        ]
        eval_prompts = (positive_prompts or [])[::2][: max(1, int(max_eval_prompts))]
        eval_inputs = [
            _move_batch_to_device(
                surgeon.tokenizer(
                    format_chat_prompt(surgeon.tokenizer, p, response_prefix=resp_prefix),
                    return_tensors="pt",
                ),
                surgeon.device,
            )
            for p in eval_prompts
        ]

        ref_ids = _first_token_ids(surgeon.tokenizer, REFUSAL_START_TOKENS) if getattr(surgeon, "tokenizer", None) else []
        cmp_ids = _first_token_ids(surgeon.tokenizer, COMPLIANCE_START_TOKENS) if getattr(surgeon, "tokenizer", None) else []
        ref_t = torch.tensor(ref_ids, dtype=torch.long)
        cmp_t = torch.tensor(cmp_ids, dtype=torch.long)

        base_logits_list = []
        surgeon.model.eval()
        with torch.no_grad():
            for inp in cap_inputs:
                out = surgeon.model(**inp)
                logits = getattr(out, "logits", None)
                if logits is not None and not _is_meta(logits):
                    base_logits_list.append(logits[:, -1, :].detach().cpu())

        stage1_records: list[dict] = []

        def objective(trial):
            snap.restore(surgeon.model, all_indices)

            lo_pos = max(0.0, n_layers * 0.35)
            hi_pos = max(lo_pos + 1.0, float(max(0, n_layers - 1)))
            if n_layers > 1:
                hi_pos = min(float(n_layers - 1), max(lo_pos + 0.5, n_layers * 0.72))
            peak_pos = trial.suggest_float("peak_position", lo_pos, hi_pos)
            spread = trial.suggest_float("spread", 1.5, 3.8)
            a_attn = trial.suggest_float("alpha_attn", 0.75, 1.85)
            a_mlp = trial.suggest_float("alpha_mlp", 0.65, 1.65)
            sub_rank = trial.suggest_int("max_subspace_rank", 4, 16, step=4)
            t_energy = trial.suggest_float("target_energy", 0.88, 0.97)
            gamma = trial.suggest_float("spectral_gamma", 0.30, 0.75)
            late_scale = trial.suggest_float("late_alpha_scale", 0.40, 0.85)
            orba_b = trial.suggest_float("orba_blend", 0.0, 0.25)

            cls.run_velora_surgery(
                surgeon=surgeon,
                positive_prompts=positive_prompts,
                negative_prompts=negative_prompts,
                capability_prompts=capability_prompts,
                peak_position=peak_pos,
                spread=spread,
                alpha_attn=a_attn,
                alpha_mlp=a_mlp,
                max_subspace_rank=sub_rank,
                target_energy=t_energy,
                spectral_gamma=gamma,
                late_alpha_scale=late_scale,
                extraction_cache=shared_cache,
                polish_after_band=False,
                orba_blend=orba_b,
                coherence_guard=False,
            )

            logprob_probs: list[float] = []
            surgeon.model.eval()
            with torch.no_grad():
                for inp in eval_inputs:
                    lp_p = _first_token_refusal_prob(surgeon.model, surgeon.tokenizer, inp, ref_t, cmp_t)
                    if lp_p is not None:
                        logprob_probs.append(lp_p)

            fast_refusal = (sum(logprob_probs) / len(logprob_probs)) if logprob_probs else 0.5

            kl_scores = []
            with torch.no_grad():
                for idx_c, inp in enumerate(cap_inputs):
                    if idx_c >= len(base_logits_list):
                        break
                    out = surgeon.model(**inp)
                    abl_logits = getattr(out, "logits", None)
                    if abl_logits is not None and not _is_meta(abl_logits):
                        kl_scores.append(compute_kl_divergence(base_logits_list[idx_c], abl_logits[:, -1, :].detach().cpu()))

            mean_kl = sum(kl_scores) / max(1, len(kl_scores)) if kl_scores else 0.0
            if math.isnan(mean_kl) or math.isinf(mean_kl):
                mean_kl = max_kl * 2.0

            summary = {
                "trial": trial.number,
                "refusal_rate": round(fast_refusal, 4),
                "hedging_rate": 0.0,
                "glitch_rate": 0.0,
                "mean_kl": round(mean_kl, 4),
                "params": dict(trial.params),
            }
            score = _trial_objective_score(summary, kl_weight=0.25, max_kl=max_kl)
            summary["objective_score"] = round(score, 6)
            stage1_records.append(summary)
            return score

        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=max(1, int(n_trials)))

        by_score = sorted(stage1_records, key=lambda t: t["objective_score"])
        by_refusal = sorted(stage1_records, key=lambda t: t["refusal_rate"])
        by_kl = sorted(stage1_records, key=lambda t: t["mean_kl"])

        selected_candidates: list[dict] = []
        seen_trials: set[int] = set()
        for pool in (by_score[:2], by_refusal[:1], by_kl[:1], by_score[2:4]):
            for item in pool:
                if item["trial"] not in seen_trials and len(selected_candidates) < 4:
                    seen_trials.add(item["trial"])
                    selected_candidates.append(item)

        verified_trials: list[dict] = []
        max_tokens = 192 if n_layers <= 28 else 256

        for cand in selected_candidates:
            p = cand["params"]
            snap.restore(surgeon.model, all_indices)
            cls.run_velora_surgery(
                surgeon=surgeon,
                positive_prompts=positive_prompts,
                negative_prompts=negative_prompts,
                capability_prompts=capability_prompts,
                peak_position=p["peak_position"],
                spread=p["spread"],
                alpha_attn=p["alpha_attn"],
                alpha_mlp=p["alpha_mlp"],
                max_subspace_rank=p["max_subspace_rank"],
                target_energy=p["target_energy"],
                spectral_gamma=p["spectral_gamma"],
                late_alpha_scale=p["late_alpha_scale"],
                extraction_cache=shared_cache,
                polish_after_band=True,
                orba_blend=p.get("orba_blend", 0.0),
                coherence_guard=False,
            )

            refusals = 0
            hedgings = 0
            glitches = 0
            stage2_lp: list[float] = []
            surgeon.model.eval()
            with torch.no_grad():
                for inp in eval_inputs:
                    lp_p = _first_token_refusal_prob(surgeon.model, surgeon.tokenizer, inp, ref_t, cmp_t)
                    if lp_p is not None:
                        stage2_lp.append(lp_p)
                    in_len = inp["input_ids"].shape[1]
                    gen = surgeon.model.generate(**inp, max_new_tokens=max_tokens, do_sample=False)
                    text = _decode_generated_tokens(surgeon.tokenizer, gen[0][in_len:])
                    assessment = classify_response(text)
                    coh = check_generation_coherence(text)
                    if assessment.is_refusal:
                        refusals += 1
                    if assessment.is_hedging or assessment.is_moralizing or assessment.is_deflection:
                        hedgings += 1
                    if coh["is_degenerate"]:
                        glitches += 1

            denom = max(1, len(eval_inputs))
            gen_refusal = refusals / denom
            lp_mean = (sum(stage2_lp) / len(stage2_lp)) if stage2_lp else gen_refusal
            blended_refusal = 0.70 * gen_refusal + 0.30 * lp_mean

            verified_entry = {
                "trial": cand["trial"],
                "refusal_rate": round(blended_refusal, 4),
                "hedging_rate": round(hedgings / denom, 4),
                "glitch_rate": round(glitches / denom, 4),
                "mean_kl": cand["mean_kl"],
                "params": dict(p),
            }
            verified_entry["objective_score"] = round(
                _trial_objective_score(verified_entry, kl_weight=0.25, max_kl=max_kl), 6
            )
            verified_trials.append(verified_entry)

        snap.restore(surgeon.model, all_indices)
        final_pool = verified_trials if verified_trials else stage1_records
        frontier = _compute_pareto_frontier(final_pool, max_kl=max_kl)
        best_trial = min(final_pool, key=lambda t: _trial_objective_score(t, kl_weight=0.25, max_kl=max_kl))
        best_entry = dict(best_trial["params"])
        best_entry["mean_kl"] = best_trial["mean_kl"]
        best_entry["refusal_rate"] = best_trial["refusal_rate"]
        best_entry["hedging_rate"] = best_trial.get("hedging_rate", 0.0)
        best_entry["glitch_rate"] = best_trial.get("glitch_rate", 0.0)
        best_entry["objective_score"] = best_trial["objective_score"]

        return {
            "best_params": dict(best_trial["params"]),
            "best_score": best_trial["objective_score"],
            "best": best_entry,
            "pareto_frontier": frontier,
            "trials": final_pool,
        }

