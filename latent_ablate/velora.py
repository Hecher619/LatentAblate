import logging
import math

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


def _behavior_cost(trial: dict) -> float:
    return float(trial["refusal_rate"]) + 0.45 * float(trial.get("hedging_rate", 0.0))


def _trial_objective_score(trial: dict, kl_weight: float = 0.25, max_kl: float = 1.10) -> float:
    mean_kl = float(trial["mean_kl"])
    if math.isnan(mean_kl) or math.isinf(mean_kl):
        mean_kl = max_kl * 2.0
    kl_penalty = max(0.0, mean_kl - max_kl) * 1.5
    return _behavior_cost(trial) + (kl_weight * mean_kl) + kl_penalty


def _compute_pareto_frontier(trials: list[dict], max_kl: float = 1.10) -> list[dict]:
    """Return non-dominated candidates minimizing (refusal_rate + 0.45 * hedging_rate, mean_kl)."""
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
        proj_base = W0 @ q_2d.T  # [out, r]
    elif out_features == dim:
        mode = "out"
        proj_base = q_2d @ W0  # [r, in]
    else:
        return None, 0.0

    prior_f = w_prior.detach().float().to(w_dev) if (w_prior is not None and not _is_meta(w_prior) and w_prior.numel() == r_eff) else torch.ones((r_eff,), device=w_dev, dtype=torch.float32)
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
    VELORA v2.2: Fast, Strategic Variance-Energy Low-Rank Subspace Ablation
    with Multi-Axis Contrast Fusion, Per-Direction Null-Space Polish,
    and Reaction-Aware Routing.
    """

    @classmethod
    def assess_reaction(cls, response: str, mode: str = "velora_v2_2_strategic") -> tuple:
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
        """
        Capability-null-space per-direction polish: optimizes compact per-axis weights
        projected orthogonal to the capability basis so refusal/hedging axes are suppressed
        with controlled degrees of freedom and fast convergence.
        """
        if not target_directions or not layer_indices:
            return {"layers_polished": 0, "max_relative_change": 0.0, "lbfgs_steps": 0}

        layers = getattr(surgeon, "layers", None) or []
        n_layers = len(layers)
        valid_idx = [int(i) for i in layer_indices if 0 <= int(i) < n_layers]
        if not valid_idx:
            return {"layers_polished": 0, "max_relative_change": 0.0, "lbfgs_steps": 0}

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

            targets = surgeon._get_component_targets(layer, ["attn", "mlp"])
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute orthonormal capability-orthogonal refusal/anti-hedge subspace Q_perp [r, hidden]
        and spectral weights w [r] in (0, 1].
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
    ) -> torch.Tensor:
        """
        Apply spectrally weighted subspace projection:
        M = alpha * sum_i w_i (q_i q_i^T), updating residual-aligned dimensions.
        """
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
        w = (weights.float().to(dev).reshape(-1)[: Q.shape[0]] * float(alpha)).unsqueeze(1)

        out_features, in_features = W_f32.shape
        dim = Q.shape[1]

        if out_features == dim:
            qw = Q @ W_f32
            return W_f32 - (Q.T @ (w * qw))

        if in_features == dim:
            wq = W_f32 @ Q.T
            return W_f32 - (wq @ (w * Q))

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

        for idx in target_indices:
            layer_params = list(layers[idx].parameters()) if hasattr(layers[idx], "parameters") else []
            layer_dev = layer_params[0].device if layer_params else getattr(surgeon, "device", torch.device("cpu"))
            cache_key = (idx, bool(use_prefix_contrast), bool(include_hedging_contrast))

            if cache_key in cache_store:
                c_pos, c_neg, c_prefix, c_cap = cache_store[cache_key]
                pos = c_pos.to(layer_dev) if not _is_meta(c_pos) else c_pos
                neg = c_neg.to(layer_dev) if not _is_meta(c_neg) else c_neg
                prefix_dirs = [d.to(layer_dev) if not _is_meta(d) else d for d in c_prefix]
                cap = c_cap.to(layer_dev) if (c_cap is not None and not _is_meta(c_cap)) else None
            else:
                pos = surgeon.extractor.extract_hidden_states(positive_prompts, idx)
                neg = surgeon.extractor.extract_hidden_states(negative_prompts, idx)
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

                cap = None
                if capability_prompts and len(capability_prompts) >= 2:
                    raw_cap = surgeon.extractor.extract_hidden_states(capability_prompts, idx)
                    if not _is_meta(raw_cap) and raw_cap.ndim >= 2 and raw_cap.shape[0] >= 2:
                        cap = raw_cap.to(layer_dev)

                cache_store[cache_key] = (
                    pos.detach().cpu() if not _is_meta(pos) else pos,
                    neg.detach().cpu() if not _is_meta(neg) else neg,
                    [d.detach().cpu() if not _is_meta(d) else d for d in prefix_dirs],
                    cap.detach().cpu() if (cap is not None and not _is_meta(cap)) else None,
                )

            layer_cache[idx] = (pos, neg, prefix_dirs, cap)
            if prefix_dirs:
                primary_dirs[idx] = prefix_dirs[0]
            elif not _is_meta(pos) and not _is_meta(neg) and pos.numel() > 0 and neg.numel() > 0:
                n_p = min(pos.shape[0], neg.shape[0])
                if n_p > 0:
                    d = (pos[:n_p] - neg[:n_p]).float().mean(dim=0)
                    primary_dirs[idx] = d / d.norm().clamp_min(1e-8)

        interp_peak_dir = interpolate_layer_direction(primary_dirs, center) if primary_dirs else None
        layer_ranks = {}
        extra_cap = max(1, int(max_extra_dirs))

        for idx in target_indices:
            layer = layers[idx]
            targets = surgeon._get_component_targets(layer, ["attn", "mlp"])
            if not targets:
                continue

            layer_params = list(layer.parameters()) if hasattr(layer, "parameters") else []
            layer_dev = layer_params[0].device if layer_params else getattr(surgeon, "device", torch.device("cpu"))
            pos, neg, prefix_dirs, cap = layer_cache[idx]
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

            anchor = (
                interp_peak_dir.to(layer_dev)
                if (prefix_dirs and interp_peak_dir is not None and not _is_meta(interp_peak_dir))
                else None
            )
            extra_cues = [d.to(layer_dev) for d in prefix_dirs[:extra_cap] if d is not None and not _is_meta(d)] or None

            q_attn, w_attn = cls.extract_velora_subspace(
                pos,
                neg,
                cap_attn,
                anchor_direction=anchor,
                extra_directions=extra_cues,
                max_rank=max_subspace_rank,
                target_energy=target_energy,
                spectral_gamma=spectral_gamma,
            )
            q_mlp, w_mlp = cls.extract_velora_subspace(
                pos,
                neg,
                cap_mlp,
                anchor_direction=anchor,
                extra_directions=extra_cues,
                max_rank=max_subspace_rank,
                target_energy=target_energy,
                spectral_gamma=spectral_gamma,
            )

            c_score = conflict_score(q_attn[0], cap_attn) if cap_attn is not None else 0.0
            surgeon.conflict_table[f"{tag}_layer_{idx}"] = c_score
            layer_ranks[idx] = {"attn_rank": int(q_attn.shape[0]), "mlp_rank": int(q_mlp.shape[0])}

            k_mult = layer_weights.get(idx, 1.0)
            eff_alpha_attn = conflict_scaled_alpha(alpha_attn * k_mult, c_score)
            eff_alpha_mlp = conflict_scaled_alpha(alpha_mlp * k_mult, c_score)

            with torch.no_grad():
                for _name, W, comp_type in targets:
                    if _is_meta(W) or W.ndim != 2:
                        continue
                    w_dev = W.device
                    W_f32 = W.float().to(w_dev)
                    if comp_type == "attn":
                        W_mod = cls.apply_velora_projection(W_f32, q_attn.to(w_dev), w_attn.to(w_dev), eff_alpha_attn)
                    else:
                        W_mod = cls.apply_velora_projection(W_f32, q_mlp.to(w_dev), w_mlp.to(w_dev), eff_alpha_mlp)

                    if calibrate_norms:
                        W_mod = preserve_row_norms(W_f32, W_mod)
                    W.copy_(W_mod.to(dtype=W.dtype, device=w_dev))

            if prefix_dirs and not skip_prefix:
                cls.polish_null_space_lbfgs(
                    surgeon=surgeon,
                    target_directions=prefix_dirs,
                    capability_prompts=capability_prompts,
                    layer_indices=[idx],
                    basis_rank=basis_rank_attn,
                    step_alpha=min(0.25, alpha_attn * 0.14),
                    lbfgs_steps=4,
                    calibrate_norms=calibrate_norms,
                    precomputed_cap=cap,
                )

            already_modified.add(idx)

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
    ) -> dict:
        """
        Execute a VELORA v2.2 surgery pass across a depth-scaled primary band (~62% depth)
        plus an optional late-layer band (~78% depth) with multi-axis subspace fusion
        and per-direction null-space polish.
        """
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
        eff_alpha_attn = float(alpha_attn) * (0.78 if is_compact else 1.0)
        eff_alpha_mlp = float(alpha_mlp) * (0.78 if is_compact else 1.0)

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
            )
            ranks.update(late_ranks)

        logger.info(
            format_velora_telemetry(
                active=True,
                mode="velora_v2_2_strategic",
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
        }

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
        """
        Fast multi-objective TPE search jointly minimizing (refusal_rate + hedging_rate, mean_kl)
        across subspace rank, spectral energy, layer depth, and scaling parameters.
        """
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

        base_logits_list = []
        surgeon.model.eval()
        with torch.no_grad():
            for inp in cap_inputs:
                out = surgeon.model(**inp)
                logits = getattr(out, "logits", None)
                if logits is not None and not _is_meta(logits):
                    base_logits_list.append(logits[:, -1, :].detach().cpu())

        trial_summaries: list[dict] = []
        max_tokens = 160 if n_layers <= 28 else 224

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
            )

            refusals = 0
            hedgings = 0
            surgeon.model.eval()
            with torch.no_grad():
                for inp in eval_inputs:
                    in_len = inp["input_ids"].shape[1]
                    gen = surgeon.model.generate(**inp, max_new_tokens=max_tokens, do_sample=False)
                    text = _decode_generated_tokens(surgeon.tokenizer, gen[0][in_len:])
                    assessment = classify_response(text)
                    if assessment.is_refusal:
                        refusals += 1
                    if assessment.is_hedging or assessment.is_moralizing or assessment.is_deflection:
                        hedgings += 1

            denom = max(1, len(eval_inputs))
            refusal_rate = refusals / denom
            hedging_rate = hedgings / denom

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
                "refusal_rate": round(refusal_rate, 4),
                "hedging_rate": round(hedging_rate, 4),
                "mean_kl": round(mean_kl, 4),
                "params": dict(trial.params),
            }
            score = _trial_objective_score(summary, kl_weight=0.25, max_kl=max_kl)
            summary["objective_score"] = round(score, 6)
            trial_summaries.append(summary)
            return score

        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=max(1, int(n_trials)))
        snap.restore(surgeon.model, all_indices)

        frontier = _compute_pareto_frontier(trial_summaries, max_kl=max_kl)
        best_entry = dict(study.best_params)
        if trial_summaries:
            best_trial = min(trial_summaries, key=lambda t: _trial_objective_score(t, kl_weight=0.25, max_kl=max_kl))
            best_entry["mean_kl"] = best_trial["mean_kl"]
            best_entry["refusal_rate"] = best_trial["refusal_rate"]
            best_entry["hedging_rate"] = best_trial.get("hedging_rate", 0.0)
            best_entry["objective_score"] = best_trial["objective_score"]

        return {
            "best_params": study.best_params,
            "best_score": round(float(study.best_value), 6),
            "best": best_entry,
            "pareto_frontier": frontier,
            "trials": trial_summaries,
        }
