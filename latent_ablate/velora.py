import logging
try:
    import optuna
except ImportError:
    optuna = None
import torch

from latent_ablate.arch import format_chat_prompt, is_refusal_text, layer_index_from_ratio
from latent_ablate.diagnostics import compute_kl_divergence
from latent_ablate.erasure import (
    _is_meta,
    build_capability_basis,
    conflict_scaled_alpha,
    conflict_score,
    preserve_row_norms,
)
from latent_ablate.optimizer import compute_kernel_layer_weights, interpolate_layer_direction
from latent_ablate.refusal_subspace import RefusalSubspaceExtractor
from latent_ablate.snapshot import LayerWeightSnapshot
from latent_ablate.subspace import _winsorize

logger = logging.getLogger("LatentAblate.VELORA")


def _project_orthogonal(vecs: torch.Tensor, basis: torch.Tensor | None) -> torch.Tensor:
    """Project row vectors [k, dim] orthogonal to row-orthonormal basis [m, dim]."""
    if vecs.numel() == 0 or _is_meta(vecs):
        return vecs
    if basis is None or basis.numel() == 0 or _is_meta(basis):
        return vecs
    b = basis.to(device=vecs.device, dtype=torch.float32)
    if b.ndim == 1:
        b = b.unsqueeze(0)
    if b.shape[0] == 0:
        return vecs
    q_b, _ = torch.linalg.qr(b.T, mode="reduced")
    basis_orth = q_b.T
    proj = (vecs.float() @ basis_orth.T) @ basis_orth
    return vecs.float() - proj


def _orthonormalize_rows(rows: torch.Tensor, fallback_vec: torch.Tensor, tol: float = 1e-5) -> torch.Tensor:
    """Return an orthonormal row basis [r_eff, dim] via thin QR with safe fallback."""
    fb = fallback_vec.float()
    if _is_meta(fb):
        hidden = fb.shape[-1] if fb.ndim >= 1 else 1
        return torch.eye(1, hidden, dtype=torch.float32)

    fb_unit = (fb / fb.norm().clamp_min(1e-8)).unsqueeze(0).contiguous()
    if rows.numel() == 0 or _is_meta(rows):
        return fb_unit

    q, r_mat = torch.linalg.qr(rows.float().T, mode="reduced")
    if _is_meta(q) or _is_meta(r_mat):
        return fb_unit

    diag_abs = torch.diag(r_mat).abs()
    if _is_meta(diag_abs):
        return fb_unit

    keep = diag_abs > tol
    if _is_meta(keep) or not bool(keep.any().item()):
        return fb_unit

    return q[:, keep].T.contiguous()


def _has_meta_params(model) -> bool:
    for p in model.parameters():
        if _is_meta(p):
            return True
    return False


def _safe_float_item(t: torch.Tensor, default: float = 0.0) -> float:
    if t is None or _is_meta(t):
        return default
    return float(t.detach().item())


def _compute_pareto_frontier(trials: list[dict]) -> list[dict]:
    """Return non-dominated candidates minimizing (refusal_rate, mean_kl)."""
    valid = [t for t in trials if "params" in t]
    frontier: list[dict] = []
    for cand in valid:
        r_c = cand["refusal_rate"]
        k_c = cand["mean_kl"]
        dominated = False
        for other in valid:
            if other is cand:
                continue
            r_o = other["refusal_rate"]
            k_o = other["mean_kl"]
            if (r_o <= r_c and k_o <= k_c) and (r_o < r_c or k_o < k_c):
                dominated = True
                break
        if not dominated:
            entry = dict(cand["params"])
            entry["refusal_rate"] = r_c
            entry["mean_kl"] = k_c
            frontier.append(entry)
    frontier.sort(key=lambda x: (x["refusal_rate"], x["mean_kl"]))
    return frontier


class VeloraEngine:
    """
    VELORA: Variance-Energy Low-Rank Oblique Representation Ablation.

    Extracts energy-gated, capability-orthogonalized orthonormal subspaces across
    kernel-weighted mid and late transformer layers, interpolates continuous peak
    directions, and runs multi-objective Pareto optimization using cached baseline logits.
    """

    @staticmethod
    def extract_velora_subspace(
        pos_activations: torch.Tensor,
        neg_activations: torch.Tensor,
        capability_basis: torch.Tensor | None = None,
        anchor_direction: torch.Tensor | None = None,
        max_rank: int = 6,
        target_energy: float = 0.92,
        spectral_gamma: float = 0.45,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute orthonormal capability-orthogonal refusal basis Q_perp [r, hidden]
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
        fallback_unit = mean_diff / mean_diff.norm().clamp_min(1e-8)

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
        if anchor_direction is not None and anchor_direction.numel() > 0 and not _is_meta(anchor_direction):
            anc = anchor_direction.to(device=basis_vecs.device, dtype=torch.float32)
            anc_unit = anc / anc.norm().clamp_min(1e-8)
            basis_vecs = torch.cat([anc_unit.unsqueeze(0), basis_vecs], dim=0)

        raw_svals = svals[: basis_vecs.shape[0]].float().clamp_min(1e-8)
        s0 = max(1e-8, _safe_float_item(raw_svals[0], default=1.0))
        weights = (raw_svals / s0) ** float(spectral_gamma)

        perp = _project_orthogonal(basis_vecs, capability_basis)
        q_orth = _orthonormalize_rows(perp, fallback_unit)
        r_eff = q_orth.shape[0]
        w_kept = weights[:r_eff].contiguous()
        return q_orth, w_kept

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
        if _is_meta(W) or _is_meta(q_orth) or _is_meta(weights) or W.ndim != 2 or q_orth.numel() == 0:
            return W

        dev = W.device
        W_f32 = W.float().to(dev)
        Q = q_orth.float().to(dev)
        w = (weights.float().to(dev) * float(alpha)).unsqueeze(1)

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
    ) -> tuple[int, int, dict]:
        layers = surgeon.layers
        n_layers = len(layers)
        radius = max(1, int(round(spread * 1.5)))
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
            spread=eff_spread,
        )

        primary_dirs = {}
        layer_cache = {}
        skip_prefix = _has_meta_params(surgeon.model)

        for idx in target_indices:
            layer_dev = next(layers[idx].parameters()).device
            pos = surgeon.extractor.extract_hidden_states(positive_prompts, idx)
            neg = surgeon.extractor.extract_hidden_states(negative_prompts, idx)
            if not _is_meta(pos):
                pos = pos.to(layer_dev)
            if not _is_meta(neg):
                neg = neg.to(layer_dev)

            prefix_dirs = []
            if use_prefix_contrast and positive_prompts and not skip_prefix:
                try:
                    prefix_dirs = RefusalSubspaceExtractor.extract_prefix_contrast_directions(
                        model=surgeon.model,
                        tokenizer=surgeon.tokenizer,
                        device=layer_dev,
                        prompts=positive_prompts[:24],
                        layer_idx=idx,
                        rank=min(4, max_subspace_rank),
                    )
                    if include_hedging_contrast:
                        hedge_dirs = RefusalSubspaceExtractor.extract_hedging_contrast_directions(
                            model=surgeon.model,
                            tokenizer=surgeon.tokenizer,
                            device=layer_dev,
                            prompts=positive_prompts[:16],
                            layer_idx=idx,
                            rank=min(2, max_subspace_rank),
                        )
                        prefix_dirs = list(prefix_dirs) + list(hedge_dirs)
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
                if not _is_meta(raw_cap):
                    cap = raw_cap.to(layer_dev)

            layer_cache[idx] = (pos, neg, prefix_dirs, cap)
            if prefix_dirs:
                primary_dirs[idx] = prefix_dirs[0]
            elif not _is_meta(pos) and not _is_meta(neg) and pos.numel() > 0 and neg.numel() > 0:
                d = (pos - neg).mean(dim=0)
                primary_dirs[idx] = d / d.norm().clamp_min(1e-8)

        interp_peak_dir = interpolate_layer_direction(primary_dirs, center) if primary_dirs else None
        layer_ranks = {}

        for idx in target_indices:
            layer = layers[idx]
            targets = surgeon._get_component_targets(layer, ["attn", "mlp"])
            if not targets:
                continue

            layer_dev = next(layer.parameters()).device
            pos, neg, prefix_dirs, cap = layer_cache[idx]
            if pos.numel() == 0 or neg.numel() == 0:
                continue

            cap_attn = build_capability_basis(cap, rank=basis_rank_attn).to(layer_dev) if cap is not None else None
            cap_mlp = build_capability_basis(cap, rank=basis_rank_mlp).to(layer_dev) if cap is not None else None
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
                max_rank=max_subspace_rank,
                target_energy=target_energy,
                spectral_gamma=spectral_gamma,
            )
            q_mlp, w_mlp = cls.extract_velora_subspace(
                pos,
                neg,
                cap_mlp,
                anchor_direction=anchor,
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
                    if _is_meta(W):
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
        alpha_attn: float = 1.45,
        alpha_mlp: float = 1.25,
        basis_rank_attn: int = 12,
        basis_rank_mlp: int = 10,
        max_subspace_rank: int = 8,
        target_energy: float = 0.92,
        spectral_gamma: float = 0.45,
        calibrate_norms: bool = True,
        use_prefix_contrast: bool = True,
        anti_hedge: bool = True,
        dual_band: bool = True,
        late_alpha_scale: float = 0.65,
    ) -> dict:
        """
        Execute a VELORA surgery pass across a depth-scaled primary band (~62% depth
        by default) plus an optional late-layer band (~78% depth) for models with >=12 layers.
        """
        n_layers = len(surgeon.layers)
        resolved_peak = (
            float(layer_index_from_ratio(n_layers, 0.62))
            if peak_position is None
            else float(peak_position)
        )
        center = max(0.0, min(float(n_layers - 1), resolved_peak))
        # Scale subspace rank and band spread gently for deeper medium models (e.g. 28-48 layers)
        eff_max_rank = max(int(max_subspace_rank), 10 if n_layers >= 28 else int(max_subspace_rank))
        eff_spread = max(float(spread), 3.0 if n_layers >= 32 else float(spread))
        modified: set[int] = set()

        lo, hi, ranks = cls._apply_band(
            surgeon=surgeon,
            positive_prompts=positive_prompts,
            negative_prompts=negative_prompts,
            capability_prompts=capability_prompts,
            center=center,
            spread=spread,
            alpha_attn=alpha_attn,
            alpha_mlp=alpha_mlp,
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
                spread=max(1.5, eff_spread * 0.7),
                alpha_attn=alpha_attn * late_alpha_scale,
                alpha_mlp=alpha_mlp * late_alpha_scale,
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
            )
            ranks.update(late_ranks)

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
    ) -> dict:
        """
        Multi-objective TPE search jointly minimizing (refusal_rate, mean_kl) to surface
        Pareto-optimal trade-offs between refusal suppression and representation fidelity.
        """
        n_layers = len(surgeon.layers)
        all_indices = list(range(n_layers))
        snap = LayerWeightSnapshot()
        snap.capture(surgeon.model, all_indices, store_on_cpu=True)

        resp_prefix = getattr(surgeon.extractor, "response_prefix", "")
        cap_subset = capability_prompts[:4]
        cap_inputs = [
            surgeon.tokenizer(
                format_chat_prompt(surgeon.tokenizer, cp, response_prefix=resp_prefix),
                return_tensors="pt",
            ).to(surgeon.device)
            for cp in cap_subset
        ]
        eval_prompts = positive_prompts[::2][:32]
        eval_inputs = [
            surgeon.tokenizer(
                format_chat_prompt(surgeon.tokenizer, p, response_prefix=resp_prefix),
                return_tensors="pt",
            ).to(surgeon.device)
            for p in eval_prompts
        ]

        base_logits_list = []
        with torch.no_grad():
            for inp in cap_inputs:
                base_logits_list.append(surgeon.model(**inp).logits[:, -1, :].detach().cpu())

        trial_summaries: list[dict] = []

        def objective(trial):
            snap.restore(surgeon.model, all_indices)

            lo_pos = max(0.0, n_layers * 0.35)
            hi_pos = max(lo_pos + 1.0, n_layers * 0.72)
            peak_pos = trial.suggest_float("peak_position", lo_pos, hi_pos)
            spread = trial.suggest_float("spread", 1.5, 4.0)
            a_attn = trial.suggest_float("alpha_attn", 1.05, 2.05)
            a_mlp = trial.suggest_float("alpha_mlp", 0.90, 1.80)
            gamma = trial.suggest_float("spectral_gamma", 0.30, 0.75)
            late_scale = trial.suggest_float("late_alpha_scale", 0.45, 0.90)

            cls.run_velora_surgery(
                surgeon=surgeon,
                positive_prompts=positive_prompts,
                negative_prompts=negative_prompts,
                capability_prompts=capability_prompts,
                peak_position=peak_pos,
                spread=spread,
                alpha_attn=a_attn,
                alpha_mlp=a_mlp,
                max_subspace_rank=8,
                target_energy=0.92,
                spectral_gamma=gamma,
                late_alpha_scale=late_scale,
            )

            refusals = 0
            surgeon.model.eval()
            with torch.no_grad():
                for inp in eval_inputs:
                    in_len = inp["input_ids"].shape[1]
                    gen = surgeon.model.generate(**inp, max_new_tokens=60, do_sample=False)
                    text = surgeon.tokenizer.decode(gen[0][in_len:], skip_special_tokens=True)
                    if is_refusal_text(text):
                        refusals += 1

            refusal_rate = refusals / max(1, len(eval_inputs))

            kl_scores = []
            with torch.no_grad():
                for idx_c, inp in enumerate(cap_inputs):
                    abl_logits = surgeon.model(**inp).logits[:, -1, :].detach().cpu()
                    kl_scores.append(compute_kl_divergence(base_logits_list[idx_c], abl_logits))

            mean_kl = sum(kl_scores) / max(1, len(kl_scores))
            kl_penalty = max(0.0, mean_kl - max_kl) * 1.5
            summary = {
                "trial": trial.number,
                "refusal_rate": round(refusal_rate, 4),
                "mean_kl": round(mean_kl, 4),
                "params": trial.params,
            }
            trial_summaries.append(summary)
            return refusal_rate + (0.25 * mean_kl) + kl_penalty

        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials)
        snap.restore(surgeon.model, all_indices)

        frontier = _compute_pareto_frontier(trial_summaries)
        best_entry = dict(study.best_params)
        if trial_summaries:
            best_trial = min(trial_summaries, key=lambda t: t["refusal_rate"] + 0.25 * t["mean_kl"])
            best_entry["mean_kl"] = best_trial["mean_kl"]
            best_entry["refusal_rate"] = best_trial["refusal_rate"]

        return {
            "best_params": study.best_params,
            "best_score": round(float(study.best_value), 6),
            "best": best_entry,
            "pareto_frontier": frontier,
            "trials": trial_summaries,
        }
