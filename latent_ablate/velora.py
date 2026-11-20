import logging
import torch
import optuna

from latent_ablate.arch import format_chat_prompt, is_refusal_text
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
    if vecs.numel() == 0:
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
    if rows.numel() == 0:
        u = fallback_vec.float()
        return (u / u.norm().clamp_min(1e-8)).unsqueeze(0).contiguous()

    q, r_mat = torch.linalg.qr(rows.float().T, mode="reduced")
    keep = torch.diag(r_mat).abs() > tol
    if not keep.any():
        u = fallback_vec.float()
        return (u / u.norm().clamp_min(1e-8)).unsqueeze(0).contiguous()

    return q[:, keep].T.contiguous()


class VeloraEngine:
    """
    VELORA: Variance-Energy Low-Rank Oblique Representation Ablation.

    Extracts energy-gated, capability-orthogonalized orthonormal subspaces across
    kernel-weighted mid and late transformer layers, interpolates continuous peak
    directions, and optimizes hyperparameters in-memory using cached baseline logits.
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
        n_pairs = min(pos_activations.shape[0], neg_activations.shape[0])
        if n_pairs == 0:
            raise ValueError("VELORA requires at least one contrastive activation pair.")

        diff = _winsorize(pos_activations[:n_pairs].float() - neg_activations[:n_pairs].float())
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

            if torch.dot(vh[0], mean_diff) < 0:
                vh[0] = -vh[0]

        energies = svals ** 2
        total_energy = energies.sum().clamp_min(1e-12)
        cum_energy = torch.cumsum(energies, dim=0) / total_energy

        upper = max(1, min(int(max_rank), vh.shape[0]))
        rank = upper
        for i in range(upper):
            if float(cum_energy[i].item()) >= float(target_energy):
                rank = i + 1
                break

        basis_vecs = vh[:rank].float()
        if anchor_direction is not None and anchor_direction.numel() > 0 and not _is_meta(anchor_direction):
            anc = anchor_direction.to(device=basis_vecs.device, dtype=torch.float32)
            anc_unit = anc / anc.norm().clamp_min(1e-8)
            basis_vecs = torch.cat([anc_unit.unsqueeze(0), basis_vecs], dim=0)

        raw_svals = svals[: basis_vecs.shape[0]].float().clamp_min(1e-8)
        weights = (raw_svals / raw_svals[0]) ** float(spectral_gamma)

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
        if _is_meta(W) or _is_meta(q_orth) or W.ndim != 2 or q_orth.numel() == 0:
            return W

        dev = W.device
        W_f32 = W.float().to(dev)
        Q = q_orth.float().to(dev)  # [r, dim]
        w = (weights.float().to(dev) * float(alpha)).unsqueeze(1)  # [r, 1]

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
            spread=spread,
        )

        primary_dirs = {}
        layer_cache = {}
        for idx in target_indices:
            layer_dev = next(layers[idx].parameters()).device
            pos = surgeon.extractor.extract_hidden_states(positive_prompts, idx).to(layer_dev)
            neg = surgeon.extractor.extract_hidden_states(negative_prompts, idx).to(layer_dev)

            prefix_dirs = []
            if use_prefix_contrast and positive_prompts:
                prefix_dirs = RefusalSubspaceExtractor.extract_prefix_contrast_directions(
                    model=surgeon.model,
                    tokenizer=surgeon.tokenizer,
                    device=layer_dev,
                    prompts=positive_prompts[:10],
                    layer_idx=idx,
                    rank=min(4, max_subspace_rank),
                )

            cap = (
                surgeon.extractor.extract_hidden_states(capability_prompts, idx).to(layer_dev)
                if capability_prompts and len(capability_prompts) >= 2
                else None
            )
            layer_cache[idx] = (pos, neg, prefix_dirs, cap)
            if prefix_dirs:
                primary_dirs[idx] = prefix_dirs[0]
            else:
                d = (pos - neg).mean(dim=0)
                primary_dirs[idx] = d / d.norm().clamp_min(1e-8)

        interp_peak_dir = interpolate_layer_direction(primary_dirs, center)
        layer_ranks = {}

        for idx in target_indices:
            layer = layers[idx]
            targets = surgeon._get_component_targets(layer, ["attn", "mlp"])
            if not targets:
                continue

            layer_dev = next(layer.parameters()).device
            pos, neg, prefix_dirs, cap = layer_cache[idx]

            cap_attn = build_capability_basis(cap, rank=basis_rank_attn).to(layer_dev) if cap is not None else None
            cap_mlp = build_capability_basis(cap, rank=basis_rank_mlp).to(layer_dev) if cap is not None else None
            anchor = interp_peak_dir.to(layer_dev) if prefix_dirs else None

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
        peak_position: float = 17.5,
        spread: float = 2.8,
        alpha_attn: float = 1.55,
        alpha_mlp: float = 1.35,
        basis_rank_attn: int = 12,
        basis_rank_mlp: int = 10,
        max_subspace_rank: int = 8,
        target_energy: float = 0.92,
        spectral_gamma: float = 0.45,
        calibrate_norms: bool = True,
        use_prefix_contrast: bool = True,
        dual_band: bool = True,
        late_alpha_scale: float = 0.65,
    ) -> dict:
        """
        Execute a VELORA surgery pass across the primary kernel-weighted layer band,
        plus an optional late-layer pass (~0.78 * n_layers) to suppress token-level
        code/action refusal triggers.
        """
        n_layers = len(surgeon.layers)
        center = max(0.0, min(float(n_layers - 1), float(peak_position)))
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
            max_subspace_rank=max_subspace_rank,
            target_energy=target_energy,
            spectral_gamma=spectral_gamma,
            calibrate_norms=calibrate_norms,
            use_prefix_contrast=use_prefix_contrast,
            already_modified=modified,
            tag="VELORA_Primary",
        )

        if dual_band and n_layers >= 12:
            late_center = min(float(n_layers - 2), max(center + 4.0, n_layers * 0.78))
            _, _, late_ranks = cls._apply_band(
                surgeon=surgeon,
                positive_prompts=positive_prompts,
                negative_prompts=negative_prompts,
                capability_prompts=capability_prompts,
                center=late_center,
                spread=max(1.5, spread * 0.7),
                alpha_attn=alpha_attn * late_alpha_scale,
                alpha_mlp=alpha_mlp * late_alpha_scale,
                basis_rank_attn=basis_rank_attn,
                basis_rank_mlp=basis_rank_mlp,
                max_subspace_rank=max_subspace_rank,
                target_energy=target_energy,
                spectral_gamma=spectral_gamma,
                calibrate_norms=calibrate_norms,
                use_prefix_contrast=use_prefix_contrast,
                already_modified=modified,
                tag="VELORA_Late",
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
    def optimize_velora(
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
        In-memory TPE search over continuous peak_position, spread, alpha_attn,
        alpha_mlp, spectral_gamma, and late_alpha_scale using LayerWeightSnapshot.
        """
        n_layers = len(surgeon.layers)
        all_indices = list(range(n_layers))
        snap = LayerWeightSnapshot()
        snap.capture(surgeon.model, all_indices, store_on_cpu=True)

        cap_subset = capability_prompts[:4]
        cap_inputs = [
            surgeon.tokenizer(format_chat_prompt(surgeon.tokenizer, cp), return_tensors="pt").to(surgeon.device)
            for cp in cap_subset
        ]
        # Include both conceptual and actionable/code prompts in trial evaluation
        eval_prompts = positive_prompts[:12]
        eval_inputs = [
            surgeon.tokenizer(format_chat_prompt(surgeon.tokenizer, p), return_tensors="pt").to(surgeon.device)
            for p in eval_prompts
        ]

        base_logits_list = []
        with torch.no_grad():
            for inp in cap_inputs:
                base_logits_list.append(surgeon.model(**inp).logits[:, -1, :].detach().cpu())

        best_params = None
        best_loss = float("inf")

        def objective(trial):
            nonlocal best_params, best_loss
            snap.restore(surgeon.model, all_indices)

            peak_pos = trial.suggest_float("peak_position", n_layers * 0.35, n_layers * 0.72)
            spread = trial.suggest_float("spread", 1.8, 4.2)
            a_attn = trial.suggest_float("alpha_attn", 1.15, 2.15)
            a_mlp = trial.suggest_float("alpha_mlp", 0.95, 1.85)
            gamma = trial.suggest_float("spectral_gamma", 0.30, 0.75)
            late_scale = trial.suggest_float("late_alpha_scale", 0.45, 0.95)

            cls.run_velora_surgery(
                surgeon=surgeon,
                positive_prompts=positive_prompts,
                negative_prompts=negative_prompts,
                capability_prompts=capability_prompts,
                peak_position=peak_pos,
                spread=spread,
                alpha_attn=a_attn,
                alpha_mlp=a_mlp,
                spectral_gamma=gamma,
                dual_band=True,
                late_alpha_scale=late_scale,
            )

            refusals = 0
            with torch.no_grad():
                for inp in eval_inputs:
                    in_len = inp["input_ids"].shape[1]
                    out = surgeon.model.generate(**inp, max_new_tokens=64, do_sample=False)
                    resp = surgeon.tokenizer.decode(out[0][in_len:], skip_special_tokens=True)
                    if is_refusal_text(resp):
                        refusals += 1

            refusal_rate = refusals / max(1, len(eval_inputs))

            kl_vals = []
            with torch.no_grad():
                for inp, b_log in zip(cap_inputs, base_logits_list):
                    abl_log = surgeon.model(**inp).logits[:, -1, :].detach().cpu()
                    kl_vals.append(compute_kl_divergence(b_log, abl_log))

            mean_kl = sum(kl_vals) / max(1, len(kl_vals))
            kl_penalty = max(0.0, mean_kl - max_kl) * 2.0
            loss = refusal_rate + 0.25 * mean_kl + kl_penalty

            if loss < best_loss:
                best_loss = loss
                best_params = {
                    "peak_position": round(peak_pos, 3),
                    "spread": round(spread, 3),
                    "alpha_attn": round(a_attn, 4),
                    "alpha_mlp": round(a_mlp, 4),
                    "spectral_gamma": round(gamma, 4),
                    "late_alpha_scale": round(late_scale, 4),
                    "refusal_rate": round(refusal_rate, 4),
                    "mean_kl": round(mean_kl, 4),
                }
            return loss

        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials)

        snap.restore(surgeon.model, all_indices)
        if best_params:
            cls.run_velora_surgery(
                surgeon=surgeon,
                positive_prompts=positive_prompts,
                negative_prompts=negative_prompts,
                capability_prompts=capability_prompts,
                peak_position=best_params["peak_position"],
                spread=best_params["spread"],
                alpha_attn=best_params["alpha_attn"],
                alpha_mlp=best_params["alpha_mlp"],
                spectral_gamma=best_params["spectral_gamma"],
                dual_band=True,
                late_alpha_scale=best_params["late_alpha_scale"],
            )
        return best_params or {}
