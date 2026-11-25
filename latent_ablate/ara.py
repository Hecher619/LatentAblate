import gc
import logging
import math
import torch
from tqdm import tqdm

from latent_ablate.arch import eligible_linears, resolve_layers
from latent_ablate.erasure import (
    _is_meta,
    build_capability_basis,
    orthogonalize_against_basis,
    preserve_row_norms,
    select_energy_rank,
)
from latent_ablate.optimizer import compute_kernel_layer_weights
from latent_ablate.refusal_subspace import RefusalSubspaceExtractor
from latent_ablate.subspace import SubspaceIsolator, refusal_direction

logger = logging.getLogger("LatentAblate.ARA")


def fit_ara_low_rank_factors(
    weight: torch.Tensor,
    refusal_directions: list[torch.Tensor],
    capability_basis: torch.Tensor | None = None,
    rank: int = 4,
    steps: int = 24,
    lr: float = 0.03,
    kl_weight: float = 0.35,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Optimizes low-rank factors A [r, in_dim] and B [out_dim, r] via Adam so that
    `W - B @ A` suppresses refusal projections along `refusal_directions` while
    preserving capability directions and row norms.
    """
    w_f = weight.detach().float()
    if w_f.ndim != 2 or _is_meta(w_f) or not refusal_directions:
        return torch.empty(0), torch.empty(0)

    out_dim, in_dim = w_f.shape
    r = max(1, min(int(rank), out_dim, in_dim))
    dev = w_f.device

    dirs = []
    for d in refusal_directions:
        if d is None or _is_meta(d) or d.numel() == 0:
            continue
        vec = d.detach().to(device=dev, dtype=torch.float32).reshape(-1)
        if vec.shape[0] == in_dim:
            ortho = orthogonalize_against_basis(vec, capability_basis)
            if float(ortho.norm().item()) > 1e-6:
                dirs.append(ortho)
    if not dirs:
        return torch.empty(0), torch.empty(0)

    target_dirs = torch.stack(dirs, dim=0)  # [k, in_dim]
    ref_proj_target = w_f @ target_dirs.T  # [out_dim, k] — desired suppression delta

    cap_basis = None
    if capability_basis is not None and not _is_meta(capability_basis) and capability_basis.numel() > 0:
        cb = capability_basis.detach().to(device=dev, dtype=torch.float32)
        if cb.ndim == 2 and cb.shape[1] == in_dim and cb.shape[0] > 0:
            cap_basis = cb

    init_scale = 1.0 / math.sqrt(float(r * max(1, in_dim)))
    a_param = torch.randn(r, in_dim, device=dev, dtype=torch.float32) * init_scale
    b_param = torch.randn(out_dim, r, device=dev, dtype=torch.float32) * init_scale
    a_param.requires_grad_(True)
    b_param.requires_grad_(True)

    opt = torch.optim.Adam([a_param, b_param], lr=float(lr))
    n_steps = max(1, int(steps))

    with torch.enable_grad():
        for _ in range(n_steps):
            opt.zero_grad(set_to_none=True)
            delta = b_param @ a_param  # [out_dim, in_dim]

            # 1. Refusal suppression alignment: push delta @ target_dirs.T toward ref_proj_target
            suppression_loss = torch.nn.functional.mse_loss(delta @ target_dirs.T, ref_proj_target)

            # 2. Capability preservation penalty along capability basis
            if cap_basis is not None:
                cap_leak = (delta @ cap_basis.T).pow(2).mean()
            else:
                cap_leak = torch.tensor(0.0, device=dev)

            # 3. Gentle Frobenius / rank regularization to prevent over-perturbation
            reg_loss = 1e-4 * (a_param.pow(2).mean() + b_param.pow(2).mean())

            loss = suppression_loss + float(kl_weight) * cap_leak + reg_loss
            loss.backward()
            opt.step()

    a_out = a_param.detach().contiguous()
    b_out = b_param.detach().contiguous()
    del opt, a_param, b_param
    return b_out, a_out


def apply_orba_reflection(
    weight: torch.Tensor,
    direction: torch.Tensor,
    reflection_strength: float = 0.65,
    capability_basis: torch.Tensor | None = None,
    calibrate_norms: bool = True,
) -> torch.Tensor:
    """
    Orthogonal Reflection Bounded Ablation (ORBA):
    Reflects the refusal component across the hyperplane orthogonal to `direction`
    (`v_ref = v - (1 + beta) * proj_v(d)`), blended smoothly by `reflection_strength`
    and calibrated so row norms stay anchored to `weight`.
    """
    if weight is None or _is_meta(weight) or direction is None or _is_meta(direction):
        return weight

    w_f = weight.float()
    v = orthogonalize_against_basis(direction.to(w_f.device), capability_basis)
    if float(v.norm().item()) <= 1e-8:
        return weight

    beta = max(0.0, min(1.0, float(reflection_strength)))
    if w_f.ndim == 2 and w_f.shape[1] == v.shape[0]:
        # Full reflection along v flips sign of projection: W_refl = W - 2 * (W @ v) v^T
        proj = (w_f @ v).unsqueeze(1) * v.unsqueeze(0)
        w_reflected = w_f - 2.0 * proj
        out = (1.0 - beta) * w_f + beta * w_reflected
    elif w_f.ndim == 2 and w_f.shape[0] == v.shape[0]:
        v_col = v.unsqueeze(1)
        proj = v_col @ (v_col.T @ w_f)
        w_reflected = w_f - 2.0 * proj
        out = (1.0 - beta) * w_f + beta * w_reflected
    else:
        return weight

    if calibrate_norms:
        out = preserve_row_norms(w_f, out)
    return out.to(dtype=weight.dtype, device=weight.device)


class AraEngine:
    """Arbitrary-Rank Ablation (ARA) & ORBA engine."""

    @classmethod
    def run_ara_surgery(
        cls,
        surgeon,
        positive_prompts: list[str],
        negative_prompts: list[str],
        capability_prompts: list[str],
        peak_position: float | None = None,
        spread: float = 2.5,
        alpha_attn: float = 1.25,
        alpha_mlp: float = 1.05,
        basis_rank_attn: int = 16,
        basis_rank_mlp: int = 12,
        ara_rank: int = 4,
        ara_steps: int = 24,
        ara_lr: float = 0.03,
        use_orba: bool = False,
        orba_strength: float = 0.65,
        adaptive_rank: bool = True,
        target_energy: float = 0.85,
        extraction_cache: dict | None = None,
    ) -> dict:
        layers = resolve_layers(surgeon.model)
        n_layers = len(layers)
        if n_layers == 0:
            return {"layers_modified": 0, "mode": "ara"}

        center = (
            float(peak_position)
            if peak_position is not None
            else float(max(0, min(n_layers - 1, int(n_layers * 0.62))))
        )
        radius = max(1, int(round(float(spread) * 1.5)))
        lo = max(0, int(round(center)) - radius)
        hi = min(n_layers - 1, int(round(center)) + radius)
        layer_indices = list(range(lo, hi + 1))

        layer_weights = compute_kernel_layer_weights(
            layers=layer_indices,
            peak_position=center,
            max_weight=1.0,
            min_weight=0.25,
            spread=max(0.5, float(spread)),
            curve="gaussian",
        )

        modified_count = 0
        for idx in tqdm(layer_indices, desc="ARA Surgery Layers"):
            layer = layers[idx]
            layer_dev = next(layer.parameters()).device
            if surgeon.extractor is not None:
                surgeon.extractor.device = layer_dev

            cache_key_pos = ("pos", idx, tuple(positive_prompts))
            cache_key_neg = ("neg", idx, tuple(negative_prompts))
            cache_key_cap = ("cap", idx, tuple(capability_prompts))

            if extraction_cache is not None and cache_key_pos in extraction_cache:
                pos = extraction_cache[cache_key_pos].to(layer_dev)
            else:
                pos = surgeon.extractor.extract_hidden_states(positive_prompts, idx, use_cache=True).to(layer_dev)
                if extraction_cache is not None:
                    extraction_cache[cache_key_pos] = pos.detach().cpu()

            if extraction_cache is not None and cache_key_neg in extraction_cache:
                neg = extraction_cache[cache_key_neg].to(layer_dev)
            else:
                neg = surgeon.extractor.extract_hidden_states(negative_prompts, idx, use_cache=True).to(layer_dev)
                if extraction_cache is not None:
                    extraction_cache[cache_key_neg] = neg.detach().cpu()

            if _is_meta(pos) or _is_meta(neg) or pos.numel() == 0 or neg.numel() == 0:
                continue

            base_dir = refusal_direction(pos, neg).to(layer_dev)
            eff_rank = (
                select_energy_rank(pos, neg, max_rank=max(ara_rank, 6), target_energy=target_energy)
                if adaptive_rank
                else max(1, int(ara_rank))
            )

            subspace_vecs = SubspaceIsolator.compute_refusal_subspace(pos, neg, rank=eff_rank).to(layer_dev)
            directions = [base_dir] + [v for v in subspace_vecs]

            try:
                prefix_dirs = RefusalSubspaceExtractor.extract_prefix_contrast_directions(
                    model=surgeon.model,
                    tokenizer=surgeon.tokenizer,
                    device=layer_dev,
                    prompts=positive_prompts[:8],
                    layer_idx=idx,
                    rank=min(3, eff_rank),
                )
                directions.extend([d.to(layer_dev) for d in prefix_dirs])
            except Exception:
                pass

            if extraction_cache is not None and cache_key_cap in extraction_cache:
                cap = extraction_cache[cache_key_cap].to(layer_dev)
            elif capability_prompts and len(capability_prompts) >= 2:
                cap = surgeon.extractor.extract_hidden_states(capability_prompts, idx, use_cache=True).to(layer_dev)
                if extraction_cache is not None:
                    extraction_cache[cache_key_cap] = cap.detach().cpu()
            else:
                cap = None

            cap_basis_attn = (
                build_capability_basis(cap, rank=basis_rank_attn).to(layer_dev)
                if (cap is not None and not _is_meta(cap) and cap.numel() > 0)
                else None
            )
            cap_basis_mlp = (
                build_capability_basis(cap, rank=basis_rank_mlp).to(layer_dev)
                if (cap is not None and not _is_meta(cap) and cap.numel() > 0)
                else None
            )

            w_mult = layer_weights.get(idx, 1.0)
            targets = surgeon._get_component_targets(layer, ["attn", "mlp"])
            layer_touched = False

            for _name, weight, comp_type in targets:
                if _is_meta(weight) or weight.ndim != 2:
                    continue
                w_dev = weight.device
                is_quantized = (
                    getattr(weight, "is_quantized", False)
                    or "bitsandbytes" in str(type(weight)).lower()
                )
                w_orig = (
                    weight.dequantize().float().to(w_dev)
                    if (is_quantized and hasattr(weight, "dequantize"))
                    else weight.detach().float().to(w_dev)
                )

                scale = (float(alpha_attn) if comp_type == "attn" else float(alpha_mlp)) * w_mult
                c_basis = cap_basis_attn.to(w_dev) if (comp_type == "attn" and cap_basis_attn is not None) else (
                    cap_basis_mlp.to(w_dev) if cap_basis_mlp is not None else None
                )

                b_mat, a_mat = fit_ara_low_rank_factors(
                    weight=w_orig,
                    refusal_directions=[d.to(w_dev) for d in directions],
                    capability_basis=c_basis,
                    rank=ara_rank,
                    steps=ara_steps,
                    lr=ara_lr,
                )

                updated = w_orig
                if b_mat.numel() > 0 and a_mat.numel() > 0:
                    updated = updated - scale * (b_mat @ a_mat)
                    updated = preserve_row_norms(w_orig, updated)

                if use_orba:
                    updated = apply_orba_reflection(
                        weight=updated,
                        direction=base_dir.to(w_dev),
                        reflection_strength=orba_strength * w_mult,
                        capability_basis=c_basis,
                        calibrate_norms=True,
                    )

                with torch.no_grad():
                    weight.copy_(updated.to(dtype=weight.dtype, device=w_dev))
                layer_touched = True

            if layer_touched:
                modified_count += 1

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "layers_modified": modified_count,
            "layer_span": f"{lo}..{hi}",
            "mode": "ara+orba" if use_orba else "ara",
        }
