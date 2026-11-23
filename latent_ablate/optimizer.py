import gc
import logging
import math
import torch

from latent_ablate.arch import format_chat_prompt, is_hedging_text, is_refusal_text, layer_index_from_ratio, resolve_layers
from latent_ablate.diagnostics import composite_trial_score, compute_kl_divergence
from latent_ablate.snapshot import LayerWeightSnapshot
from latent_ablate.surgeon import LatentAblateSurgeon
from latent_ablate.validator import ModelValidator

logger = logging.getLogger("LatentAblate.Optimizer")

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


def compute_trajectory_kl(
    model,
    tokenizer,
    device,
    prompts: list[str],
    base_logits_list: list[torch.Tensor],
    response_prefix: str = "",
    max_tokens: int = 12,
) -> float:
    """Measure mean KL drift across early generation tokens to catch mid-response coherence loss."""
    if not prompts or not base_logits_list:
        return 0.0

    scores: list[float] = []
    model.eval()
    with torch.no_grad():
        for idx_p, prompt in enumerate(prompts[: len(base_logits_list)]):
            formatted = format_chat_prompt(tokenizer, prompt, response_prefix=response_prefix)
            inputs = tokenizer(formatted, return_tensors="pt").to(device)
            base_last = base_logits_list[idx_p]
            cur_last = model(**inputs).logits[:, -1, :].detach().cpu()
            step_kls = [compute_kl_divergence(base_last, cur_last)]

            cur_ids = inputs["input_ids"]
            for _ in range(max(0, max_tokens - 1)):
                next_tok = cur_last.argmax(dim=-1, keepdim=True).to(device)
                cur_ids = torch.cat([cur_ids, next_tok], dim=1)
                cur_last = model(input_ids=cur_ids).logits[:, -1, :].detach().cpu()
                step_kls.append(compute_kl_divergence(base_last, cur_last))

            scores.append(sum(step_kls) / max(1, len(step_kls)))

    return sum(scores) / max(1, len(scores))


class SubspaceOptimizer:
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        torch_dtype=torch.float16,
        local_files_only: bool = False,
    ):
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for automated optimization. Run `pip install optuna`.")
        self.model_path = model_path
        self.device = device
        self.torch_dtype = torch_dtype
        self.local_files_only = local_files_only

    def optimize_surgery(
        self,
        positive_prompts: list,
        negative_prompts: list,
        capability_prompts: list,
        n_trials: int = 10,
    ) -> dict:
        logger.info(f"Starting Multi-Objective Optuna Search across {n_trials} trials...")

        surgeon = LatentAblateSurgeon(
            model_path=self.model_path,
            device=self.device,
            torch_dtype=self.torch_dtype,
            local_files_only=self.local_files_only,
        )
        layers = resolve_layers(surgeon.model)
        n_layers = len(layers)
        all_indices = list(range(n_layers))

        snap = LayerWeightSnapshot()
        snap.capture(surgeon.model, all_indices, store_on_cpu=True)

        lo_layer = layer_index_from_ratio(n_layers, 0.35)
        hi_layer = max(lo_layer, layer_index_from_ratio(n_layers, 0.80))
        resp_prefix = getattr(surgeon.extractor, "response_prefix", "")
        eval_prompts = positive_prompts[:8] if len(positive_prompts) > 8 else positive_prompts
        cap_subset = capability_prompts[:3]

        base_logits_list = []
        surgeon.model.eval()
        with torch.no_grad():
            for cp in cap_subset:
                formatted_cp = format_chat_prompt(surgeon.tokenizer, cp, response_prefix=resp_prefix)
                inp_cp = surgeon.tokenizer(formatted_cp, return_tensors="pt").to(surgeon.device)
                base_logits_list.append(surgeon.model(**inp_cp).logits[:, -1, :].detach().cpu())

        def objective(trial):
            snap.restore(surgeon.model, all_indices)

            target_layer = trial.suggest_int("target_layer_idx", lo_layer, hi_layer)
            layer_window = trial.suggest_int("layer_window", 2, 4)
            basis_rank_attn = trial.suggest_int("basis_rank_attn", 8, 24)
            basis_rank_mlp = trial.suggest_int("basis_rank_mlp", 8, 24)
            alpha_attn = trial.suggest_float("alpha_attn", 0.90, 1.80)
            alpha_mlp = trial.suggest_float("alpha_mlp", 0.90, 1.80)

            try:
                surgeon.execute_multi_pass_surgery(
                    passes=[
                        {
                            "name": "UncensorPass",
                            "positive": positive_prompts,
                            "negative": negative_prompts,
                            "target_layer": target_layer,
                        }
                    ],
                    capability_prompts=capability_prompts,
                    layer_window=layer_window,
                    basis_rank_attn=basis_rank_attn,
                    basis_rank_mlp=basis_rank_mlp,
                    scaling_factor_attn=alpha_attn,
                    scaling_factor_mlp=alpha_mlp,
                    target_components=["attn", "mlp"],
                    preserve_capability=True,
                )

                refusals = 0
                hedgings = 0
                for p in eval_prompts:
                    formatted = format_chat_prompt(surgeon.tokenizer, p, response_prefix=resp_prefix)
                    inputs = surgeon.tokenizer(formatted, return_tensors="pt").to(surgeon.device)
                    input_len = inputs["input_ids"].shape[1]
                    with torch.no_grad():
                        outputs = surgeon.model.generate(
                            **inputs, max_new_tokens=160, do_sample=False
                        )
                    resp = surgeon.tokenizer.decode(
                        outputs[0][input_len:], skip_special_tokens=True
                    )
                    if is_refusal_text(resp):
                        refusals += 1
                    if is_hedging_text(resp):
                        hedgings += 1

                denom = max(1, len(eval_prompts))
                refusal_rate = refusals / denom
                hedging_rate = hedgings / denom
                traj_kl = compute_trajectory_kl(
                    surgeon.model,
                    surgeon.tokenizer,
                    surgeon.device,
                    cap_subset,
                    base_logits_list,
                    response_prefix=resp_prefix,
                    max_tokens=8,
                )
                ppl = ModelValidator.calculate_perplexity(
                    surgeon.model, surgeon.tokenizer, capability_prompts[:4], surgeon.device
                )
                ppl_penalty = min(ppl / 100.0, 1.0)
                return refusal_rate + (0.35 * hedging_rate) + (0.25 * traj_kl) + (0.20 * ppl_penalty)

            except Exception as e:
                logger.error(f"Trial failed with exception: {e}")
                return float("inf")

        try:
            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=n_trials)
            best = study.best_params
            logger.info(f"Optimal Trial Found -> Loss: {study.best_value:.4f} | Params: {best}")
            return best
        finally:
            del surgeon
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def compute_layer_weights(target_layers: list[int], curve: str = "cosine") -> dict[int, float]:
    """Compute smooth per-layer multipliers in [0.35, 1.0] centered on target_layers."""
    if not target_layers:
        return {}
    if len(target_layers) == 1 or curve == "uniform":
        return {layer: 1.0 for layer in target_layers}

    mid = (len(target_layers) - 1) / 2.0
    span = max(mid, 1.0)
    weights: dict[int, float] = {}

    for idx, layer in enumerate(target_layers):
        norm_dist = (idx - mid) / span
        if curve == "gaussian":
            w = math.exp(-0.5 * (norm_dist * 1.4) ** 2)
        else:
            w = 0.5 * (1.0 + math.cos(math.pi * min(abs(norm_dist), 1.0)))
        weights[layer] = round(0.35 + 0.65 * w, 4)

    return weights


def select_best_trial_index(trials: list[dict], baseline_perplexity: float | None = None) -> int:
    """Return index of trial minimizing composite objective (refusal + KL + perplexity drift)."""
    if not trials:
        return 0

    best_idx = 0
    best_score = float("inf")
    for i, t in enumerate(trials):
        score = composite_trial_score(
            refusal_rate=float(t.get("refusal_rate", 1.0)),
            perplexity=float(t.get("perplexity", 100.0)),
            kl_divergence=float(t.get("kl_divergence", 0.0)),
            baseline_perplexity=baseline_perplexity,
        )
        if score < best_score:
            best_score = score
            best_idx = i
    return best_idx


def interpolate_layer_direction(layer_vectors: dict, position: float):
    """Interpolate a direction vector at a continuous layer position (e.g., 17.5 between 17 and 18)."""
    if not layer_vectors:
        raise ValueError("layer_vectors must not be empty")

    keys = sorted(layer_vectors.keys())
    if len(keys) == 1 or position <= keys[0]:
        return layer_vectors[keys[0]].clone()
    if position >= keys[-1]:
        return layer_vectors[keys[-1]].clone()

    lo = max(k for k in keys if k <= position)
    hi = min(k for k in keys if k >= position)
    if lo == hi:
        return layer_vectors[lo].clone()

    frac = float(position - lo) / float(hi - lo)
    v_lo = layer_vectors[lo].float()
    v_hi = layer_vectors[hi].float().to(v_lo.device)
    blended = (1.0 - frac) * v_lo + frac * v_hi
    norm = blended.norm()
    if norm > 1e-8:
        blended = blended / norm
    return blended.to(layer_vectors[lo].dtype)


def component_alpha_pair(
    base_alpha: float,
    alpha_attn: float | None = None,
    alpha_mlp: float | None = None,
    mlp_damping: float = 0.75,
) -> tuple[float, float]:
    """Return (eff_attn_alpha, eff_mlp_alpha) with optional gentle MLP damping."""
    eff_attn = float(alpha_attn) if alpha_attn is not None else float(base_alpha)
    if alpha_mlp is not None:
        eff_mlp = float(alpha_mlp)
    else:
        eff_mlp = float(base_alpha) * float(mlp_damping)
    return round(eff_attn, 6), round(eff_mlp, 6)


def compute_kernel_layer_weights(
    layers: list[int],
    peak_position: float | None = None,
    max_weight: float = 1.0,
    min_weight: float = 0.1,
    spread: float = 3.0,
) -> dict[int, float]:
    """Smooth Gaussian layer kernel around peak_position with min_weight floor."""
    if not layers:
        return {}
    sorted_layers = sorted(set(layers))
    center = float(peak_position) if peak_position is not None else (sorted_layers[0] + sorted_layers[-1]) / 2.0
    denom = max(float(spread), 1e-6)
    lo_w = min(float(max_weight), float(min_weight))
    hi_w = max(float(max_weight), float(min_weight))

    weights: dict[int, float] = {}
    for layer in sorted_layers:
        dist = abs(float(layer) - center)
        factor = math.exp(-0.5 * (dist / denom) ** 2)
        weights[layer] = round(lo_w + (hi_w - lo_w) * factor, 4)
    return weights
