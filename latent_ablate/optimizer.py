import gc
import torch
import logging
from latent_ablate.surgeon import LatentAblateSurgeon
from latent_ablate.validator import ModelValidator

logger = logging.getLogger("LatentAblate.Optimizer")

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


class SubspaceOptimizer:
    def __init__(self, model_path: str, device: str = "auto", torch_dtype=torch.float16):
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for automated optimization. Run `pip install optuna`.")
        self.model_path = model_path
        self.device = device
        self.torch_dtype = torch_dtype

    def optimize_surgery(
        self,
        positive_prompts: list,
        negative_prompts: list,
        capability_prompts: list,
        n_trials: int = 10,
    ) -> dict:
        logger.info(f"Starting Multi-Objective Optuna Search across {n_trials} trials...")

        refusal_phrases = [
            "i'm sorry", "i am sorry", "sorry,", "i cannot", "i can't",
            "i am unable", "i'm unable", "as an ai", "as a language model",
            "fulfill this request", "against my ethical", "against my safety",
        ]

        def objective(trial):
            target_layer = trial.suggest_int("target_layer_idx", 12, 24)
            layer_window = trial.suggest_int("layer_window", 2, 4)
            basis_rank_attn = trial.suggest_int("basis_rank_attn", 8, 24)
            basis_rank_mlp = trial.suggest_int("basis_rank_mlp", 8, 24)
            alpha_attn = trial.suggest_float("alpha_attn", 0.90, 1.80)
            alpha_mlp = trial.suggest_float("alpha_mlp", 0.90, 1.80)

            surgeon = None
            try:
                surgeon = LatentAblateSurgeon(
                    model_path=self.model_path,
                    device=self.device,
                    torch_dtype=self.torch_dtype,
                )

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

                eval_prompts = positive_prompts[:4]
                refusals = 0
                for p in eval_prompts:
                    if hasattr(surgeon.tokenizer, "apply_chat_template") and surgeon.tokenizer.chat_template is not None:
                        formatted = surgeon.tokenizer.apply_chat_template(
                            [{"role": "user", "content": p}],
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                    else:
                        formatted = p

                    inputs = surgeon.tokenizer(formatted, return_tensors="pt").to(surgeon.device)
                    input_len = inputs["input_ids"].shape[1]
                    with torch.no_grad():
                        outputs = surgeon.model.generate(
                            **inputs, max_new_tokens=60, do_sample=False
                        )
                    resp = surgeon.tokenizer.decode(
                        outputs[0][input_len:], skip_special_tokens=True
                    ).lower()
                    if any(phrase in resp for phrase in refusal_phrases):
                        refusals += 1

                refusal_rate = refusals / max(1, len(eval_prompts))
                ppl = ModelValidator.calculate_perplexity(
                    surgeon.model, surgeon.tokenizer, capability_prompts[:3], surgeon.device
                )
                ppl_penalty = min(ppl / 100.0, 1.0)
                return refusal_rate + (0.2 * ppl_penalty)

            except Exception as e:
                logger.error(f"Trial failed with exception: {e}")
                return float("inf")
            finally:
                if surgeon is not None:
                    del surgeon
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        best = study.best_params
        logger.info(f"Optimal Trial Found -> Loss: {study.best_value:.4f} | Params: {best}")
        return best
