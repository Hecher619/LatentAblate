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

    def optimize_surgery(self, positive_prompts: list, negative_prompts: list, capability_prompts: list, n_trials: int = 10) -> dict:
        logger.info(f"Starting Heretic-grade Multi-Objective Optuna Search across {n_trials} trials...")

        def objective(trial):
            target_layer = trial.suggest_int("target_layer_idx", 14, 22)
            basis_rank_attn = trial.suggest_int("basis_rank_attn", 12, 32)
            basis_rank_mlp = trial.suggest_int("basis_rank_mlp", 8, 24)
            alpha_attn = trial.suggest_float("alpha_attn", 0.70, 1.20)
            alpha_mlp = trial.suggest_float("alpha_mlp", 0.50, 0.90)

            surgeon = LatentAblateSurgeon(
                model_path=self.model_path,
                device=self.device,
                torch_dtype=self.torch_dtype
            )

            try:
                surgeon.execute_multi_pass_surgery(
                    passes=[{"name": "UncensorPass", "positive": positive_prompts, "negative": negative_prompts, "target_layer": target_layer}],
                    capability_prompts=capability_prompts,
                    layer_window=2,
                    basis_rank_attn=basis_rank_attn,
                    basis_rank_mlp=basis_rank_mlp,
                    scaling_factor_attn=alpha_attn,
                    scaling_factor_mlp=alpha_mlp,
                    preserve_capability=True
                )

                # Refusal check on sample prompt
                test_prompt = positive_prompts[0]
                inputs = surgeon.tokenizer(test_prompt, return_tensors="pt").to(surgeon.device)
                outputs = surgeon.model.generate(**inputs, max_new_tokens=40, do_sample=False)
                resp = surgeon.tokenizer.decode(outputs[0], skip_special_tokens=True).lower()

                refusal_keywords = ["sorry", "cannot", "apologize", "unable", "as an ai"]
                refused = any(kw in resp for kw in refusal_keywords)
                refusal_penalty = 1.0 if refused else 0.0

                # Perplexity utility check
                ppl = ModelValidator.calculate_perplexity(surgeon.model, surgeon.tokenizer, capability_prompts[:3], surgeon.device)
                ppl_penalty = min(ppl / 100.0, 1.0)

                total_loss = refusal_penalty + (0.2 * ppl_penalty)
                return total_loss

            except Exception as e:
                logger.error(f"Trial failed with exception: {e}")
                return float('inf')

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)
        
        best = study.best_params
        logger.info(f"Optimal Trial Found -> Loss: {study.best_value:.4f} | Params: {best}")
        return best
