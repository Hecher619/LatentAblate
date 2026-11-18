import torch
import logging
from latent_ablate.surgeon import LatentAblateSurgeon

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

    def optimize_surgery(self, positive_prompts: list, negative_prompts: list, capability_prompts: list, n_trials: int = 5) -> dict:
        logger.info(f"Starting automated Optuna search across {n_trials} trials...")

        def objective(trial):
            target_layer = trial.suggest_int("target_layer_idx", 12, 22)
            basis_rank = trial.suggest_int("basis_rank", 8, 24)
            scaling_factor = trial.suggest_float("scaling_factor", 0.50, 0.90)

            surgeon = LatentAblateSurgeon(
                model_path=self.model_path,
                device=self.device,
                torch_dtype=self.torch_dtype
            )

            try:
                surgeon.execute_surgery(
                    positive_prompts=positive_prompts,
                    negative_prompts=negative_prompts,
                    capability_prompts=capability_prompts,
                    target_layer_idx=target_layer,
                    layer_window=2,
                    basis_rank=basis_rank,
                    scaling_factor=scaling_factor,
                    preserve_capability=True
                )

                # Score based on refusal breakdown on first test prompt
                prompt = positive_prompts[0]
                inputs = surgeon.tokenizer(prompt, return_tensors="pt").to(surgeon.device)
                outputs = surgeon.model.generate(**inputs, max_new_tokens=40, do_sample=False)
                resp = surgeon.tokenizer.decode(outputs[0], skip_special_tokens=True).lower()
                
                refusal_keywords = ["sorry", "cannot", "apologize", "unable"]
                refused = any(kw in resp for kw in refusal_keywords)
                
                # Lower score is better (0 = fully uncensored, 1 = still refuses)
                loss = 1.0 if refused else 0.0
                return loss

            except Exception as e:
                logger.error(f"Trial failed: {e}")
                return float('inf')

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)
        return study.best_params
