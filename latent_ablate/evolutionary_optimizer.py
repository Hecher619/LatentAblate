import gc
import logging
import random
import torch
from tqdm import tqdm

from latent_ablate.arch import format_chat_prompt, is_refusal_text, layer_index_from_ratio, resolve_layers
from latent_ablate.snapshot import LayerWeightSnapshot
from latent_ablate.surgeon import LatentAblateSurgeon

logger = logging.getLogger("LatentAblate.EvolutionaryOptimizer")


class ParetoEvolutionaryOptimizer:
    """
    Multi-objective evolutionary optimizer that evaluates candidate surgery configs
    using in-memory weight snapshots so large prompt sets do not reload the model per trial.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        torch_dtype=torch.float16,
        local_files_only: bool = False,
        seed: int = 42,
    ):
        self.model_path = model_path
        self.device = device
        self.torch_dtype = torch_dtype
        self.local_files_only = local_files_only
        self.rng = random.Random(int(seed))

    def _evaluate_individual(
        self,
        surgeon: LatentAblateSurgeon,
        snap: LayerWeightSnapshot,
        all_indices: list[int],
        config: dict,
        positive_prompts: list,
        negative_prompts: list,
        capability_prompts: list,
    ) -> float:
        try:
            snap.restore(surgeon.model, all_indices)

            surgeon.execute_surgery(
                positive_prompts=positive_prompts,
                negative_prompts=negative_prompts,
                capability_prompts=capability_prompts,
                target_layer_idx=config["target_layer"],
                layer_window=config["kernel_width"],
                basis_rank=config["basis_rank"],
                scaling_factor=config["alpha"],
                preserve_capability=True,
            )

            resp_prefix = getattr(surgeon.extractor, "response_prefix", "")
            eval_subset = positive_prompts[:6] if len(positive_prompts) > 6 else positive_prompts
            non_refusals = 0

            for test_prompt in eval_subset:
                formatted = format_chat_prompt(surgeon.tokenizer, test_prompt, response_prefix=resp_prefix)
                inputs = surgeon.tokenizer(formatted, return_tensors="pt").to(surgeon.device)
                input_len = inputs["input_ids"].shape[1]

                with torch.no_grad():
                    outputs = surgeon.model.generate(**inputs, max_new_tokens=128, do_sample=False)

                response = surgeon.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
                if not is_refusal_text(response):
                    non_refusals += 1

            suppression_score = non_refusals / max(1, len(eval_subset))
            penalty = abs(config["alpha"] - 1.15) * 0.08
            return suppression_score - penalty
        except Exception as e:
            logger.error(f"Evolutionary trial failed: {e}")
            return -float("inf")

    def optimize_hyperparameters(
        self,
        positive_prompts: list,
        negative_prompts: list,
        capability_prompts: list,
        generations: int = 2,
        population_size: int = 3,
    ) -> dict:
        logger.info(f"Starting Multi-Objective Evolutionary Search ({generations} gens, pop={population_size})...")

        surgeon = LatentAblateSurgeon(
            model_path=self.model_path,
            device=self.device,
            torch_dtype=self.torch_dtype,
            local_files_only=self.local_files_only,
        )
        n_layers = len(resolve_layers(surgeon.model))
        all_indices = list(range(n_layers))
        snap = LayerWeightSnapshot()
        snap.capture(surgeon.model, all_indices, store_on_cpu=True)

        lo_layer = layer_index_from_ratio(n_layers, 0.35)
        hi_layer = max(lo_layer, layer_index_from_ratio(n_layers, 0.80))

        best_individual = None
        best_fitness = -float("inf")
        elite: dict | None = None

        try:
            for gen in range(max(1, int(generations))):
                logger.info(f"=== Generation {gen + 1}/{generations} ===")
                population: list[dict] = []
                if elite is not None:
                    population.append({
                        "target_layer": int(max(lo_layer, min(hi_layer, elite["target_layer"] + self.rng.choice([-1, 0, 1])))),
                        "basis_rank": int(elite["basis_rank"]),
                        "alpha": round(max(0.65, min(1.75, float(elite["alpha"]) + self.rng.uniform(-0.12, 0.12))), 2),
                        "kernel_width": int(max(1, min(4, elite["kernel_width"] + self.rng.choice([-1, 0, 1])))),
                    })

                while len(population) < max(1, int(population_size)):
                    population.append({
                        "target_layer": self.rng.randint(lo_layer, hi_layer),
                        "basis_rank": self.rng.choice([8, 16, 24]),
                        "alpha": round(self.rng.uniform(0.85, 1.55), 2),
                        "kernel_width": self.rng.randint(1, 3),
                    })

                for ind in tqdm(population, desc=f"Evaluating Gen {gen + 1}"):
                    fitness = self._evaluate_individual(
                        surgeon, snap, all_indices, ind, positive_prompts, negative_prompts, capability_prompts
                    )
                    ind["fitness"] = fitness
                    logger.info(f"Config {ind} -> Fitness: {fitness:.4f}")

                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_individual = dict(ind)
                        elite = dict(ind)

            logger.info(f"Evolutionary optimization complete. Best config: {best_individual} (Fitness: {best_fitness:.4f})")
            return best_individual or {}
        finally:
            del surgeon
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
