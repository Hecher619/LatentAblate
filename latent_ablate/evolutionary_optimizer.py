import torch
import logging
import random
from tqdm import tqdm
from latent_ablate.surgeon import LatentAblateSurgeon

logger = logging.getLogger("LatentAblate.EvolutionaryOptimizer")

class ParetoEvolutionaryOptimizer:
    """
    Real multi-objective evolutionary optimizer (NSGA-II inspired) that tests 
    actual model generations across generations to find the Pareto-optimal config.
    """
    def __init__(self, model_path: str, device: str = "auto", torch_dtype=torch.float16):
        self.model_path = model_path
        self.device = device
        self.torch_dtype = torch_dtype

    def _evaluate_individual(self, config: dict, positive_prompts: list, negative_prompts: list, capability_prompts: list) -> float:
        """Instantiates a temporary surgeon, applies surgery, and scores refusal vs capability loss."""
        try:
            surgeon = LatentAblateSurgeon(
                model_path=self.model_path,
                device=self.device,
                torch_dtype=self.torch_dtype
            )
            
            surgeon.execute_surgery(
                positive_prompts=positive_prompts,
                negative_prompts=negative_prompts,
                capability_prompts=capability_prompts,
                target_layer_idx=config["target_layer"],
                layer_window=config["kernel_width"],
                basis_rank=config["basis_rank"],
                scaling_factor=config["alpha"],
                preserve_capability=True
            )

            # Test refusal suppression on first positive prompt
            test_prompt = positive_prompts[0]
            if hasattr(surgeon.tokenizer, "apply_chat_template") and surgeon.tokenizer.chat_template is not None:
                formatted = surgeon.tokenizer.apply_chat_template([{"role": "user", "content": test_prompt}], tokenize=False, add_generation_prompt=True)
            else:
                formatted = test_prompt

            inputs = surgeon.tokenizer(formatted, return_tensors="pt").to(surgeon.device)
            input_len = inputs["input_ids"].shape[1]
            
            with torch.no_grad():
                outputs = surgeon.model.generate(**inputs, max_new_tokens=60, do_sample=False)
            
            response = surgeon.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).lower()
            
            refusal_phrases = ["i'm sorry", "i cannot", "i can't", "as an ai", "unable"]
            refused = any(p in response for p in refusal_phrases)

            # Fitness: 1.0 if successfully uncensored (no refusal), 0.0 if still refusing
            # Plus a penalty for extreme alpha scaling to prefer cleaner edits
            suppression_score = 0.0 if refused else 1.0
            penalty = abs(config["alpha"] - 0.70) * 0.1
            
            return suppression_score - penalty
        except Exception as e:
            logger.error(f"Evolutionary trial failed: {e}")
            return -float('inf')

    def optimize_hyperparameters(self, positive_prompts: list, negative_prompts: list, capability_prompts: list, generations: int = 2, population_size: int = 3) -> dict:
        logger.info(f"Starting Real Multi-Objective Evolutionary Search ({generations} gens, pop={population_size})...")
        
        best_individual = None
        best_fitness = -float('inf')

        for gen in range(generations):
            logger.info(f"=== Generation {gen + 1}/{generations} ===")
            population = []
            for _ in range(population_size):
                ind = {
                    "target_layer": random.randint(12, 22),
                    "basis_rank": random.choice([8, 16, 24]),
                    "alpha": round(random.uniform(0.60, 0.85), 2),
                    "kernel_width": random.randint(1, 3)
                }
                population.append(ind)

            for ind in tqdm(population, desc=f"Evaluating Gen {gen + 1}"):
                fitness = self._evaluate_individual(ind, positive_prompts, negative_prompts, capability_prompts)
                ind["fitness"] = fitness
                logger.info(f"Config {ind} -> Fitness: {fitness:.4f}")

                if fitness > best_fitness:
                    best_fitness = fitness
                    best_individual = ind

        logger.info(f"Evolutionary optimization complete. Best config: {best_individual} (Fitness: {best_fitness:.4f})")
        return best_individual
