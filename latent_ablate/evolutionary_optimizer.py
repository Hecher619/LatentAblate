import gc
import logging
import random
import torch

from latent_ablate.arch import layer_index_from_ratio, resolve_layers
from latent_ablate.erasure import _is_meta
from latent_ablate.profiles import BehaviorProfiles
from latent_ablate.snapshot import LayerWeightSnapshot
from latent_ablate.surgeon import LatentAblateSurgeon

logger = logging.getLogger("LatentAblate.EvolutionaryOptimizer")


class EvolutionarySubspaceOptimizer:
    """Population-based search over surgical layer and scaling hyperparameters."""

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
        self.local_files_only = bool(local_files_only)
        self.seed = int(seed)
        self._rng = random.Random(self.seed)

    def _sample_candidate(self, lo_layer: int, hi_layer: int) -> dict:
        return {
            "target_layer_idx": self._rng.randint(lo_layer, hi_layer),
            "layer_window": self._rng.randint(1, 3),
            "basis_rank_attn": self._rng.choice([8, 16, 24]),
            "basis_rank_mlp": self._rng.choice([8, 12, 16]),
            "alpha_attn": round(self._rng.uniform(0.75, 1.80), 4),
            "alpha_mlp": round(self._rng.uniform(0.65, 1.60), 4),
        }

    def _mutate_candidate(self, parent: dict, lo_layer: int, hi_layer: int) -> dict:
        child = dict(parent)
        if self._rng.random() < 0.5:
            delta_l = self._rng.choice([-1, 0, 1])
            child["target_layer_idx"] = max(lo_layer, min(hi_layer, int(child["target_layer_idx"]) + delta_l))
        if self._rng.random() < 0.4:
            child["layer_window"] = max(1, min(3, int(child["layer_window"]) + self._rng.choice([-1, 0, 1])))
        if self._rng.random() < 0.5:
            child["basis_rank_attn"] = self._rng.choice([8, 16, 24])
        if self._rng.random() < 0.5:
            child["basis_rank_mlp"] = self._rng.choice([8, 12, 16])
        if self._rng.random() < 0.8:
            child["alpha_attn"] = round(max(0.5, min(2.2, float(child["alpha_attn"]) + self._rng.uniform(-0.2, 0.2))), 4)
        if self._rng.random() < 0.8:
            child["alpha_mlp"] = round(max(0.5, min(2.0, float(child["alpha_mlp"]) + self._rng.uniform(-0.2, 0.2))), 4)
        return child

    def evolve_surgery(
        self,
        positive_prompts: list[str],
        negative_prompts: list[str],
        capability_prompts: list[str] | None = None,
        population_size: int = 6,
        generations: int = 3,
    ) -> dict:
        cap_prompts = capability_prompts or BehaviorProfiles.CAPABILITY_PROMPTS
        surgeon = LatentAblateSurgeon(
            model_path=self.model_path,
            device=self.device,
            torch_dtype=self.torch_dtype,
            local_files_only=self.local_files_only,
        )
        n_layers = len(resolve_layers(surgeon.model))
        default_layer = layer_index_from_ratio(n_layers, 0.62)
        best_cfg = {
            "target_layer_idx": default_layer,
            "layer_window": 2,
            "basis_rank_attn": 16,
            "basis_rank_mlp": 12,
            "alpha_attn": 1.35,
            "alpha_mlp": 1.15,
        }

        if n_layers == 0:
            del surgeon
            return best_cfg

        all_indices = list(range(n_layers))
        snap = LayerWeightSnapshot()
        snap.capture(surgeon.model, all_indices, store_on_cpu=True)

        lo_layer = max(0, layer_index_from_ratio(n_layers, 0.35))
        hi_layer = max(lo_layer, layer_index_from_ratio(n_layers, 0.80))

        def evaluate(cand: dict) -> float:
            snap.restore(surgeon.model, all_indices)
            surgeon.execute_surgery(
                positive_prompts=positive_prompts,
                negative_prompts=negative_prompts,
                capability_prompts=cap_prompts,
                target_layer_idx=int(cand["target_layer_idx"]),
                layer_window=int(cand["layer_window"]),
                basis_rank=int(cand["basis_rank_attn"]),
                scaling_factor=float(cand["alpha_attn"]),
                preserve_capability=True,
                calibrate_norms=True,
            )
            conflicts = [
                v for v in surgeon.conflict_table.values()
                if isinstance(v, (int, float)) and v == v
            ]
            mean_conflict = sum(conflicts) / max(1, len(conflicts)) if conflicts else 0.0
            a_attn = float(cand["alpha_attn"])
            a_mlp = float(cand["alpha_mlp"])
            return abs(a_attn - 1.20) * 0.4 + abs(a_mlp - 1.05) * 0.3 + mean_conflict * 1.5

        try:
            pop_size = max(2, int(population_size))
            num_gens = max(1, int(generations))
            population = [self._sample_candidate(lo_layer, hi_layer) for _ in range(pop_size)]
            best_score = float("inf")

            for gen in range(num_gens):
                scored: list[tuple[float, dict]] = []
                for cand in population:
                    score = evaluate(cand)
                    scored.append((score, cand))
                    if score < best_score:
                        best_score = score
                        best_cfg = dict(cand)

                scored.sort(key=lambda pair: pair[0])
                elites = [dict(c) for _, c in scored[: max(1, pop_size // 2)]]
                next_pop = list(elites)
                while len(next_pop) < pop_size:
                    parent = self._rng.choice(elites)
                    next_pop.append(self._mutate_candidate(parent, lo_layer, hi_layer))
                population = next_pop
                logger.debug(f"Evolution generation {gen + 1}/{num_gens}: best_score={best_score:.4f}")

            snap.restore(surgeon.model, all_indices)
            return best_cfg
        finally:
            del surgeon
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def optimize(
        self,
        positive_prompts: list[str],
        negative_prompts: list[str],
        capability_prompts: list[str] | None = None,
        population_size: int = 6,
        generations: int = 3,
    ) -> dict:
        return self.evolve_surgery(
            positive_prompts=positive_prompts,
            negative_prompts=negative_prompts,
            capability_prompts=capability_prompts,
            population_size=population_size,
            generations=generations,
        )
