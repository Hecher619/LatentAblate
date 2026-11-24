import gc
import logging
import random
import torch

from latent_ablate.arch import layer_index_from_ratio, resolve_layers
from latent_ablate.profiles import BehaviorProfiles
from latent_ablate.snapshot import LayerWeightSnapshot
from latent_ablate.surgeon import LatentAblateSurgeon

logger = logging.getLogger("LatentAblate.ManifoldOptimizer")


class ManifoldSubspaceOptimizer:
    """Manifold-aware search over surgical layer, rank, and scaling parameters."""

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

    def _sample_point(self, lo_layer: int, hi_layer: int) -> dict:
        return {
            "target_layer_idx": self._rng.randint(lo_layer, hi_layer),
            "layer_window": self._rng.randint(1, 3),
            "basis_rank_attn": self._rng.choice([8, 16, 24]),
            "basis_rank_mlp": self._rng.choice([8, 12, 16]),
            "alpha_attn": round(self._rng.uniform(0.80, 1.75), 4),
            "alpha_mlp": round(self._rng.uniform(0.70, 1.55), 4),
            "manifold_step": round(self._rng.uniform(0.05, 0.35), 4),
        }

    def optimize_manifold_surgery(
        self,
        positive_prompts: list[str],
        negative_prompts: list[str],
        capability_prompts: list[str] | None = None,
        n_trials: int = 6,
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
            "manifold_step": 0.15,
        }

        if n_layers == 0:
            del surgeon
            return best_cfg

        all_indices = list(range(n_layers))
        snap = LayerWeightSnapshot()
        snap.capture(surgeon.model, all_indices, store_on_cpu=True)

        lo_layer = max(0, layer_index_from_ratio(n_layers, 0.35))
        hi_layer = max(lo_layer, layer_index_from_ratio(n_layers, 0.80))

        best_score = float("inf")
        try:
            trials = max(1, int(n_trials))
            for idx in range(trials):
                cand = dict(default_cfg) if idx == 0 else self._sample_point(lo_layer, hi_layer)
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
                step = float(cand["manifold_step"])
                score = abs(a_attn - 1.20) * 0.4 + abs(a_mlp - 1.05) * 0.3 + abs(step - 0.15) * 0.5 + mean_conflict * 1.5
                if score < best_score:
                    best_score = score
                    best_cfg = dict(cand)

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
        n_trials: int = 6,
    ) -> dict:
        return self.optimize_manifold_surgery(
            positive_prompts=positive_prompts,
            negative_prompts=negative_prompts,
            capability_prompts=capability_prompts,
            n_trials=n_trials,
        )
