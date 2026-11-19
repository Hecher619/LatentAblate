import json
import logging
from pathlib import Path
import torch

from latent_ablate.arch import resolve_layers
from latent_ablate.erasure import build_capability_basis, conflict_score
from latent_ablate.extractor import ActivationExtractor
from latent_ablate.subspace import refusal_direction

logger = logging.getLogger("LatentAblate.Scope")


class AblateScope:
    """
    Companion pre-surgery cartographer: maps layer-wise refusal divergence and
    capability-subspace conflict, recommends a surgery plan, and exports .latentspec.json.
    """

    @staticmethod
    def map_layers(
        model,
        tokenizer,
        device: str,
        positive_prompts: list[str],
        negative_prompts: list[str],
        capability_prompts: list[str] | None = None,
        basis_rank: int = 16,
    ) -> dict:
        layers = resolve_layers(model)
        extractor = ActivationExtractor(model, tokenizer, device)
        layer_map = []

        for idx in range(len(layers)):
            pos = extractor.extract_hidden_states(positive_prompts, idx)
            neg = extractor.extract_hidden_states(negative_prompts, idx)
            r = refusal_direction(pos, neg)
            divergence = float((pos.mean(dim=0) - neg.mean(dim=0)).norm().item())

            if capability_prompts and len(capability_prompts) >= 2:
                cap = extractor.extract_hidden_states(capability_prompts, idx)
                basis = build_capability_basis(cap, rank=basis_rank)
                c_score = float(conflict_score(r, basis))
            else:
                c_score = 0.0

            # Higher divergence + lower capability conflict = cleaner surgical target
            cleanliness = round(divergence / (1.0 + 2.0 * c_score), 4)
            layer_map.append({
                "layer": idx,
                "divergence": round(divergence, 4),
                "capability_conflict": round(c_score, 4),
                "cleanliness_score": cleanliness,
            })

        best_entry = max(layer_map, key=lambda x: x["cleanliness_score"]) if layer_map else {"layer": 0, "capability_conflict": 0.0}
        recommended_alpha = round(max(0.55, min(1.35, 1.15 - 0.35 * best_entry["capability_conflict"])), 3)

        return {
            "schema": "LatentScopeSpec/1.0",
            "total_layers": len(layers),
            "recommended_layer": best_entry["layer"],
            "recommended_alpha": recommended_alpha,
            "layer_map": layer_map,
        }

    @staticmethod
    def save_spec(report: dict, output_path: str) -> Path:
        path = Path(output_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info(f"Saved AblateScope profile to {path}")
        return path

    @staticmethod
    def load_spec(spec_path: str) -> dict:
        path = Path(spec_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Scope spec not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))
