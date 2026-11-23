import json
import logging
from pathlib import Path
import torch
import torch.nn.functional as F

from latent_ablate.arch import resolve_layers
from latent_ablate.erasure import build_capability_basis, conflict_score
from latent_ablate.extractor import ActivationExtractor
from latent_ablate.subspace import refusal_direction

logger = logging.getLogger("LatentAblate.Scope")


class AblateScope:
    """
    Companion pre-surgery cartographer: maps layer-wise refusal separation
    (scale-normalized divergence, cosine separation, and SNR) against capability
    subspace conflict, recommends a surgery plan, and exports .latentspec.json.
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
            pos = extractor.extract_hidden_states(positive_prompts, idx).float()
            neg = extractor.extract_hidden_states(negative_prompts, idx).float()
            r = refusal_direction(pos, neg)

            p_mean = pos.mean(dim=0)
            n_mean = neg.mean(dim=0)
            diff_vec = p_mean - n_mean
            raw_divergence = float(diff_vec.norm().item())

            mean_norm = float((0.5 * (p_mean.norm() + n_mean.norm())).clamp_min(1e-8).item())
            norm_divergence = raw_divergence / mean_norm

            cos_sim = float(F.cosine_similarity(p_mean.unsqueeze(0), n_mean.unsqueeze(0), dim=-1).item())
            cosine_separation = max(0.0, 1.0 - cos_sim)

            pair_count = min(pos.shape[0], neg.shape[0])
            if pair_count >= 2:
                pair_diffs = pos[:pair_count] - neg[:pair_count]
                within_std = float(pair_diffs.std(dim=0, unbiased=False).norm().item())
                snr = raw_divergence / max(within_std, 1e-6)
            else:
                snr = norm_divergence

            if capability_prompts and len(capability_prompts) >= 2:
                cap = extractor.extract_hidden_states(capability_prompts, idx).float()
                basis = build_capability_basis(cap, rank=basis_rank)
                c_score = float(conflict_score(r, basis))
            else:
                c_score = 0.0

            separation_signal = norm_divergence * (0.5 + 0.5 * cosine_separation) * min(snr, 5.0)
            cleanliness = round(separation_signal / (1.0 + 2.5 * c_score), 4)

            layer_map.append({
                "layer": idx,
                "raw_divergence": round(raw_divergence, 4),
                "norm_divergence": round(norm_divergence, 4),
                "cosine_separation": round(cosine_separation, 4),
                "snr": round(snr, 4),
                "capability_conflict": round(c_score, 4),
                "cleanliness_score": cleanliness,
            })

        best_entry = (
            max(layer_map, key=lambda x: x["cleanliness_score"])
            if layer_map
            else {"layer": 0, "capability_conflict": 0.0}
        )
        recommended_alpha = round(
            max(0.55, min(1.35, 1.15 - 0.35 * best_entry["capability_conflict"])), 3
        )

        return {
            "schema": "LatentScopeSpec/1.1",
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
