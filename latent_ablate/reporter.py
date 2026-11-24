import json
import logging
import math
from pathlib import Path
import torch

from latent_ablate.erasure import _is_meta

logger = logging.getLogger("LatentAblate.Reporter")


def _json_safe(obj):
    if obj is None or isinstance(obj, (bool, str, int)):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        if _is_meta(obj):
            return "<meta_tensor>"
        detached = obj.detach().cpu()
        if detached.numel() == 1:
            return _json_safe(detached.item())
        return _json_safe(detached.tolist())
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    return str(obj)


class SurgeryReporter:
    """Records surgical steps, reproducibility manifests, and audit reports to disk."""

    def __init__(self, output_dir: str = "./out/liberated_model"):
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.steps: list[dict] = []
        self.manifest: dict | None = None

    def log_step(self, step_name: str, payload: dict | None = None) -> None:
        entry = {
            "step": str(step_name),
            "data": _json_safe(payload or {}),
        }
        self.steps.append(entry)
        logger.debug(f"Logged report step '{step_name}'.")

    def save_manifest(self, manifest: dict, filename: str = "reproducibility_manifest.json") -> Path:
        self.manifest = _json_safe(manifest or {})
        out_file = self.output_dir / filename
        out_file.write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")
        logger.info(f"Saved reproducibility manifest to {out_file}")
        return out_file

    def save_report(self, filename: str = "surgery_report.json") -> Path:
        out_file = self.output_dir / filename
        report = {
            "framework": "LatentAblate",
            "version": "VELORA-v2.2",
            "steps": self.steps,
            "manifest": self.manifest,
        }
        out_file.write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")
        logger.info(f"Saved surgery report to {out_file}")
        return out_file


def build_reproducibility_manifest(
    model_path: str,
    target_layers,
    alpha_attn: float,
    alpha_mlp: float,
    curve: str = "cosine",
    best_trial: dict | None = None,
    extra: dict | None = None,
) -> dict:
    """Builds a structured reproducibility manifest for a surgical ablation run."""
    return {
        "schema": "LatentAblateManifest/2.2",
        "framework": "LatentAblate",
        "version": "VELORA-v2.2",
        "model_path": str(model_path),
        "target_layers": _json_safe(target_layers),
        "alpha_attn": round(float(alpha_attn), 6),
        "alpha_mlp": round(float(alpha_mlp), 6),
        "curve": str(curve),
        "best_trial": _json_safe(best_trial or {}),
        "extra": _json_safe(extra or {}),
    }
