import json
from pathlib import Path
import logging
import torch

logger = logging.getLogger("LatentAblate.Reporter")

class SurgeryReporter:
    def __init__(self, output_dir: str = "./out", *args, **kwargs):
        self.output_dir = Path(output_dir) if output_dir else Path("./out")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report_data = {
            "framework": "LatentAblate",
            "version": "5.0.0-Hyperion",
            "method": "Conflict-Weighted Oblique Erasure (CWOE)",
            "surgery_steps": []
        }

    def log_step(self, step_name: str, details: dict):
        logger.info(f"Logging audit entry: {step_name}")
        entry = {
            "step": step_name,
            "details": details
        }
        self.report_data["surgery_steps"].append(entry)

    def save_report(self):
        def clean_tensors(obj):
            if isinstance(obj, dict):
                return {k: clean_tensors(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_tensors(v) for v in obj]
            elif torch.is_tensor(obj):
                return obj.tolist() if obj.numel() > 1 else obj.item()
            return obj

        report_path = self.output_dir / "surgery_report.json"
        cleaned_data = clean_tensors(self.report_data)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(cleaned_data, f, indent=4)
        logger.info(f"Audit report successfully saved to {report_path}")

Reporter = SurgeryReporter


def build_reproducibility_manifest(
    model_path: str,
    target_layers: list[int],
    alpha_attn: float,
    alpha_mlp: float,
    curve: str = "cosine",
    best_trial: dict | None = None,
) -> dict:
    """Create a structured reproducibility record for a surgical ablation run."""
    return {
        "schema_version": "1.0",
        "tool": "LatentAblate",
        "model_path": model_path,
        "target_layers": list(target_layers),
        "alpha_attn": float(alpha_attn),
        "alpha_mlp": float(alpha_mlp),
        "layer_curve": curve,
        "best_trial": best_trial or {},
    }
