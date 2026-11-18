import json
from pathlib import Path
import logging

logger = logging.getLogger("LatentAblate.Reporter")

class SurgeryReporter:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report_data = {
            "framework": "LatentAblate",
            "version": "3.0.0",
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
        report_path = self.output_dir / "surgery_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(self.report_data, f, indent=4)
        logger.info(f"Audit report successfully saved to {report_path}")
