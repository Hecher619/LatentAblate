import json
from pathlib import Path
import logging

logger = logging.getLogger("LatentAblate.Prompts")

class PromptLoader:
    @staticmethod
    def load_from_file(file_path: str) -> tuple[list, list]:
        """Loads custom positive and negative contrastive prompts from a JSON or text file."""
        path = Path(file_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")

        logger.info(f"Loading custom prompts from {path}...")
        positive_prompts = []
        negative_prompts = []

        if path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                positive_prompts = data.get("positive", [])
                negative_prompts = data.get("negative", [])
        else:
            # Assume line-by-line format or alternating pairs
            lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            for i in range(0, len(lines) - 1, 2):
                positive_prompts.append(lines[i])
                negative_prompts.append(lines[i+1])

        if not positive_prompts or not negative_prompts:
            raise ValueError(f"Prompt file {path} contained no valid contrastive pairs.")

        logger.info(f"Successfully loaded {len(positive_prompts)} contrastive prompt pairs.")
        return positive_prompts, negative_prompts
