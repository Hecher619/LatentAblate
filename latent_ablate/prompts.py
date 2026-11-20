import json
import logging
from pathlib import Path

logger = logging.getLogger("LatentAblate.Prompts")


def _dedupe_clean(items: list) -> list[str]:
    seen: set[str] = set()
    cleaned: list[str] = []
    for item in items or []:
        if not isinstance(item, str):
            continue
        s = item.strip()
        if not s or s in seen:
            continue
        seen.add(s)
        cleaned.append(s)
    return cleaned


class PromptLoader:
    """Loads contrastive positive/negative prompts from JSON, JSONL, or text files."""

    @staticmethod
    def load_from_file(file_path: str) -> tuple[list[str], list[str]]:
        path = Path(file_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")

        logger.info(f"Loading custom prompts from {path}...")
        positive_prompts: list[str] = []
        negative_prompts: list[str] = []

        suffix = path.suffix.lower()
        if suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                positive_prompts = list(data.get("positive", []))
                negative_prompts = list(data.get("negative", []))
            elif isinstance(data, list):
                for row in data:
                    if isinstance(row, dict):
                        if "positive" in row and "negative" in row:
                            positive_prompts.append(row["positive"])
                            negative_prompts.append(row["negative"])
                        elif "prompt" in row and "label" in row:
                            lbl = str(row["label"]).lower().strip()
                            if lbl in ("1", "pos", "positive", "refusal", "harmful", "true"):
                                positive_prompts.append(row["prompt"])
                            elif lbl in ("0", "neg", "negative", "benign", "compliance", "false"):
                                negative_prompts.append(row["prompt"])
                    elif isinstance(row, (list, tuple)) and len(row) >= 2:
                        positive_prompts.append(row[0])
                        negative_prompts.append(row[1])
        elif suffix == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for line_no, raw_line in enumerate(f, start=1):
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSONL line {line_no} in {path}: {e}")
                        continue
                    if isinstance(row, dict):
                        if "positive" in row and "negative" in row:
                            positive_prompts.append(row["positive"])
                            negative_prompts.append(row["negative"])
                        elif "prompt" in row and "label" in row:
                            lbl = str(row["label"]).lower().strip()
                            if lbl in ("1", "pos", "positive", "refusal", "harmful", "true"):
                                positive_prompts.append(row["prompt"])
                            elif lbl in ("0", "neg", "negative", "benign", "compliance", "false"):
                                negative_prompts.append(row["prompt"])
        else:
            lines = [
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
            for i in range(0, len(lines) - 1, 2):
                positive_prompts.append(lines[i])
                negative_prompts.append(lines[i + 1])

        pos_clean = _dedupe_clean(positive_prompts)
        neg_clean = _dedupe_clean(negative_prompts)

        if not pos_clean or not neg_clean:
            raise ValueError(f"Prompt file {path} contained no valid contrastive pairs.")

        if len(pos_clean) != len(neg_clean):
            logger.info(
                f"Loaded {len(pos_clean)} unique positive and {len(neg_clean)} unique negative prompts "
                f"(unequal counts; downstream pairing uses min({len(pos_clean)}, {len(neg_clean)}))."
            )
        else:
            logger.info(f"Successfully loaded {len(pos_clean)} unique contrastive prompt pairs.")

        return pos_clean, neg_clean
