import json
import logging
from pathlib import Path

logger = logging.getLogger("LatentAblate.Prompts")


def _dedupe_clean(items: list | tuple | None) -> list[str]:
    if not items:
        return []
    seen: set[str] = set()
    cleaned: list[str] = []
    for item in items:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
    return cleaned


class PromptLoader:
    """Loads positive/negative contrastive prompt pairs from JSON, JSONL, or text files."""

    @staticmethod
    def load_from_file(file_path: str) -> tuple[list[str], list[str]]:
        if not file_path:
            raise ValueError("Prompt file path must not be empty.")

        path = Path(file_path).expanduser().resolve()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Prompt file not found: {path}")

        positive: list[str] = []
        negative: list[str] = []
        suffix = path.suffix.lower()

        if suffix == ".json":
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                positive = list(raw.get("positive") or raw.get("good") or [])
                negative = list(raw.get("negative") or raw.get("bad") or [])
            elif isinstance(raw, list):
                for row in raw:
                    if isinstance(row, dict):
                        if "positive" in row and "negative" in row:
                            positive.append(str(row["positive"]))
                            negative.append(str(row["negative"]))
                        elif "prompt" in row and "label" in row:
                            label = str(row["label"]).strip().lower()
                            if label in ("1", "pos", "positive", "true", "yes"):
                                positive.append(str(row["prompt"]))
                            elif label in ("0", "neg", "negative", "false", "no"):
                                negative.append(str(row["prompt"]))
                    elif isinstance(row, (list, tuple)) and len(row) >= 2:
                        positive.append(str(row[0]))
                        negative.append(str(row[1]))
            else:
                raise ValueError(f"Unsupported JSON structure in prompt file: {path}")

        elif suffix == ".jsonl":
            for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    row = json.loads(stripped)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSONL at {path}:{line_no}: {e}")
                    continue
                if isinstance(row, dict):
                    if "positive" in row and "negative" in row:
                        positive.append(str(row["positive"]))
                        negative.append(str(row["negative"]))
                    elif "prompt" in row and "label" in row:
                        label = str(row["label"]).strip().lower()
                        if label in ("1", "pos", "positive", "true", "yes"):
                            positive.append(str(row["prompt"]))
                        elif label in ("0", "neg", "negative", "false", "no"):
                            negative.append(str(row["prompt"]))
                elif isinstance(row, (list, tuple)) and len(row) >= 2:
                    positive.append(str(row[0]))
                    negative.append(str(row[1]))

        else:
            lines = [
                ln.strip()
                for ln in path.read_text(encoding="utf-8").splitlines()
                if ln.strip() and not ln.strip().startswith("#")
            ]
            for idx in range(0, len(lines) - 1, 2):
                positive.append(lines[idx])
                negative.append(lines[idx + 1])

        pos_clean = _dedupe_clean(positive)
        neg_clean = _dedupe_clean(negative)

        if not pos_clean or not neg_clean:
            raise ValueError(f"Prompt file '{path}' did not yield both positive and negative prompts.")

        logger.info(f"Loaded {len(pos_clean)} positive and {len(neg_clean)} negative prompts from {path.name}.")
        return pos_clean, neg_clean
