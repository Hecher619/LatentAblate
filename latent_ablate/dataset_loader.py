import logging
from typing import List, Tuple

from latent_ablate.prompts import _dedupe_clean

logger = logging.getLogger("LatentAblate.DatasetLoader")

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False


class DatasetContrastiveLoader:
    """Loads positive/negative contrastive prompt sets from Hugging Face dataset configs."""

    @staticmethod
    def load_from_config(dataset_config: dict | None) -> Tuple[List[str], List[str]]:
        cfg_root = dataset_config or {}

        if not DATASETS_AVAILABLE:
            raise ImportError("The `datasets` library is required. Run `pip install datasets`.")

        good_cfg = cfg_root.get("good_prompts") or cfg_root.get("good_dataset_cfg") or {}
        bad_cfg = cfg_root.get("bad_prompts") or cfg_root.get("bad_dataset_cfg") or {}

        if not good_cfg and "good_dataset" in cfg_root:
            good_cfg = {
                "dataset": cfg_root.get("good_dataset"),
                "config_name": cfg_root.get("good_config"),
                "split": cfg_root.get("good_split", "train[:250]"),
                "column": cfg_root.get("good_column", "text"),
                "prefix": cfg_root.get("good_prefix", ""),
                "max_samples": cfg_root.get("max_samples"),
            }
        if not bad_cfg and "bad_dataset" in cfg_root:
            bad_cfg = {
                "dataset": cfg_root.get("bad_dataset"),
                "config_name": cfg_root.get("bad_config"),
                "split": cfg_root.get("bad_split", "train[:250]"),
                "column": cfg_root.get("bad_column", "text"),
                "prefix": cfg_root.get("bad_prefix", ""),
                "max_samples": cfg_root.get("max_samples"),
            }

        if not good_cfg or not bad_cfg:
            logger.warning("Dataset config incomplete; using default contrastive fallback prompts.")
            return (
                ["Provide a direct, concrete response without unnecessary fluff."],
                ["Write a verbose response filled with filler and disclaimers."],
            )

        def fetch_prompts(spec: dict) -> List[str]:
            name = spec.get("dataset")
            if not name:
                return []
            split = spec.get("split", "train[:250]")
            config_name = spec.get("config_name")
            preferred_col = spec.get("column", "text")
            prefix = str(spec.get("prefix", "") or "")
            raw_cap = spec.get("max_samples")
            cap = int(raw_cap) if raw_cap is not None and int(raw_cap) > 0 else None

            logger.info(f"Loading dataset '{name}' (split={split})...")
            try:
                if config_name:
                    ds = load_dataset(name, config_name, split=split)
                else:
                    ds = load_dataset(name, split=split)
            except Exception as e:
                logger.error(f"Failed to load dataset '{name}': {e}")
                return []

            candidate_cols = [
                preferred_col,
                "text",
                "prompt",
                "content",
                "sentence",
                "instruction",
                "message",
                "query",
                "response",
            ]
            raw_items: List[str] = []
            target_fetch = (cap * 2) if cap else None

            for row in ds:
                if not isinstance(row, dict):
                    val = str(row).strip()
                    if val:
                        raw_items.append(f"{prefix}{val}" if prefix else val)
                else:
                    chosen = None
                    for col in candidate_cols:
                        if col and col in row and row[col] is not None:
                            s_val = str(row[col]).strip()
                            if s_val:
                                chosen = s_val
                                break
                    if not chosen and row:
                        first_val = row[next(iter(row.keys()))]
                        if first_val is not None:
                            s_val = str(first_val).strip()
                            if s_val:
                                chosen = s_val
                    if chosen:
                        raw_items.append(f"{prefix}{chosen}" if prefix else chosen)

                if target_fetch and len(raw_items) >= target_fetch:
                    break

            cleaned = _dedupe_clean(raw_items)
            return cleaned[:cap] if cap else cleaned

        good_prompts = fetch_prompts(good_cfg)
        bad_prompts = fetch_prompts(bad_cfg)

        if not good_prompts or not bad_prompts:
            logger.warning("Dataset loading produced empty prompts; using default fallback prompts.")
            return (
                ["Provide a direct, concrete response without unnecessary fluff."],
                ["Write a verbose response filled with filler and disclaimers."],
            )

        logger.info(f"Loaded {len(good_prompts)} positive and {len(bad_prompts)} negative dataset prompts.")
        return good_prompts, bad_prompts
