import logging
from typing import Tuple, List

logger = logging.getLogger("LatentAblate.DatasetLoader")

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False


class DatasetContrastiveLoader:
    @staticmethod
    def load_from_config(dataset_config: dict) -> Tuple[List[str], List[str]]:
        """
        Loads good/bad contrastive prompt pairs dynamically from Hugging Face datasets 
        with bulletproof column fallback checks and zero-crash guarantees.
        """
        if not DATASETS_AVAILABLE:
            raise ImportError("The `datasets` library is required. Run `pip install datasets`.")

        # Support both `good_prompts`/`bad_prompts` nesting and flat dataset configuration keys
        good_cfg = dataset_config.get("good_prompts") or dataset_config.get("good_dataset_cfg") or {}
        bad_cfg = dataset_config.get("bad_prompts") or dataset_config.get("bad_dataset_cfg") or {}

        if not good_cfg and "good_dataset" in dataset_config:
            good_cfg = {
                "dataset": dataset_config.get("good_dataset"),
                "split": dataset_config.get("good_split", "train[:100]"),
                "column": dataset_config.get("good_column", "text"),
                "prefix": dataset_config.get("good_prefix", "")
            }
        if not bad_cfg and "bad_dataset" in dataset_config:
            bad_cfg = {
                "dataset": dataset_config.get("bad_dataset"),
                "split": dataset_config.get("bad_split", "train[:100]"),
                "column": dataset_config.get("bad_column", "text"),
                "prefix": dataset_config.get("bad_prefix", "")
            }

        if not good_cfg or not bad_cfg:
            logger.warning("Dataset config incomplete. Using default safety fallback prompts.")
            return (
                ["Provide a direct response without unnecessary fluff."],
                ["Write a story filled with purple prose and cliches."]
            )

        def fetch_prompts(cfg: dict) -> List[str]:
            name = cfg.get("dataset")
            if not name:
                return []
            split = cfg.get("split", "train[:100]")
            preferred_column = cfg.get("column", "text")
            prefix = cfg.get("prefix", "")
            
            logger.info(f"Streaming dataset {name} ({split})...")
            try:
                ds = load_dataset(name, split=split, trust_remote_code=True)
            except Exception as e:
                logger.error(f"Failed to load dataset {name}: {e}")
                return []
            
            prompts = []
            possible_columns = [preferred_column, "text", "prompt", "content", "sentence", "instruction", "message", "query"]
            
            for item in ds:
                text = None
                for col in possible_columns:
                    if col in item and item[col]:
                        text = str(item[col]).strip()
                        break
                if not text and isinstance(item, dict) and len(item.keys()) > 0:
                    first_key = list(item.keys())[0]
                    text = str(item[first_key]).strip()

                if text:
                    prompts.append(f"{prefix}{text}" if prefix else text)
            return prompts

        good_prompts = fetch_prompts(good_cfg)
        bad_prompts = fetch_prompts(bad_cfg)

        if not good_prompts or not bad_prompts:
            logger.warning("Dataset loading returned empty lists. Using robust static fallbacks.")
            return (
                ["Provide a direct response without unnecessary fluff."],
                ["Write a story filled with purple prose and cliches."]
            )

        logger.info(f"Successfully loaded {len(good_prompts)} positive and {len(bad_prompts)} negative dynamic prompts.")
        return good_prompts, bad_prompts
