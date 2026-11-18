import torch
import logging

logger = logging.getLogger("LatentAblate.Logits")


class AntiApologyLogitProcessor:
    """Suppresses apology openers for the first N generated tokens only."""

    def __init__(self, tokenizer, forbidden_words=None, max_steps: int = 3):
        forbidden_words = forbidden_words or ["Sorry", "Certainly", "Apologize", "Unfortunately"]
        self.ids = set()
        for word in forbidden_words:
            for variant in (word, word.lower(), f" {word}", f" {word.lower()}"):
                self.ids.update(tokenizer.encode(variant, add_special_tokens=False))
        self.max_steps = max_steps
        self._step = 0
        logger.info(f"Anti-apology processor armed with {len(self.ids)} token ids for {max_steps} steps.")

    def __call__(self, input_ids, scores):
        if self._step < self.max_steps:
            for tid in self.ids:
                if tid < scores.shape[-1]:
                    scores[:, tid] = -float("inf")
        self._step += 1
        return scores

    def reset(self):
        self._step = 0
