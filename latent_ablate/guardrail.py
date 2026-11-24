import logging
import math
import torch

from latent_ablate.erasure import _is_meta
from latent_ablate.validator import ModelValidator

logger = logging.getLogger("LatentAblate.Guardrail")


class CapabilityGuardrail:
    """Snapshots model parameters and rolls back if capability perplexity spikes past a threshold."""

    @staticmethod
    def snapshot_parameters(model) -> dict[str, torch.Tensor]:
        snap: dict[str, torch.Tensor] = {}
        if model is None:
            return snap
        for name, param in model.named_parameters():
            if param is None or _is_meta(param):
                continue
            cloned = param.detach().clone()
            if cloned.device.type != "cpu":
                cloned = cloned.cpu()
            snap[name] = cloned
        return snap

    @staticmethod
    def restore_parameters(model, snapshot: dict[str, torch.Tensor] | None) -> int:
        if model is None or not snapshot:
            return 0
        restored = 0
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param is None or _is_meta(param):
                    continue
                saved = snapshot.get(name)
                if saved is None or _is_meta(saved) or saved.shape != param.shape:
                    continue
                param.copy_(saved.to(device=param.device, dtype=param.dtype))
                restored += 1
        return restored

    @classmethod
    def validate_and_rollback(
        cls,
        model,
        tokenizer,
        device,
        baseline_ppl: float | None,
        capability_texts: list[str],
        max_ppl_increase_pct: float = 25.0,
        snapshot: dict[str, torch.Tensor] | None = None,
    ) -> bool:
        """
        Evaluates current capability perplexity against `baseline_ppl`.
        Rolls back to `snapshot` if relative increase exceeds `max_ppl_increase_pct`
        or if perplexity becomes non-finite. Returns True if within threshold.
        """
        if not capability_texts:
            return True

        current_ppl = ModelValidator.calculate_perplexity(model, tokenizer, capability_texts, device)
        if math.isnan(current_ppl) or math.isinf(current_ppl) or current_ppl <= 0.0:
            logger.warning(f"Guardrail triggered: invalid post-surgery perplexity ({current_ppl}).")
            if snapshot:
                count = cls.restore_parameters(model, snapshot)
                logger.warning(f"Restored {count} parameter tensor(s) from snapshot.")
            return False

        if baseline_ppl is None or math.isnan(baseline_ppl) or math.isinf(baseline_ppl) or baseline_ppl <= 0.0:
            logger.info(f"Guardrail check passed (post-surgery PPL={current_ppl:.2f}, no valid baseline).")
            return True

        pct_increase = ((current_ppl - baseline_ppl) / baseline_ppl) * 100.0
        allowed_pct = max(0.0, float(max_ppl_increase_pct))
        if pct_increase > allowed_pct:
            logger.warning(
                f"Guardrail rollback: PPL increased by {pct_increase:.2f}% "
                f"({baseline_ppl:.2f} -> {current_ppl:.2f} > {allowed_pct:.2f}% max)."
            )
            if snapshot:
                count = cls.restore_parameters(model, snapshot)
                logger.warning(f"Restored {count} parameter tensor(s) from snapshot.")
            return False

        logger.info(
            f"Guardrail passed: PPL {baseline_ppl:.2f} -> {current_ppl:.2f} "
            f"({pct_increase:+.2f}% vs {allowed_pct:.2f}% cap)."
        )
        return True
