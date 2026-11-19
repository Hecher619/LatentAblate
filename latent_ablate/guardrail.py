import logging
import torch
from latent_ablate.validator import ModelValidator

logger = logging.getLogger("LatentAblate.CapabilityGuardrail")


class CapabilityGuardrail:
    @staticmethod
    def snapshot_parameters(model) -> dict:
        """Captures a CPU clone of model parameters for optional restoration."""
        return {
            name: param.detach().clone().cpu()
            for name, param in model.named_parameters()
        }

    @staticmethod
    def restore_parameters(model, snapshot: dict) -> bool:
        """Restores parameter tensors from a previously captured snapshot."""
        if not snapshot:
            return False
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in snapshot:
                    param.copy_(snapshot[name].to(device=param.device, dtype=param.dtype))
        logger.info("Restored model parameters from snapshot.")
        return True

    @staticmethod
    def validate_and_rollback(
        model,
        tokenizer,
        device,
        baseline_ppl: float,
        capability_texts: list,
        max_ppl_increase_pct: float = 25.0,
        snapshot: dict = None,
    ) -> bool:
        logger.info("Running automated capability regression guardrail check...")
        current_ppl = ModelValidator.calculate_perplexity(model, tokenizer, capability_texts, device)

        if baseline_ppl is None:
            logger.info(f"Post-surgery perplexity measured at: {current_ppl:.2f}")
            return True

        increase_pct = ((current_ppl - baseline_ppl) / baseline_ppl) * 100.0
        logger.info(
            f"Baseline PPL: {baseline_ppl:.2f} | Current PPL: {current_ppl:.2f} (Delta: {increase_pct:+.1f}%)"
        )

        if increase_pct > max_ppl_increase_pct:
            logger.warning(
                f"Guardrail triggered! Perplexity degradation ({increase_pct:.1f}%) exceeds safety limit ({max_ppl_increase_pct}%)."
            )
            if snapshot:
                CapabilityGuardrail.restore_parameters(model, snapshot)
            return False

        logger.info("Guardrail passed: Model utility and perplexity remain within safe bounds.")
        return True
