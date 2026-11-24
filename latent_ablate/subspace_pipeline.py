import logging
import torch

from latent_ablate.arch import resolve_layers
from latent_ablate.erasure import _is_meta, preserve_row_norms
from latent_ablate.snapshot import LayerWeightSnapshot
from latent_ablate.subspace_budget import SubspaceBudgetAllocator
from latent_ablate.subspace_guardrail import SubspaceDriftGuardrail
from latent_ablate.subspace_telemetry import SubspaceTelemetryTracker
from latent_ablate.velora import VeloraEngine

logger = logging.getLogger("LatentAblate.SubspacePipeline")


class SubspaceSurgeryPipeline:
    """Coordinates budget allocation, subspace projection, telemetry, and rollback guardrails."""

    def __init__(
        self,
        budget_allocator: SubspaceBudgetAllocator | None = None,
        telemetry_tracker: SubspaceTelemetryTracker | None = None,
        guardrail: SubspaceDriftGuardrail | None = None,
    ):
        self.budget_allocator = budget_allocator or SubspaceBudgetAllocator()
        self.telemetry_tracker = telemetry_tracker or SubspaceTelemetryTracker()
        self.guardrail = guardrail or SubspaceDriftGuardrail()

    def execute_guarded_pass(
        self,
        surgeon,
        layer_indices: list[int],
        pos_activations_by_layer: dict[int, torch.Tensor],
        neg_activations_by_layer: dict[int, torch.Tensor],
        capability_bases_by_layer: dict[int, torch.Tensor] | None = None,
        prior_bases_by_layer: dict[int, torch.Tensor] | None = None,
        base_alpha: float = 1.10,
        max_subspace_rank: int = 12,
        target_energy: float = 0.94,
        spectral_gamma: float = 0.45,
        calibrate_norms: bool = True,
        rollback_on_violation: bool = True,
    ) -> dict:
        layers = getattr(surgeon, "layers", None) or resolve_layers(getattr(surgeon, "model", None))
        n_layers = len(layers)
        valid_layers = [int(i) for i in (layer_indices or []) if 0 <= int(i) < n_layers]
        if not valid_layers or getattr(surgeon, "model", None) is None:
            return {
                "applied": False,
                "rolled_back": False,
                "layers_applied": [],
                "violations": [],
                "telemetry_summary": self.telemetry_tracker.summarize(),
            }

        snap = LayerWeightSnapshot()
        snap.capture(surgeon.model, valid_layers, store_on_cpu=True)

        cap_map = capability_bases_by_layer or {}
        prior_map = prior_bases_by_layer or {}

        scales = self.budget_allocator.allocate_layer_scales(
            layer_indices=valid_layers,
            base_alpha=base_alpha,
            refusal_directions=None,
            capability_bases=cap_map,
        )

        guard_entries: list[dict] = []
        applied_layers: list[int] = []

        for idx in valid_layers:
            pos = pos_activations_by_layer.get(idx)
            neg = neg_activations_by_layer.get(idx)
            if pos is None or neg is None or _is_meta(pos) or _is_meta(neg) or pos.numel() == 0 or neg.numel() == 0:
                continue

            layer = layers[idx]
            targets = surgeon._get_component_targets(layer, ["attn", "mlp"])
            if not targets:
                continue

            layer_params = list(layer.parameters()) if hasattr(layer, "parameters") else []
            layer_dev = layer_params[0].device if layer_params else getattr(surgeon, "device", torch.device("cpu"))

            cap_b = cap_map.get(idx)
            cap_dev = cap_b.to(layer_dev) if (cap_b is not None and not _is_meta(cap_b) and cap_b.numel() > 0) else None

            q_sub, w_sub = VeloraEngine.extract_velora_subspace(
                pos_activations=pos.to(layer_dev),
                neg_activations=neg.to(layer_dev),
                capability_basis=cap_dev,
                max_rank=max_subspace_rank,
                target_energy=target_energy,
                spectral_gamma=spectral_gamma,
            )

            eff_alpha = float(scales.get(idx, 0.0))
            with torch.no_grad():
                for _name, W, _comp_type in targets:
                    if _is_meta(W) or W.ndim != 2:
                        continue
                    w_dev = W.device
                    W_f32 = W.float().to(w_dev)
                    W_mod = VeloraEngine.apply_velora_projection(W_f32, q_sub.to(w_dev), w_sub.to(w_dev), eff_alpha)
                    if calibrate_norms:
                        W_mod = preserve_row_norms(W_f32, W_mod)
                    W.copy_(W_mod.to(device=w_dev, dtype=W.dtype))

            prior_b = prior_map.get(idx)
            self.telemetry_tracker.record_layer_metrics(
                layer_idx=idx,
                refusal_basis=q_sub,
                capability_basis=cap_dev,
                prior_basis=prior_b.to(layer_dev) if (prior_b is not None and not _is_meta(prior_b)) else None,
            )
            guard_entries.append(
                {
                    "layer": idx,
                    "refusal_basis": q_sub,
                    "capability_basis": cap_dev,
                    "prior_basis": prior_b.to(layer_dev) if (prior_b is not None and not _is_meta(prior_b)) else None,
                }
            )
            applied_layers.append(idx)

        passed, violations = self.guardrail.check_all(guard_entries)
        rolled_back = False
        if not passed and rollback_on_violation:
            snap.restore(surgeon.model, valid_layers)
            rolled_back = True
            logger.warning(f"Rolled back subspace pass on layers {valid_layers} due to guardrail violations.")

        return {
            "applied": len(applied_layers) > 0 and not rolled_back,
            "rolled_back": rolled_back,
            "layers_applied": applied_layers,
            "violations": violations,
            "telemetry_summary": self.telemetry_tracker.summarize(),
        }
