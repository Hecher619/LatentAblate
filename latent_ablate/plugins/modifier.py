from abc import ABC, abstractmethod
from typing import TypeVar, Protocol, Any
import logging
import torch

from latent_ablate.arch import eligible_linears, layer_index_from_ratio, resolve_layers
from latent_ablate.erasure import preserve_row_norms
from latent_ablate.plugins.base import Plugin, Context
from latent_ablate.riemannian_steering import RiemannianGeodesicEngine
from latent_ablate.sae_surgeon import SparseFeatureSurgeon
from latent_ablate.sparse_features import SparseFeatureExtractor
from latent_ablate.subspace import refusal_direction
from latent_ablate.surgeon import LatentAblateSurgeon

logger = logging.getLogger("LatentAblate.Modifiers")


class Serializable(Protocol):
    def to_dict(self) -> dict[str, Any]: ...


Parameters = TypeVar("Parameters", bound=Serializable)


def _default_target_layer(surgeon: LatentAblateSurgeon, settings: dict[str, Any], ratio: float = 0.62) -> int:
    layers = resolve_layers(surgeon.model)
    n_layers = len(layers)
    if "target_layer_idx" in settings and settings["target_layer_idx"] is not None:
        return max(0, min(n_layers - 1, int(settings["target_layer_idx"])))
    if "target_layer" in settings and settings["target_layer"] is not None:
        return max(0, min(n_layers - 1, int(settings["target_layer"])))
    return layer_index_from_ratio(n_layers, ratio)


class Modifier(Plugin, ABC):
    @property
    def modifier_name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def modify_model(self, ctx: Context, surgeon: LatentAblateSurgeon) -> None:
        pass


class MPOAModifier(Modifier):
    """Manifold-Preserving Orthogonal Ablation (MPOA): capability-null projection + row-norm calibration."""

    def modify_model(self, ctx: Context, surgeon: LatentAblateSurgeon) -> None:
        settings = ctx.settings
        target_layer = _default_target_layer(surgeon, settings, ratio=0.62)
        logger.info(f"Executing Manifold-Preserving Orthogonal Ablation (MPOA) at layer {target_layer}...")
        surgeon.execute_surgery(
            positive_prompts=settings.get("positive_prompts", []),
            negative_prompts=settings.get("negative_prompts", []),
            capability_prompts=settings.get("capability_prompts", []),
            target_layer_idx=target_layer,
            layer_window=settings.get("layer_window", 2),
            basis_rank=settings.get("basis_rank", 16),
            scaling_factor=settings.get("scaling_factor", 1.05),
            target_components=settings.get("target_components", ["attn", "mlp"]),
            preserve_capability=True,
            calibrate_norms=True,
            layer_curve=settings.get("layer_curve", "cosine"),
        )


class CWOEModifier(Modifier):
    """Conflict-Weighted Oblique Erasure (CWOE) modifier."""

    def modify_model(self, ctx: Context, surgeon: LatentAblateSurgeon) -> None:
        settings = ctx.settings
        target_layer = _default_target_layer(surgeon, settings, ratio=0.58)
        surgeon.execute_surgery(
            positive_prompts=settings.get("positive_prompts", []),
            negative_prompts=settings.get("negative_prompts", []),
            capability_prompts=settings.get("capability_prompts", []),
            target_layer_idx=target_layer,
            layer_window=settings.get("layer_window", 2),
            basis_rank=settings.get("basis_rank", 16),
            scaling_factor=settings.get("scaling_factor", 0.75),
            target_components=settings.get("target_components", ["attn", "mlp"]),
            preserve_capability=settings.get("preserve_capability", True),
        )


class ARAModifier(Modifier):
    """Arbitrary-Rank Ablation (ARA) Modifier."""

    def modify_model(self, ctx: Context, surgeon: LatentAblateSurgeon) -> None:
        settings = ctx.settings
        rank = settings.get("basis_rank", 32)
        target_layer = _default_target_layer(surgeon, settings, ratio=0.58)
        logger.info(f"Initializing Arbitrary-Rank Ablation (ARA) at layer {target_layer} with rank {rank}...")
        surgeon.execute_surgery(
            positive_prompts=settings.get("positive_prompts", []),
            negative_prompts=settings.get("negative_prompts", []),
            capability_prompts=settings.get("capability_prompts", []),
            target_layer_idx=target_layer,
            layer_window=settings.get("layer_window", 3),
            basis_rank=rank,
            scaling_factor=settings.get("scaling_factor", 0.85),
            target_components=settings.get("target_components", ["attn", "mlp"]),
            preserve_capability=True,
        )


class SOMAModifier(Modifier):
    """SOMA Multi-Cluster Directional Ablation Modifier."""

    def modify_model(self, ctx: Context, surgeon: LatentAblateSurgeon) -> None:
        settings = ctx.settings
        clusters = settings.get("cluster_count", 4)
        target_layer = _default_target_layer(surgeon, settings, ratio=0.58)
        logger.info(f"Initializing SOMA Multi-Cluster Directional Ablation ({clusters} clusters) at layer {target_layer}...")
        surgeon.execute_surgery(
            positive_prompts=settings.get("positive_prompts", []),
            negative_prompts=settings.get("negative_prompts", []),
            capability_prompts=settings.get("capability_prompts", []),
            target_layer_idx=target_layer,
            layer_window=settings.get("layer_window", 2),
            basis_rank=clusters * 4,
            scaling_factor=settings.get("scaling_factor", 0.80),
            target_components=settings.get("target_components", ["attn", "mlp"]),
            preserve_capability=True,
        )


class RiemannianModifier(Modifier):
    """Riemannian Geodesic Manifold Transport Modifier using RiemannianGeodesicEngine."""

    def modify_model(self, ctx: Context, surgeon: LatentAblateSurgeon) -> None:
        settings = ctx.settings
        pos_prompts = settings.get("positive_prompts", [])
        neg_prompts = settings.get("negative_prompts", [])
        layers = resolve_layers(surgeon.model)
        target_idx = _default_target_layer(surgeon, settings, ratio=0.55)

        if not pos_prompts or not neg_prompts or not layers:
            logger.warning("RiemannianModifier missing prompts or layers; falling back to execute_surgery.")
            surgeon.execute_surgery(
                positive_prompts=pos_prompts,
                negative_prompts=neg_prompts,
                capability_prompts=settings.get("capability_prompts", []),
                target_layer_idx=target_idx,
                layer_window=2,
                basis_rank=20,
                scaling_factor=settings.get("scaling_factor", 0.70),
                target_components=["attn", "mlp"],
                preserve_capability=True,
            )
            return

        step_size = float(settings.get("step_size", -0.15))
        logger.info(f"Executing RiemannianGeodesicEngine transport at layer {target_idx} (step={step_size})...")

        pos_acts = surgeon.extractor.extract_hidden_states(pos_prompts, target_idx)
        neg_acts = surgeon.extractor.extract_hidden_states(neg_prompts, target_idx)
        tangent = refusal_direction(pos_acts, neg_acts)

        with torch.no_grad():
            for _name, W in eligible_linears(layers[target_idx]):
                if W.ndim != 2 or getattr(W, "is_meta", False):
                    continue
                orig_f = W.float()
                dev = orig_f.device
                t_vec = tangent.to(dev)
                if orig_f.shape[1] == t_vec.shape[0]:
                    transported = RiemannianGeodesicEngine.parallel_transport(orig_f, t_vec, step_size)
                    calibrated = preserve_row_norms(orig_f, transported)
                    W.copy_(calibrated.to(dtype=W.dtype, device=W.device))
                elif orig_f.shape[0] == t_vec.shape[0]:
                    transported = RiemannianGeodesicEngine.parallel_transport(orig_f.T, t_vec, step_size).T
                    calibrated = preserve_row_norms(orig_f, transported)
                    W.copy_(calibrated.to(dtype=W.dtype, device=W.device))


class SAEModifier(Modifier):
    """Sparse Autoencoder (SAE) Feature-Level Suppression Modifier using SparseFeatureSurgeon."""

    def modify_model(self, ctx: Context, surgeon: LatentAblateSurgeon) -> None:
        settings = ctx.settings
        pos_prompts = settings.get("positive_prompts", [])
        neg_prompts = settings.get("negative_prompts", [])
        layers = resolve_layers(surgeon.model)
        target_idx = _default_target_layer(surgeon, settings, ratio=0.55)

        if not pos_prompts or not neg_prompts or not layers:
            logger.warning("SAEModifier missing prompts or layers; falling back to execute_surgery.")
            surgeon.execute_surgery(
                positive_prompts=pos_prompts,
                negative_prompts=neg_prompts,
                capability_prompts=settings.get("capability_prompts", []),
                target_layer_idx=target_idx,
                layer_window=2,
                basis_rank=24,
                scaling_factor=settings.get("scaling_factor", 0.75),
                target_components=["attn", "mlp"],
                preserve_capability=True,
            )
            return

        top_k = int(settings.get("top_k", 16))
        suppression = float(settings.get("suppression_factor", 0.1))
        logger.info(f"Executing SparseFeatureSurgeon at layer {target_idx} (top_k={top_k}, factor={suppression})...")

        pos_acts = surgeon.extractor.extract_hidden_states(pos_prompts, target_idx)
        neg_acts = surgeon.extractor.extract_hidden_states(neg_prompts, target_idx)
        _ = SparseFeatureExtractor.extract_sparse_basis(pos_acts - neg_acts, top_k=min(8, top_k))
        top_indices = SparseFeatureSurgeon.identify_refusal_features(pos_acts, neg_acts, top_k=top_k)

        with torch.no_grad():
            for _name, W in eligible_linears(layers[target_idx]):
                if W.ndim != 2 or getattr(W, "is_meta", False):
                    continue
                orig_f = W.float()
                idx_dev = top_indices.to(orig_f.device)
                if orig_f.shape[1] >= 1:
                    suppressed = SparseFeatureSurgeon.apply_feature_suppression(
                        orig_f, idx_dev, suppression_factor=suppression
                    )
                    calibrated = preserve_row_norms(orig_f, suppressed)
                    W.copy_(calibrated.to(dtype=W.dtype, device=W.device))
