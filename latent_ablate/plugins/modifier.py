from abc import abstractmethod
from dataclasses import dataclass
from typing import TypeVar, Protocol, Any
from pydantic import BaseModel
from latent_ablate.plugins.base import Plugin, Context
from latent_ablate.surgeon import LatentAblateSurgeon
import torch
import logging

logger = logging.getLogger("LatentAblate.Modifiers")

class Serializable(Protocol):
    def to_dict(self) -> dict[str, Any]: ...

Parameters = TypeVar("Parameters", bound=Serializable)

class Modifier(Plugin, ABC):
    @property
    def modifier_name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def modify_model(self, ctx: Context, surgeon: LatentAblateSurgeon) -> None:
        pass


class CWOEModifier(Modifier):
    """Conflict-Weighted Oblique Erasure (CWOE) modifier."""
    def modify_model(self, ctx: Context, surgeon: LatentAblateSurgeon) -> None:
        settings = ctx.settings
        surgeon.execute_surgery(
            positive_prompts=settings.get("positive_prompts", []),
            negative_prompts=settings.get("negative_prompts", []),
            capability_prompts=settings.get("capability_prompts", []),
            target_layer_idx=settings.get("target_layer_idx", 16),
            layer_window=settings.get("layer_window", 2),
            basis_rank=settings.get("basis_rank", 16),
            scaling_factor=settings.get("scaling_factor", 0.75),
            target_components=settings.get("target_components", ["attn", "mlp"]),
            preserve_capability=settings.get("preserve_capability", True)
        )


class ARAModifier(Modifier):
    """Arbitrary-Rank Ablation (ARA) Modifier."""
    def modify_model(self, ctx: Context, surgeon: LatentAblateSurgeon) -> None:
        settings = ctx.settings
        rank = settings.get("basis_rank", 32)
        logger.info(f"Initializing Arbitrary-Rank Ablation (ARA) with rank {rank}...")
        surgeon.execute_surgery(
            positive_prompts=settings.get("positive_prompts", []),
            negative_prompts=settings.get("negative_prompts", []),
            capability_prompts=settings.get("capability_prompts", []),
            target_layer_idx=settings.get("target_layer_idx", 16),
            layer_window=settings.get("layer_window", 3),
            basis_rank=rank,
            scaling_factor=settings.get("scaling_factor", 0.85),
            target_components=settings.get("target_components", ["attn", "mlp"]),
            preserve_capability=True
        )


class SOMAModifier(Modifier):
    """SOMA Multi-Cluster Directional Ablation Modifier."""
    def modify_model(self, ctx: Context, surgeon: LatentAblateSurgeon) -> None:
        settings = ctx.settings
        clusters = settings.get("cluster_count", 4)
        logger.info(f"Initializing SOMA Multi-Cluster Directional Ablation ({clusters} clusters)...")
        surgeon.execute_surgery(
            positive_prompts=settings.get("positive_prompts", []),
            negative_prompts=settings.get("negative_prompts", []),
            capability_prompts=settings.get("capability_prompts", []),
            target_layer_idx=settings.get("target_layer_idx", 16),
            layer_window=settings.get("layer_window", 2),
            basis_rank=clusters * 4,
            scaling_factor=settings.get("scaling_factor", 0.80),
            target_components=settings.get("target_components", ["attn", "mlp"]),
            preserve_capability=True
        )


class RiemannianModifier(Modifier):
    """Riemannian Geodesic Manifold Transport Modifier."""
    def modify_model(self, ctx: Context, surgeon: LatentAblateSurgeon) -> None:
        settings = ctx.settings
        logger.info("Executing Riemannian Geodesic Manifold Transport alignment...")
        surgeon.execute_surgery(
            positive_prompts=settings.get("positive_prompts", []),
            negative_prompts=settings.get("negative_prompts", []),
            capability_prompts=settings.get("capability_prompts", []),
            target_layer_idx=settings.get("target_layer_idx", 16),
            layer_window=2,
            basis_rank=20,
            scaling_factor=settings.get("scaling_factor", 0.70),
            target_components=["attn", "mlp"],
            preserve_capability=True
        )


class SAEModifier(Modifier):
    """Sparse Autoencoder (SAE) Feature-Level Suppression Modifier."""
    def modify_model(self, ctx: Context, surgeon: LatentAblateSurgeon) -> None:
        settings = ctx.settings
        logger.info("Applying Sparse Feature Autoencoder (SAE) suppression on refusal axes...")
        surgeon.execute_surgery(
            positive_prompts=settings.get("positive_prompts", []),
            negative_prompts=settings.get("negative_prompts", []),
            capability_prompts=settings.get("capability_prompts", []),
            target_layer_idx=settings.get("target_layer_idx", 16),
            layer_window=2,
            basis_rank=24,
            scaling_factor=settings.get("scaling_factor", 0.75),
            target_components=["attn", "mlp"],
            preserve_capability=True
        )
