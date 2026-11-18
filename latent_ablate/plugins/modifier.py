from abc import abstractmethod
from dataclasses import dataclass
from typing import TypeVar, Protocol, Any
from pydantic import BaseModel
from latent_ablate.plugins.base import Plugin, Context
from latent_ablate.surgeon import LatentAblateSurgeon

class Serializable(Protocol):
    def to_dict(self) -> dict[str, Any]: ...

Parameters = TypeVar("Parameters", bound=Serializable)

class Modifier(Plugin, ABC):
    """Abstract base class for model modification plugins (e.g. CWOE, ARA)."""
    @property
    def modifier_name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def modify_model(self, ctx: Context, surgeon: LatentAblateSurgeon) -> None:
        pass

class CWOEModifier(Modifier):
    """Default Conflict-Weighted Oblique Erasure modifier plugin."""
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
