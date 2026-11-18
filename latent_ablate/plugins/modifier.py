from abc import ABC, abstractmethod
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
