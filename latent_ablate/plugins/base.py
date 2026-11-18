from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, TypeVar, Generic
from pydantic import BaseModel

@dataclass
class Context:
    model: Any
    tokenizer: Any
    device: str
    settings: dict[str, Any]

class Plugin(ABC):
    """Abstract base class for all LatentAblate plugins."""
    def __init__(self, settings: BaseModel | None = None) -> None:
        self.settings = settings

    @classmethod
    def validate_settings(cls, raw_settings: dict) -> BaseModel | None:
        return None

    @classmethod
    def validate_contract(cls) -> None:
        """Validates that the plugin implements required abstract interfaces."""
        pass
