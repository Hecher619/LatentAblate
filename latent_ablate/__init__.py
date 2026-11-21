from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from latent_ablate.surgeon import LatentAblateSurgeon

__all__ = ["LatentAblateSurgeon"]


def __getattr__(name: str):
    if name == "LatentAblateSurgeon":
        from latent_ablate.surgeon import LatentAblateSurgeon

        return LatentAblateSurgeon
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
