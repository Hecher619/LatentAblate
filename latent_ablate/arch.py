import torch
import logging

logger = logging.getLogger("LatentAblate.Arch")

_LAYER_ATTRS = ("layers", "h", "block", "blocks")
_ROOT_ATTRS = ("model", "transformer", "base_model")


def resolve_layers(model) -> list:
    for attr in _ROOT_ATTRS:
        sub = getattr(model, attr, None)
        if sub is None:
            continue
        for sub_attr in _LAYER_ATTRS:
            if hasattr(sub, sub_attr):
                return getattr(sub, sub_attr)
    if hasattr(model, "layers"):
        return model.layers
    raise AttributeError("Could not resolve transformer layers for this architecture.")


def hidden_size(model) -> int:
    cfg = model.config
    for key in ("hidden_size", "n_embd", "d_model"):
        if hasattr(cfg, key):
            return getattr(cfg, key)
    raise AttributeError("Could not determine hidden size from model config.")


def eligible_linears(layer) -> list:
    """
    Linear modules whose OUTPUT space is the residual stream, i.e. W.shape[0] == hidden.
    """
    out = []
    for name, module in layer.named_modules():
        if isinstance(module, torch.nn.Linear) and module.weight is not None and module.weight.ndim == 2:
            out.append((name, module.weight))
    return out
