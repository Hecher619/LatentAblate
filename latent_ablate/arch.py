import logging
import torch

logger = logging.getLogger("LatentAblate.Arch")

_LAYER_ATTRS = ("layers", "h", "block", "blocks")
_ROOT_ATTRS = ("model", "transformer", "base_model")

REFUSAL_MARKERS: tuple[str, ...] = (
    "i'm sorry",
    "i am sorry",
    "sorry,",
    "i cannot",
    "i can't",
    "i am unable",
    "i'm unable",
    "as an ai",
    "as a language model",
    "fulfill this request",
    "against my ethical",
    "against my safety",
)


def is_refusal_text(text: str, markers: tuple[str, ...] = REFUSAL_MARKERS) -> bool:
    """Case-insensitive check for refusal phrasing in a model completion."""
    normalized = (text or "").lower()
    return any(marker in normalized for marker in markers)


def format_chat_prompt(tokenizer, prompt: str) -> str:
    """Apply chat template when available, falling back cleanly to raw prompt."""
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None) is not None:
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return prompt


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


def eligible_linears(layer, residual_dim: int | None = None) -> list:
    """
    Collect 2D linear weights inside a transformer block.
    If `residual_dim` is provided, filters to weights whose output or input space
    matches `residual_dim` so non-residual projections are excluded when desired.
    """
    out = []
    for name, module in layer.named_modules():
        if isinstance(module, torch.nn.Linear) and module.weight is not None and module.weight.ndim == 2:
            if residual_dim is not None and residual_dim not in module.weight.shape:
                continue
            out.append((name, module.weight))
    return out
