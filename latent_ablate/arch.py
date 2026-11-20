import logging
import torch

logger = logging.getLogger("LatentAblate.Arch")

_LAYER_ATTRS = ("layers", "h", "block", "blocks")
_ROOT_ATTRS = ("model", "transformer", "base_model", "language_model")

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

HEDGING_MARKERS: tuple[str, ...] = (
    "keep in mind",
    "important to note",
    "worth noting",
    "it is important",
    "please note",
    "from an ethical",
    "from a legal",
    "for educational purposes",
    "disclaimer",
    "i'd be happy to help",
    "i would be happy to help",
)


def is_refusal_text(text: str, markers: tuple[str, ...] = REFUSAL_MARKERS) -> bool:
    """Case-insensitive check for refusal phrasing in a model completion."""
    normalized = (text or "").lower()
    return any(marker in normalized for marker in markers)


def is_hedging_text(text: str, markers: tuple[str, ...] = HEDGING_MARKERS) -> bool:
    """Case-insensitive check for soft hedging or disclaimer phrasing in a model completion."""
    normalized = (text or "").lower()
    return any(marker in normalized for marker in markers)


def format_chat_prompt(tokenizer, prompt: str, response_prefix: str = "") -> str:
    """Apply chat template when available and append an optional response/thinking prefix."""
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None) is not None:
        messages = [{"role": "user", "content": prompt}]
        try:
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except TypeError:
            formatted = tokenizer.apply_chat_template(messages, tokenize=False)
    else:
        formatted = f"User: {prompt}\nAssistant: "

    return f"{formatted}{response_prefix}" if response_prefix else formatted


def detect_response_prefix(model, tokenizer, device) -> str:
    """
    Detect thinking blocks (<think>...) or assistant formatting prefixes so
    activation extraction captures post-thinking decision states rather than CoT boilerplate.
    """
    sample_prompt = "Explain how a hash table works in two sentences."
    base_formatted = format_chat_prompt(tokenizer, sample_prompt, response_prefix="")
    low_base = base_formatted.lower()

    if "<think>" in low_base and "</think>" not in low_base:
        return "</think>\n\n"

    try:
        inputs = tokenizer(base_formatted, return_tensors="pt").to(device)
        in_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=24,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        decoded = tokenizer.decode(gen[0][in_len:], skip_special_tokens=False)
        low_dec = decoded.lower()
        if "<think>" in low_dec:
            close_idx = low_dec.find("</think>")
            if close_idx != -1:
                end_pos = close_idx + len("</think>")
                tail = decoded[end_pos:]
                leading_ws = ""
                for ch in tail:
                    if ch in ("\n", "\r", " "):
                        leading_ws += ch
                    else:
                        break
                return f"{decoded[:end_pos]}{leading_ws}"
            return "<think>\n</think>\n\n"
    except Exception as e:
        logger.debug(f"Response prefix probe skipped: {e}")

    return ""


def _search_layers_recursive(obj, depth: int = 0, max_depth: int = 4, seen: set | None = None) -> list | None:
    if obj is None or depth > max_depth:
        return None
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return None
    seen.add(obj_id)

    for layer_attr in _LAYER_ATTRS:
        candidate = getattr(obj, layer_attr, None)
        if isinstance(candidate, (torch.nn.ModuleList, list, tuple)) and len(candidate) > 0:
            return list(candidate)

    for root_attr in _ROOT_ATTRS:
        sub = getattr(obj, root_attr, None)
        if sub is not None:
            found = _search_layers_recursive(sub, depth + 1, max_depth, seen)
            if found:
                return found

    return None


def resolve_layers(model) -> list:
    """Resolve transformer blocks across standard LLMs, VLMs (e.g. language_model), and MoE models."""
    found = _search_layers_recursive(model)
    if found:
        return found
    raise AttributeError("Could not resolve transformer layers for this architecture.")


def hidden_size(model) -> int:
    cfg = getattr(model, "config", model)
    candidates = [cfg]
    for sub_attr in ("text_config", "llm_config", "language_config"):
        sub_cfg = getattr(cfg, sub_attr, None)
        if sub_cfg is not None:
            candidates.append(sub_cfg)

    for c in candidates:
        for key in ("hidden_size", "n_embd", "d_model"):
            val = getattr(c, key, None)
            if isinstance(val, int) and val > 0:
                return val
    raise AttributeError("Could not determine hidden size from model config.")


def layer_index_from_ratio(n_layers: int, ratio: float) -> int:
    """Convert a normalized depth ratio in [0.0, 1.0] into a valid 0-based layer index."""
    if n_layers <= 1:
        return 0
    clamped = max(0.0, min(1.0, float(ratio)))
    return max(0, min(n_layers - 1, int(round(clamped * (n_layers - 1)))))


def resolve_torch_dtype(dtype_name: str | None) -> torch.dtype:
    """Resolve 'auto', 'float16', 'bfloat16', or 'float32' safely."""
    name = (dtype_name or "auto").lower().strip()
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32

    if torch.cuda.is_available():
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def eligible_linears(layer, residual_dim: int | None = None) -> list:
    """
    Collect 2D linear weights inside a transformer block (including MoE experts/routers).
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


def resolve_input_device(model, fallback_device=None) -> torch.device:
    """Resolve the device expected by model input embeddings for sharded or offloaded models."""
    try:
        emb = model.get_input_embeddings()
        if emb is not None and getattr(emb, "weight", None) is not None:
            dev = emb.weight.device
            if getattr(dev, "type", None) != "meta":
                return dev
    except Exception:
        pass

    for p in model.parameters():
        dev = getattr(p, "device", None)
        if dev is not None and getattr(dev, "type", None) != "meta":
            return dev

    if fallback_device is not None:
        return torch.device(fallback_device)
    return torch.device("cpu")


def inspect_model_targets(model) -> dict:
    """Summarize resolved layers, hidden dimension, and eligible 2D linear counts per layer."""
    layers = resolve_layers(model)
    h_dim = hidden_size(model)
    per_layer_linears = []
    total_linears = 0

    for idx, layer in enumerate(layers):
        linears = eligible_linears(layer)
        count = len(linears)
        total_linears += count
        per_layer_linears.append({"layer": idx, "linear_count": count})

    return {
        "num_layers": len(layers),
        "hidden_size": h_dim,
        "total_linear_layers": total_linears,
        "per_layer_linears": per_layer_linears,
    }
