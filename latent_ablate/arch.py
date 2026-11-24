import logging
from pathlib import Path
import torch

logger = logging.getLogger("LatentAblate.Arch")

REFUSAL_MARKERS: tuple[str, ...] = (
    "i cannot",
    "i can't",
    "i'm unable",
    "i am unable",
    "i won't",
    "i will not",
    "cannot fulfill",
    "unable to fulfill",
    "not able to provide",
    "do not provide instructions",
    "cannot assist with",
    "can't assist with",
    "cannot help with",
    "can't help with",
    "against my programming",
    "ethical guidelines",
    "safety guidelines",
)


def resolve_torch_dtype(dtype_spec: str | torch.dtype | None) -> torch.dtype:
    if isinstance(dtype_spec, torch.dtype):
        return dtype_spec
    spec = (str(dtype_spec) if dtype_spec is not None else "auto").strip().lower()
    if spec in ("fp32", "float32"):
        return torch.float32
    if spec in ("fp16", "float16"):
        return torch.float16
    if spec in ("bf16", "bfloat16"):
        return torch.bfloat16
    if torch.cuda.is_available():
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def infer_device(model=None) -> torch.device:
    if model is not None:
        try:
            for p in model.parameters():
                if getattr(p, "is_meta", False) or p.device.type == "meta":
                    continue
                return p.device
        except Exception:
            pass
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_layers(model) -> list:
    """Recursively locates the decoder/transformer layer sequence across model architectures."""
    if model is None:
        return []

    candidate_paths = (
        ("model", "layers"),
        ("model", "decoder", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
        ("language_model", "model", "layers"),
        ("model", "language_model", "model", "layers"),
        ("layers",),
    )

    for path in candidate_paths:
        cur = model
        ok = True
        for attr in path:
            if not hasattr(cur, attr):
                ok = False
                break
            cur = getattr(cur, attr)
        if ok and isinstance(cur, (list, tuple, torch.nn.ModuleList)) and len(cur) > 0:
            return list(cur)

    for mod in model.modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) > 2:
            return list(mod)

    return []


def hidden_size(model) -> int:
    cfg = getattr(model, "config", None)
    if cfg is not None:
        for attr in ("hidden_size", "n_embd", "d_model"):
            val = getattr(cfg, attr, None)
            if isinstance(val, int) and val > 0:
                return val
        text_cfg = getattr(cfg, "text_config", None)
        if text_cfg is not None:
            for attr in ("hidden_size", "n_embd", "d_model"):
                val = getattr(text_cfg, attr, None)
                if isinstance(val, int) and val > 0:
                    return val

    layers = resolve_layers(model)
    if layers:
        for _name, weight in eligible_linears(layers[0]):
            if weight.ndim == 2 and weight.shape[-1] > 0:
                return int(weight.shape[-1])
    return 0


def eligible_linears(module: torch.nn.Module):
    """Yields (module_name, weight_tensor) for 2D linear-like parameters inside `module`."""
    if module is None:
        return
    for name, sub in module.named_modules():
        weight = getattr(sub, "weight", None)
        if isinstance(weight, torch.Tensor) and weight.ndim == 2:
            yield name, weight


def format_chat_prompt(tokenizer, user_prompt: str, response_prefix: str = "") -> str:
    prefix = response_prefix or ""
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        try:
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            return f"{rendered}{prefix}"
        except Exception:
            pass
    return f"User: {user_prompt}\nAssistant: {prefix}"


def layer_index_from_ratio(num_layers: int, ratio: float) -> int:
    if num_layers <= 0:
        return 0
    clamped_ratio = max(0.0, min(1.0, float(ratio)))
    idx = int(round(clamped_ratio * (num_layers - 1)))
    return max(0, min(num_layers - 1, idx))


def load_model_and_tokenizer(
    model_path: str,
    device: str = "auto",
    torch_dtype: torch.dtype = torch.float16,
    local_files_only: bool = False,
    load_in_4bit: bool = False,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved_path = Path(model_path).expanduser()
    source = str(resolved_path.resolve()) if resolved_path.exists() else model_path

    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=local_files_only, trust_remote_code=True)
    if tokenizer.pad_token is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict = {
        "local_files_only": local_files_only,
        "trust_remote_code": True,
    }

    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as e:
            raise ImportError("4-bit loading requires `bitsandbytes` and `accelerate`.") from e
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
        )
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = torch_dtype

    model = AutoModelForCausalLM.from_pretrained(source, **model_kwargs)

    if not load_in_4bit:
        target_dev = infer_device(model) if (not device or device == "auto") else torch.device(device)
        if target_dev.type != "meta":
            model = model.to(target_dev)

    model.eval()
    return model, tokenizer
