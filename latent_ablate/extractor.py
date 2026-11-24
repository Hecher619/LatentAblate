import gc
import logging
import torch

from latent_ablate.arch import format_chat_prompt, infer_device, resolve_layers
from latent_ablate.erasure import _is_meta

logger = logging.getLogger("LatentAblate.Extractor")


class ActivationExtractor:
    """Extracts per-layer last-token hidden states across prompts with OOM fallback."""

    def __init__(self, model, tokenizer, device="auto", response_prefix: str = ""):
        self.model = model
        self.tokenizer = tokenizer
        if device is None or str(device) == "auto":
            self.device = infer_device(model)
        else:
            self.device = torch.device(device)
        self.response_prefix = response_prefix or ""
        self._cache: dict[tuple[int, tuple[str, ...]], torch.Tensor] = {}

    def clear_cache(self) -> None:
        self._cache.clear()

    def _run_batch(self, prompts_batch: list[str], target_layer) -> torch.Tensor | None:
        formatted = [
            format_chat_prompt(self.tokenizer, p, response_prefix=self.response_prefix)
            for p in prompts_batch
        ]
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        if pad_id is None:
            pad_id = eos_id if eos_id is not None else 0

        if getattr(self.tokenizer, "pad_token", None) is None and getattr(self.tokenizer, "eos_token", None) is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        try:
            encoded = self.tokenizer(
                formatted,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
        except Exception:
            encoded = self.tokenizer(formatted[0], return_tensors="pt", truncation=True)

        if hasattr(encoded, "to"):
            inputs = encoded.to(self.device)
        else:
            inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in encoded.items()}

        captured: list[torch.Tensor] = []

        def _hook(_module, _inp, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden is None or _is_meta(hidden):
                return
            if hidden.ndim == 3:
                captured.append(hidden[:, -1, :].detach().float().cpu())
            elif hidden.ndim == 2:
                captured.append(hidden.detach().float().cpu())

        handle = target_layer.register_forward_hook(_hook)
        try:
            self.model.eval()
            with torch.no_grad():
                self.model(**inputs)
        finally:
            handle.remove()

        if not captured:
            return None
        return torch.cat(captured, dim=0)

    def extract_hidden_states(
        self,
        prompts: list[str],
        layer_idx: int,
        batch_size: int = 8,
        use_cache: bool = False,
    ) -> torch.Tensor:
        if not prompts:
            return torch.empty(0, dtype=torch.float32)

        valid_prompts = [str(p).strip() for p in prompts if isinstance(p, str) and p.strip()]
        if not valid_prompts:
            return torch.empty(0, dtype=torch.float32)

        layers = resolve_layers(self.model)
        if not layers or layer_idx < 0 or layer_idx >= len(layers):
            logger.warning(f"Layer index {layer_idx} out of range for model with {len(layers)} layers.")
            return torch.empty(0, dtype=torch.float32)

        cache_key = (int(layer_idx), tuple(valid_prompts))
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key].to(self.device)

        target_layer = layers[int(layer_idx)]
        collected: list[torch.Tensor] = []
        bs = max(1, int(batch_size))

        for start in range(0, len(valid_prompts), bs):
            chunk = valid_prompts[start : start + bs]
            try:
                batch_out = self._run_batch(chunk, target_layer)
            except torch.cuda.OutOfMemoryError:
                logger.warning("CUDA OOM during activation extraction; retrying prompts one by one.")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                mini_parts = []
                for single_p in chunk:
                    part = self._run_batch([single_p], target_layer)
                    if part is not None and part.numel() > 0:
                        mini_parts.append(part)
                batch_out = torch.cat(mini_parts, dim=0) if mini_parts else None
            except RuntimeError as e:
                if "meta" in str(e).lower():
                    logger.warning(f"Meta tensor encountered during extraction at layer {layer_idx}: {e}")
                    return torch.empty(0, dtype=torch.float32)
                raise

            if batch_out is not None and batch_out.numel() > 0:
                collected.append(batch_out)

        if not collected:
            return torch.empty(0, dtype=torch.float32)

        result_cpu = torch.cat(collected, dim=0).contiguous()
        if use_cache:
            self._cache[cache_key] = result_cpu
        return result_cpu.to(self.device)
