import gc
import hashlib
import logging
import torch

from latent_ablate.arch import detect_response_prefix, format_chat_prompt, resolve_layers

logger = logging.getLogger("LatentAblate.Extractor")


class ActivationExtractor:
    """Extracts last-token hidden states with caching, length-bucketing, OOM recovery, and thinking-prefix support."""

    def __init__(
        self,
        model,
        tokenizer,
        device,
        batch_size: int | None = None,
        response_prefix: str | None = None,
        max_length: int = 512,
        enable_cache: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max(32, int(max_length))
        self.enable_cache = enable_cache
        self._cache: dict[tuple[int, str], torch.Tensor] = {}

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.response_prefix = (
            detect_response_prefix(self.model, self.tokenizer, self.device)
            if response_prefix is None
            else response_prefix
        )
        self.batch_size = (
            batch_size
            if batch_size is not None and batch_size > 0
            else self.probe_optimal_batch_size()
        )

    def clear_cache(self):
        self._cache.clear()

    def _prompt_key(self, formatted_prompt: str) -> str:
        payload = f"{self.response_prefix}\n{formatted_prompt}".encode("utf-8", errors="replace")
        return hashlib.blake2b(payload, digest_size=16).hexdigest()

    def probe_optimal_batch_size(self, max_batch: int = 32, seq_len: int = 128) -> int:
        """Probe forward-pass memory headroom on CUDA to pick a safe batch size automatically."""
        if not torch.cuda.is_available() or getattr(self.device, "type", str(self.device)) != "cuda":
            return 4

        best = 1
        candidate = 1
        self.model.eval()
        vocab_size = max(1024, int(getattr(self.model.config, "vocab_size", 32000)))

        while candidate <= max_batch:
            try:
                torch.cuda.synchronize()
                dummy_ids = torch.randint(0, vocab_size, (candidate, seq_len), device=self.device)
                dummy_mask = torch.ones_like(dummy_ids)
                with torch.no_grad():
                    _ = self.model(input_ids=dummy_ids, attention_mask=dummy_mask)
                torch.cuda.synchronize()
                best = candidate
                candidate *= 2
                del dummy_ids, dummy_mask
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.info(f"Batch probe hit VRAM limit at batch_size={candidate}; using {best}.")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    break
                raise

        logger.info(f"Auto-selected activation batch size: {best}")
        return best

    def _run_chunk_with_oom_retry(self, chunk_formatted: list[str], target_layer, initial_bs: int) -> list[torch.Tensor]:
        """Run a chunk in mini-batches, halving batch size automatically if CUDA OOM occurs."""
        results: list[torch.Tensor] = []
        idx = 0
        cur_bs = max(1, initial_bs)

        while idx < len(chunk_formatted):
            sub_chunk = chunk_formatted[idx : idx + cur_bs]
            captured: list[torch.Tensor] = []

            def hook_fn(_module, _inp, output):
                hidden = output[0] if isinstance(output, tuple) else output
                captured.append(hidden.detach().float())

            handle = target_layer.register_forward_hook(hook_fn)
            try:
                enc = self.tokenizer(
                    sub_chunk,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                ).to(self.device)
                with torch.no_grad():
                    self.model(**enc)

                if captured:
                    batch_hidden = captured[-1]  # [B, T, H]
                    attn_mask = enc.get("attention_mask")
                    if attn_mask is not None and attn_mask.numel() > 0:
                        batch_size_cur = batch_hidden.shape[0]
                        seq_len_cur = batch_hidden.shape[1]
                        row_indices = torch.arange(batch_size_cur, device=batch_hidden.device)
                        if getattr(self.tokenizer, "padding_side", "left") == "left":
                            last_idx = torch.full(
                                (batch_size_cur,),
                                max(0, seq_len_cur - 1),
                                device=batch_hidden.device,
                                dtype=torch.long,
                            )
                        else:
                            lengths = attn_mask.sum(dim=1).clamp(min=1, max=seq_len_cur) - 1
                            last_idx = lengths.to(device=batch_hidden.device, dtype=torch.long)
                        vecs = batch_hidden[row_indices, last_idx, :]
                    else:
                        vecs = batch_hidden[:, -1, :]
                    results.append(vecs.cpu())
                idx += len(sub_chunk)
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and cur_bs > 1:
                    new_bs = max(1, cur_bs // 2)
                    logger.warning(f"CUDA OOM at batch_size={cur_bs}; retrying chunk with batch_size={new_bs}.")
                    cur_bs = new_bs
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                else:
                    raise
            finally:
                handle.remove()

        return results

    def extract_hidden_states(
        self,
        prompts: list[str],
        layer_idx: int,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        """
        Extract hidden states at `layer_idx` for `prompts` in original order.
        Uses length-bucketing on uncached prompts to minimize padding overhead on large datasets.
        """
        layers = resolve_layers(self.model)
        if not prompts:
            return torch.empty(0)
        if layer_idx < 0 or layer_idx >= len(layers):
            raise IndexError(f"Layer index {layer_idx} out of range for {len(layers)} layers.")

        clean_prompts = [p for p in prompts if isinstance(p, str) and p.strip()]
        if not clean_prompts:
            return torch.empty(0)

        formatted_all = [
            format_chat_prompt(self.tokenizer, p, response_prefix=self.response_prefix)
            for p in clean_prompts
        ]
        keys = [self._prompt_key(fp) for fp in formatted_all]

        n_total = len(formatted_all)
        out_slots: list[torch.Tensor | None] = [None] * n_total
        missing_indices: list[int] = []

        for i, k in enumerate(keys):
            cache_key = (layer_idx, k)
            if self.enable_cache and cache_key in self._cache:
                out_slots[i] = self._cache[cache_key]
            else:
                missing_indices.append(i)

        if missing_indices:
            bs = max(1, int(batch_size or self.batch_size))
            target_layer = layers[layer_idx]
            self.model.eval()
            prev_side = getattr(self.tokenizer, "padding_side", "right")
            self.tokenizer.padding_side = "left"

            sorted_missing = sorted(missing_indices, key=lambda idx: len(formatted_all[idx]))
            if len(sorted_missing) >= 128:
                logger.info(
                    f"Extracting layer {layer_idx} activations for {len(sorted_missing)} prompts "
                    f"(cached: {n_total - len(sorted_missing)}, batch_size={bs})..."
                )

            try:
                for pos_start in range(0, len(sorted_missing), bs):
                    batch_orig_indices = sorted_missing[pos_start : pos_start + bs]
                    batch_texts = [formatted_all[orig_i] for orig_i in batch_orig_indices]
                    chunk_vecs_list = self._run_chunk_with_oom_retry(batch_texts, target_layer, bs)
                    if chunk_vecs_list:
                        stacked = torch.cat(chunk_vecs_list, dim=0)
                        for row_offset, orig_i in enumerate(batch_orig_indices):
                            vec = stacked[row_offset].contiguous()
                            out_slots[orig_i] = vec
                            if self.enable_cache:
                                self._cache[(layer_idx, keys[orig_i])] = vec
            finally:
                self.tokenizer.padding_side = prev_side

        valid_vecs = [v for v in out_slots if v is not None]
        if not valid_vecs:
            return torch.empty(0)
        return torch.stack(valid_vecs, dim=0)
