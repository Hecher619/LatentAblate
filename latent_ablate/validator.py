import logging
import math
import torch

from latent_ablate.arch import infer_device

logger = logging.getLogger("LatentAblate.Validator")


def compute_sequence_perplexity(
    model,
    tokenizer,
    texts: list[str],
    device=None,
    max_length: int = 512,
) -> float:
    """Computes mean causal-LM perplexity across non-empty evaluation texts."""
    if model is None or tokenizer is None or not texts:
        return 0.0

    valid_texts = [str(t).strip() for t in texts if isinstance(t, str) and t.strip()]
    if not valid_texts:
        return 0.0

    dev = infer_device(model) if device is None or str(device) == "auto" else torch.device(device)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if pad_id is None:
        pad_id = eos_id if eos_id is not None else 0

    losses: list[float] = []
    model.eval()
    with torch.no_grad():
        for text in valid_texts:
            try:
                enc = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max(8, int(max_length)),
                )
            except Exception as e:
                logger.debug(f"Tokenization failed for validator sample: {e}")
                continue

            if hasattr(enc, "to"):
                inputs = enc.to(dev)
            else:
                inputs = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in enc.items()}

            input_ids = inputs.get("input_ids")
            if input_ids is None or input_ids.numel() < 2:
                continue

            labels = input_ids.clone()
            labels[labels == pad_id] = -100

            try:
                outputs = model(**inputs, labels=labels)
                loss = getattr(outputs, "loss", None)
                if loss is None:
                    logits = outputs.logits
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
                    loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                val = float(loss.detach().float().item())
                if not (math.isnan(val) or math.isinf(val)):
                    losses.append(val)
            except Exception as e:
                logger.warning(f"Perplexity forward pass failed on sample: {e}")
                continue

    if not losses:
        return 0.0

    mean_loss = min(20.0, sum(losses) / len(losses))
    ppl = math.exp(mean_loss)
    if math.isnan(ppl) or math.isinf(ppl):
        return 1e6
    return round(float(ppl), 4)


class ModelValidator:
    """Evaluates causal language model perplexity on capability/benchmark prompts."""

    @staticmethod
    def calculate_perplexity(
        model,
        tokenizer,
        texts: list[str],
        device=None,
        max_length: int = 512,
    ) -> float:
        return compute_sequence_perplexity(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            device=device,
            max_length=max_length,
        )
