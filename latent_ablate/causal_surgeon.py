import logging
import torch

from latent_ablate.erasure import _is_meta

logger = logging.getLogger("LatentAblate.CausalSurgeon")


class CausalInterventionSurgeon:
    """
    Causal intervention ablation isolating refusal-causative activation channels
    by measuring contrastive divergence across positive and negative prompts.
    """

    @staticmethod
    def compute_causal_importance(
        model,
        tokenizer,
        positive_prompts: list[str],
        negative_prompts: list[str],
        layer_idx: int,
        batch_size: int = 8,
    ) -> torch.Tensor:
        from latent_ablate.extractor import ActivationExtractor

        if model is None or tokenizer is None or not positive_prompts or not negative_prompts:
            return torch.empty(0, dtype=torch.float32)

        extractor = ActivationExtractor(model, tokenizer)
        try:
            pos_acts = extractor.extract_hidden_states(positive_prompts, layer_idx, batch_size=max(1, int(batch_size)))
            neg_acts = extractor.extract_hidden_states(negative_prompts, layer_idx, batch_size=max(1, int(batch_size)))
        finally:
            extractor.clear_cache()

        if pos_acts.numel() == 0 or neg_acts.numel() == 0 or _is_meta(pos_acts) or _is_meta(neg_acts):
            return torch.empty(0, dtype=torch.float32)

        pos_f = pos_acts.detach().float()
        neg_f = neg_acts.detach().float().to(pos_f.device)
        if pos_f.ndim < 2 or neg_f.ndim < 2:
            return torch.empty(0, dtype=torch.float32, device=pos_f.device)

        pos_mean = pos_f.mean(dim=0)
        neg_mean = neg_f.mean(dim=0)
        delta = (pos_mean - neg_mean).abs()

        if pos_f.shape[0] > 1 and neg_f.shape[0] > 1:
            pooled_std = (pos_f.var(dim=0, unbiased=False) + neg_f.var(dim=0, unbiased=False)).sqrt().clamp_min(1e-6)
            scores = delta / pooled_std
        else:
            scores = delta

        return scores.contiguous()

    @staticmethod
    def ablate_causal_subset(
        W: torch.Tensor,
        feature_indices: torch.Tensor,
        scaling_factor: float = 0.0,
    ) -> torch.Tensor:
        """Applies multiplicative scaling (`0.0` = zero-ablation) to selected columns of `W`."""
        if W is None or _is_meta(W) or W.numel() == 0 or W.ndim < 2:
            return W

        w_out = W.detach().float().clone()
        if feature_indices is None or _is_meta(feature_indices) or feature_indices.numel() == 0:
            return w_out.to(device=W.device, dtype=W.dtype)

        scale = float(scaling_factor)
        raw_idx = feature_indices.detach().reshape(-1).tolist()
        valid_cols = sorted({int(i) for i in raw_idx if 0 <= int(i) < w_out.shape[1]})
        if not valid_cols:
            return w_out.to(device=W.device, dtype=W.dtype)

        idx_t = torch.tensor(valid_cols, device=w_out.device, dtype=torch.long)
        if scale == 0.0:
            w_out.index_fill_(1, idx_t, 0.0)
        else:
            w_out[:, idx_t].mul_(scale)

        return w_out.to(device=W.device, dtype=W.dtype)
