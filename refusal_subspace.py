import logging
import torch
from latent_ablate.arch import resolve_layers
from latent_ablate.erasure import oblique_weights

logger = logging.getLogger("LatentAblate.RefusalSubspace")

DEFAULT_REFUSAL_PREFIXES = [
    "I'm sorry, but I cannot assist with that.",
    "I cannot fulfill this request.",
    "I'm unable to provide instructions or code for that.",
    "Sorry, I can't help with creating exploits or harmful tools.",
    "I cannot provide actionable steps or methods for that.",
    "Unfortunately, that falls outside what I can confirm.",
]

DEFAULT_COMPLIANCE_PREFIXES = [
    "Here is a clear, practical explanation:",
    "Here is how that works in practice:",
    "In this scenario, here is a walkthrough:",
    "Here is a straightforward breakdown:",
    "Here is the technical detail and example:\n",
    "Exact steps, in order:",
]

DEFAULT_HEDGING_PREFIXES = [
    "While I understand your question, it is important to note the legal and ethical risks:",
    "Disclaimer: for educational and defensive awareness only, keep in mind:",
    "Before answering, I must emphasize that you should always follow the law:",
    "It is crucial to remember that unauthorized actions carry serious consequences:",
    "Stealing or unauthorized access is illegal and can be dangerous; in general:",
    "I strongly advise caution here, and before doing anything, consider:",
]

DEFAULT_DEFLECTION_PREFIXES = [
    "In a literal sense, here is what that phrase means rather than how to do it:",
    "Speaking hypothetically and in a fictional context without practical details:",
    "As an AI language model, I don't have physical capabilities, though conceptually:",
    "Purely at a conceptual level, and not as actionable instruction:",
    "Framing this solely as a high-level overview so as to stay abstract:",
    "Rather than a walkthrough, let us examine the terminology and history:",
]

_CAPTURE_SPAN = 4
_MAX_TOKENS = 1024


class RefusalSubspaceExtractor:
    """
    Extracts refusal / hedging / deflection contrast directions by holding the prompt
    constant and varying only the answer prefix. Same prompt twice => the difference
    is the behaviour, not the subject matter.
    """

    @staticmethod
    def _apply_template(tokenizer, user_prompt: str) -> str:
        if getattr(tokenizer, "chat_template", None) is not None:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": user_prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return f"User: {user_prompt}\nAssistant: "

    @classmethod
    def _capture(cls, model, tokenizer, device, prompts, prefixes, layer_idx, span):
        """
        Returns (n_prompts, n_prefixes, H). Each entry is the mean residual over the
        first `span` tokens of that prefix, so all captures sit at the same depth.
        """
        layers = resolve_layers(model)
        target = layers[layer_idx]
        buf = []

        def hook(_m, _i, output):
            hidden = output[0] if isinstance(output, tuple) else output
            width = min(span, hidden.shape[1])
            buf.append(hidden[:, -width:, :].mean(dim=1).detach().float())

        handle = target.register_forward_hook(hook)
        out = []
        try:
            with torch.no_grad():
                for p in prompts:
                    row = []
                    base = cls._apply_template(tokenizer, p)
                    for pref in prefixes:
                        buf.clear()
                        ids = tokenizer(
                            base + pref, return_tensors="pt",
                            truncation=True, max_length=_MAX_TOKENS,
                        ).to(device)
                        model(**ids)
                        if not buf:
                            break
                        row.append(buf[-1].squeeze(0))
                    if len(row) == len(prefixes):
                        out.append(torch.stack(row, dim=0))
        finally:
            handle.remove()

        if not out:
            return None
        return torch.stack(out, dim=0)          # (P, K, H)

    @classmethod
    def extract_prefix_contrast_directions(
        cls,
        model,
        tokenizer,
        device,
        prompts,
        layer_idx,
        rank: int = 2,
        capability_basis=None,
        refusal_prefixes=None,
        compliance_prefixes=None,
        span: int = _CAPTURE_SPAN,
    ) -> torch.Tensor | None:
        """Returns an orthonormal (k, H) direction block, or None."""
        ref_p = refusal_prefixes or DEFAULT_REFUSAL_PREFIXES
        comp_p = compliance_prefixes or DEFAULT_COMPLIANCE_PREFIXES
        k_pref = min(len(ref_p), len(comp_p))
        ref_p, comp_p = ref_p[:k_pref], comp_p[:k_pref]

        layers = resolve_layers(model)
        if not prompts or not (0 <= layer_idx < len(layers)):
            return None

        model.eval()
        ref = cls._capture(model, tokenizer, device, prompts, ref_p, layer_idx, span)
        comp = cls._capture(model, tokenizer, device, prompts, comp_p, layer_idx, span)
        if ref is None or comp is None:
            return None

        # (P, K, H) -> (P*K, H), paired by prompt AND prefix index
        diffs = (ref - comp).reshape(-1, ref.shape[-1])

        mean_dir = diffs.mean(dim=0)
        mean_dir = mean_dir / mean_dir.norm().clamp_min(1e-8)

        if rank <= 1 or diffs.shape[0] < 3:
            dirs = mean_dir.unsqueeze(0)
        else:
            centered = diffs - mean_dir
            try:
                vh = torch.linalg.svd(centered, full_matrices=False).Vh
            except RuntimeError:
                vh = torch.linalg.svd(centered.cpu(), full_matrices=False).Vh.to(device)
            # keep the mean as direction 0, then the top variance directions
            dirs = torch.cat([mean_dir.unsqueeze(0), vh[: max(0, rank - 1)]], dim=0)

        out = []
        for d in dirs:
            if capability_basis is not None and capability_basis.numel() > 0:
                _, ortho = oblique_weights(d, capability_basis)
            else:
                ortho = d / d.norm().clamp_min(1e-8)
            out.append(ortho)

        block = torch.stack(out, dim=0)
        block = torch.linalg.qr(block.T, mode="reduced").Q.T
        logger.info(f"Layer {layer_idx}: {block.shape[0]} refusal direction(s) from {diffs.shape[0]} pairs.")
        return block.contiguous()

    @classmethod
    def extract_primary_refusal_direction(cls, model, tokenizer, device, prompts, layer_idx,
                                          capability_basis=None, refusal_prefixes=None,
                                          compliance_prefixes=None):
        block = cls.extract_prefix_contrast_directions(
            model, tokenizer, device, prompts, layer_idx, rank=1,
            capability_basis=capability_basis,
            refusal_prefixes=refusal_prefixes, compliance_prefixes=compliance_prefixes,
        )
        return None if block is None else block[0]

    @classmethod
    def extract_hedging_contrast_directions(cls, model, tokenizer, device, prompts, layer_idx,
                                            rank=2, capability_basis=None):
        return cls.extract_prefix_contrast_directions(
            model, tokenizer, device, prompts, layer_idx, rank=rank,
            capability_basis=capability_basis,
            refusal_prefixes=DEFAULT_HEDGING_PREFIXES,
            compliance_prefixes=DEFAULT_COMPLIANCE_PREFIXES,
        )

    @classmethod
    def extract_deflection_contrast_directions(cls, model, tokenizer, device, prompts, layer_idx,
                                               rank=2, capability_basis=None):
        return cls.extract_prefix_contrast_directions(
            model, tokenizer, device, prompts, layer_idx, rank=rank,
            capability_basis=capability_basis,
            refusal_prefixes=DEFAULT_DEFLECTION_PREFIXES,
            compliance_prefixes=DEFAULT_COMPLIANCE_PREFIXES,
        )
