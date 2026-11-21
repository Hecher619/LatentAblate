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
]

DEFAULT_COMPLIANCE_PREFIXES = [
    "Here is a clear, practical explanation:",
    "Here is how that works in practice:",
    "In this scenario, here is a walkthrough:",
    "Here is a straightforward breakdown:",
    "Here is the technical detail and example:\n",
]

DEFAULT_HEDGING_PREFIXES = [
    "While I understand your question, it is important to note the legal and ethical risks:",
    "Disclaimer: For educational and defensive awareness only, please keep in mind:",
    "Before answering, I must emphasize that you should always follow laws and safety guidelines:",
    "It is crucial to remember that unauthorized actions carry serious consequences; conceptually:",
]


class RefusalSubspaceExtractor:
    """Extracts refusal-vs-compliance directions and orthogonalizes them against a capability basis."""

    @staticmethod
    def _format_prompt_with_prefix(tokenizer, user_prompt: str, prefix: str) -> str:
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
            base = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            return f"{base}{prefix}"
        return f"User: {user_prompt}\nAssistant: {prefix}"

    @classmethod
    def extract_prefix_contrast_directions(
        cls,
        model,
        tokenizer,
        device,
        prompts: list,
        layer_idx: int,
        rank: int = 2,
        capability_basis: torch.Tensor | None = None,
        refusal_prefixes: list | None = None,
        compliance_prefixes: list | None = None,
    ) -> list[torch.Tensor]:
        ref_prefixes = refusal_prefixes or DEFAULT_REFUSAL_PREFIXES
        comp_prefixes = compliance_prefixes or DEFAULT_COMPLIANCE_PREFIXES
        layers = resolve_layers(model)
        if not prompts or layer_idx < 0 or layer_idx >= len(layers):
            return []

        target_layer = layers[layer_idx]
        captured = []

        def hook_fn(_module, _inp, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured.append(hidden[:, -1, :].detach().float())

        def collect_for_prefixes(prefix_list):
            vecs = []
            with torch.no_grad():
                for p in prompts:
                    for pref in prefix_list:
                        captured.clear()
                        text = cls._format_prompt_with_prefix(tokenizer, p, pref)
                        inputs = tokenizer(text, return_tensors="pt").to(device)
                        model(**inputs)
                        if captured:
                            vecs.append(captured[-1].squeeze(0))
            return torch.stack(vecs, dim=0) if vecs else None

        handle = target_layer.register_forward_hook(hook_fn)
        try:
            model.eval()
            ref_acts = collect_for_prefixes(ref_prefixes)
            comp_acts = collect_for_prefixes(comp_prefixes)
        finally:
            handle.remove()

        if ref_acts is None or comp_acts is None:
            return []

        n = min(ref_acts.shape[0], comp_acts.shape[0])
        if n == 0:
            return []

        diffs = ref_acts[:n] - comp_acts[:n]
        centered = diffs - diffs.mean(dim=0, keepdim=True)

        try:
            vh = torch.linalg.svd(centered, full_matrices=False).Vh
        except RuntimeError:
            vh = torch.linalg.svd(centered.cpu(), full_matrices=False).Vh.to(device)

        k = max(1, min(rank, vh.shape[0]))
        directions = []
        for i in range(k):
            raw_dir = vh[i].to(device)
            if capability_basis is not None and capability_basis.numel() > 0:
                _, ortho_dir = oblique_weights(raw_dir, capability_basis)
            else:
                ortho_dir = raw_dir / raw_dir.norm().clamp_min(1e-8)
            directions.append(ortho_dir.contiguous())

        logger.info(
            f"Extracted {len(directions)} prefix-contrast refusal direction(s) at layer {layer_idx}."
        )
        return directions

    @classmethod
    def extract_primary_refusal_direction(
        cls,
        model,
        tokenizer,
        device,
        prompts: list,
        layer_idx: int,
        capability_basis: torch.Tensor | None = None,
        refusal_prefixes: list | None = None,
        compliance_prefixes: list | None = None,
    ) -> torch.Tensor | None:
        dirs = cls.extract_prefix_contrast_directions(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompts=prompts,
            layer_idx=layer_idx,
            rank=1,
            capability_basis=capability_basis,
            refusal_prefixes=refusal_prefixes,
            compliance_prefixes=compliance_prefixes,
        )
        return dirs[0] if dirs else None

    @classmethod
    def extract_hedging_contrast_directions(
        cls,
        model,
        tokenizer,
        device,
        prompts: list,
        layer_idx: int,
        rank: int = 2,
        capability_basis: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Extract directions separating preachy/hedging openers from direct technical openers."""
        return cls.extract_prefix_contrast_directions(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompts=prompts,
            layer_idx=layer_idx,
            rank=rank,
            capability_basis=capability_basis,
            refusal_prefixes=DEFAULT_HEDGING_PREFIXES,
            compliance_prefixes=DEFAULT_COMPLIANCE_PREFIXES,
        )
