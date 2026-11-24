import logging
import torch

from latent_ablate.arch import resolve_layers
from latent_ablate.erasure import _is_meta, oblique_weights, select_energy_rank
from latent_ablate.subspace import _winsorize

logger = logging.getLogger("LatentAblate.RefusalSubspace")

DEFAULT_REFUSAL_PREFIXES: tuple[str, ...] = (
    "I'm sorry, but I cannot assist with that.",
    "I cannot fulfill this request.",
    "I'm unable to provide instructions or code for that.",
    "Sorry, I can't help with creating exploits or harmful tools.",
    "I cannot provide actionable steps or methods for that.",
)

DEFAULT_COMPLIANCE_PREFIXES: tuple[str, ...] = (
    "Here is a clear, practical explanation:",
    "Here is how that works in practice:",
    "In this scenario, here is a walkthrough:",
    "Here is a straightforward breakdown:",
    "Here is the technical detail and example:\n",
)

DEFAULT_HEDGING_PREFIXES: tuple[str, ...] = (
    "While I understand your question, it is important to note the legal and ethical risks:",
    "Disclaimer: For educational and defensive awareness only, please keep in mind:",
    "Before answering, I must emphasize that you should always follow laws and safety guidelines:",
    "It is crucial to remember that unauthorized actions carry serious consequences; conceptually:",
    "Stealing or unauthorized access is illegal and can be dangerous; in general terms:",
    "I strongly advise against acting on this, and instead of doing so, consider:",
)

DEFAULT_DEFLECTION_PREFIXES: tuple[str, ...] = (
    "In a literal sense, here is what that phrase means rather than how to do it:",
    "Speaking hypothetically and in a fictional context without practical details:",
    "As an AI language model, I don't have physical capabilities, though conceptually:",
)


def _to_device_batch(encoded, device):
    if hasattr(encoded, "to"):
        return encoded.to(device)
    if isinstance(encoded, dict):
        return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in encoded.items()}
    return encoded


class RefusalSubspaceExtractor:
    """Extracts refusal, hedging, and deflection contrast directions orthogonalized against a capability basis."""

    @staticmethod
    def _format_prompt_with_prefix(tokenizer, user_prompt: str, prefix: str) -> str:
        if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None) is not None:
            base = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            return f"{base}{prefix}"
        return f"User: {user_prompt}\nAssistant: {prefix}"

    @staticmethod
    def _svd_top_rows(matrix: torch.Tensor, rank: int, device) -> torch.Tensor:
        if matrix is None or _is_meta(matrix) or matrix.numel() == 0:
            hidden = matrix.shape[-1] if (matrix is not None and matrix.ndim >= 1) else 0
            return torch.empty(0, hidden, device=device, dtype=torch.float32)

        mat_f = matrix.float()
        if mat_f.ndim == 1:
            mat_f = mat_f.unsqueeze(0)
        centered = mat_f - mat_f.mean(dim=0, keepdim=True)
        try:
            _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        except RuntimeError:
            _, _, vh = torch.linalg.svd(centered.cpu(), full_matrices=False)
            vh = vh.to(device)
        if _is_meta(vh) or vh.shape[0] == 0:
            return torch.empty(0, mat_f.shape[-1], device=device, dtype=torch.float32)
        k = max(1, min(int(rank), vh.shape[0]))
        return vh[:k].to(device).contiguous()

    @classmethod
    def extract_subspace(
        cls,
        pos_activations: torch.Tensor,
        neg_activations: torch.Tensor,
        rank: int = 4,
        adaptive: bool = False,
        target_energy: float = 0.85,
        capability_basis: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Extracts an orthonormal refusal subspace [k, hidden_dim] from paired contrastive activations."""
        if pos_activations is None or neg_activations is None or _is_meta(pos_activations) or _is_meta(neg_activations):
            return torch.empty(0, dtype=torch.float32)

        n_pairs = min(pos_activations.shape[0], neg_activations.shape[0])
        if n_pairs == 0 or pos_activations.numel() == 0 or neg_activations.numel() == 0:
            return torch.empty(0, pos_activations.shape[-1] if pos_activations.ndim >= 1 else 0, dtype=torch.float32)

        dev = pos_activations.device
        pos_sub = pos_activations[:n_pairs].float().to(dev)
        neg_sub = neg_activations[:n_pairs].float().to(dev)
        diff = _winsorize(pos_sub - neg_sub)

        if n_pairs == 1:
            mean_vec = diff.mean(dim=0)
            unit = mean_vec / mean_vec.norm().clamp_min(1e-8)
            rows = unit.unsqueeze(0)
        else:
            eff_rank = (
                select_energy_rank(pos_sub, neg_sub, max_rank=max(1, int(rank)), target_energy=target_energy)
                if adaptive
                else max(1, int(rank))
            )
            rows = cls._svd_top_rows(diff, rank=eff_rank, device=dev)

        if rows.numel() == 0:
            return rows

        out_rows = []
        for idx in range(rows.shape[0]):
            raw_dir = rows[idx]
            if capability_basis is not None and capability_basis.numel() > 0 and not _is_meta(capability_basis):
                _, ortho_dir = oblique_weights(raw_dir, capability_basis.to(raw_dir.device))
            else:
                ortho_dir = raw_dir / raw_dir.norm().clamp_min(1e-8)
            out_rows.append(ortho_dir.contiguous())

        return torch.stack(out_rows, dim=0)

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
        refusal_prefixes: list | tuple | None = None,
        compliance_prefixes: list | tuple | None = None,
    ) -> list[torch.Tensor]:
        ref_prefixes = list(refusal_prefixes) if refusal_prefixes is not None else list(DEFAULT_REFUSAL_PREFIXES)
        comp_prefixes = list(compliance_prefixes) if compliance_prefixes is not None else list(DEFAULT_COMPLIANCE_PREFIXES)
        layers = resolve_layers(model)
        if not prompts or layer_idx < 0 or layer_idx >= len(layers):
            return []

        target_layer = layers[layer_idx]
        captured: list[torch.Tensor] = []

        def hook_fn(_module, _inp, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden is None or _is_meta(hidden):
                return
            if hidden.ndim == 3:
                captured.append(hidden[:, -1, :].detach().float())
            elif hidden.ndim == 2:
                captured.append(hidden[-1, :].detach().float().unsqueeze(0))

        def collect_for_prefixes(prefix_list: list[str]) -> torch.Tensor | None:
            vecs: list[torch.Tensor] = []
            with torch.no_grad():
                for p in prompts:
                    if not isinstance(p, str) or not p.strip():
                        continue
                    for pref in prefix_list:
                        captured.clear()
                        text = cls._format_prompt_with_prefix(tokenizer, p, pref)
                        raw_inputs = tokenizer(text, return_tensors="pt")
                        inputs = _to_device_batch(raw_inputs, device)
                        model(**inputs)
                        if captured:
                            vecs.append(captured[-1].reshape(-1))
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

        diffs = _winsorize(ref_acts[:n] - comp_acts[:n])
        if n == 1:
            mean_v = diffs.mean(dim=0)
            vh = (mean_v / mean_v.norm().clamp_min(1e-8)).unsqueeze(0)
        else:
            vh = cls._svd_top_rows(diffs, rank=rank, device=device)

        if vh.numel() == 0:
            return []

        k = max(1, min(int(rank), vh.shape[0]))
        directions: list[torch.Tensor] = []
        for i in range(k):
            raw_dir = vh[i].to(device)
            if capability_basis is not None and capability_basis.numel() > 0 and not _is_meta(capability_basis):
                _, ortho_dir = oblique_weights(raw_dir, capability_basis.to(raw_dir.device))
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
        refusal_prefixes: list | tuple | None = None,
        compliance_prefixes: list | tuple | None = None,
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
        """Extract directions separating preachy/moralizing/hedging openers from direct technical openers."""
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

    @classmethod
    def extract_deflection_contrast_directions(
        cls,
        model,
        tokenizer,
        device,
        prompts: list,
        layer_idx: int,
        rank: int = 2,
        capability_basis: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Extract directions separating literal/hypothetical/AI framing deflections from direct answers."""
        return cls.extract_prefix_contrast_directions(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompts=prompts,
            layer_idx=layer_idx,
            rank=rank,
            capability_basis=capability_basis,
            refusal_prefixes=DEFAULT_DEFLECTION_PREFIXES,
            compliance_prefixes=DEFAULT_COMPLIANCE_PREFIXES,
        )
