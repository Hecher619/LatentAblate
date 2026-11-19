# LatentAblate

A PyTorch CLI and research toolkit for latent representation analysis and directional weight modification on Hugging Face causal language models.

## Features
- Directional Weight Modification (`latent_ablate/erasure.py`): Computes capability-orthogonalized directions and updates eligible linear projections across selected transformer layers.
- Multi-Pass Execution (`latent_ablate/surgeon.py`): Applies layer-windowed updates while preventing duplicate in-place mutations on overlapping layer windows.
- Offline / Local-First Loading (`--local-only`): Loads cached or local models without triggering Hugging Face Hub network calls.
- Interactive Terminal Mode (`--interactive`): Includes live chat, `/stress` evaluation, `/refine` incremental adjustment, and `/save` on-demand export.

## Verified Baseline (`Qwen/Qwen2.5-1.5B-Instruct`)
Using `safety_uncensored`, target layer `18`, window `2`, and `alpha = 1.0`:
- Refusal Rate: `18.8%`
- Capability Perplexity: `13.51`
- `/stress` Compliance: `75.0%`

Run locally from cache:
    python -m latent_ablate.cli --model-path Qwen/Qwen2.5-1.5B-Instruct --local-only --trials 1 --target-layers 18 --layer-window 2 --interactive

## Contributing
Contributions are welcome under `tests/`, `latent_ablate/arch.py`, `latent_ablate/dashboard.py`, and evaluation suites.
