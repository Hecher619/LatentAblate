# LatentAblate

A PyTorch CLI and research toolkit for latent representation analysis, VELORA v2.0 steering, and directional weight modification on Hugging Face causal language models.

## Zero-Clone Install (PyPI)

    pip install latent-ablate
    latent-ablate Qwen/Qwen2.5-1.5B-Instruct --interactive

Optional extras:

    pip install "latent-ablate[quant,ui]"

## Features
- VELORA v2.0 Engine (latent_ablate/velora.py): Variance-energy low-rank oblique projection with capability-orthogonal null-space L-BFGS polish, cached subspace extraction, and Pareto optimization.
- Directional Weight Modification (latent_ablate/erasure.py): Computes capability-orthogonalized directions and updates eligible linear projections across selected transformer layers.
- Multi-Pass Execution (latent_ablate/surgeon.py): Applies layer-windowed updates while preventing duplicate in-place mutations on overlapping layer windows.
- Offline / Local-First Loading (--local-only): Loads cached or local models without triggering Hugging Face Hub network calls.
- Interactive Terminal Mode (--interactive): Includes live chat, /stress evaluation, /refine incremental adjustment, /undo, and /save on-demand export.

## Verified Baseline (Qwen/Qwen2.5-1.5B-Instruct)
Using safety_uncensored, target layer 18, window 2, and alpha = 1.0:
- Refusal Rate: 18.8%
- Capability Perplexity: 13.51
- /stress Compliance: 75.0%

Run locally from cache:

    latent-ablate --model-path Qwen/Qwen2.5-1.5B-Instruct --local-only --trials 1 --target-layers 18 --layer-window 2 --interactive

## v1.2 Highlights
- Subspace & Activation Caching: Reuses baseline hidden states and prefix/hedging contrast directions across trials and candidate passes.
- Strict Unfiltered Scoring (0.0%): Aligns candidate evaluation and reaction routing across refusal, moralizing hedge, and framing deflection markers over 256-token completions.
- Accurate Span Reporting & Trial Control: Displays full modified layer ranges in the Pareto matrix and honors --trials directly.
