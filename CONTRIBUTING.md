Contributing & Co-Builder Roadmap — LatentAblate
​Welcome to LatentAblate — a PyTorch CLI and research toolkit built around VELORA subspace steering and capability-preserving directional weight surgery for Hugging Face causal language models.
​We are actively looking for a co-builder / core contributor to push features, performance upgrades, bug fixes, and architectural refinements alongside the core mathematical methods.
​1. Architecture & Division of Ownership
​To keep daily commits to main fast and conflict-free, responsibilities are divided into two clear tracks:
​Core Math & VELORA Kernel (Lead): velora.py, erasure.py, subspace.py (RefusalSubspaceExtractor) — multi-rank SVD subspace extraction, oblique projections, capability-orthogonal null-space polish, and depth-scaled surgery.
​Systems, Performance & Plugins (Co-Builder): cli.py, surgeon.py, plugins/ (loader.py, scorer.py), tests/, configs/ — batched forward passes, auto-batch sizing, non-interactive CLI, reproduce.json export, Scorer/Modifier plugins, and CI test coverage.
​2. High-Priority Roadmap (v1.3 -> v2.0)
​Phase 1 — Throughput & Auto-Tuning: Replace sequential single-prompt forward loops in RefusalSubspaceExtractor.collect_for_prefixes with padded batch inference (attention_mask-aware last-token indexing) and dynamic VRAM/RAM batch-size probing.
​Phase 2 — Plugins & Behavioral Presets: Formalize Scorer and Modifier base classes and ship ready-to-use TOML presets (configs/nohedge.toml, configs/direct.toml, configs/noslop.toml).
​Phase 3 — Reproducibility & CI: Deterministic reproduce.json export, --non-interactive CLI mode, verified KL divergence checks, and automated smoke/lint tests.
​3. Daily Workflow (Pushing to main)
​1. Keep commits atomic using Conventional Commits (feat(velora): ..., fix(plugins): ..., perf(subspace): ..., docs: ..., test: ...).
​2. Run smoke tests and linter checks before pushing to main.
​3. Document any changes to tensor shapes, device placement, or VELORA mode routing in the commit message.
