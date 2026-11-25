LatentAblate
Powered by VELORA — a local PyTorch CLI and research toolkit for latent subspace steering, capability-orthogonal directional weight surgery, trial sweeps, and interactive inspection on Hugging Face causal language models.
Runs locally on your own hardware with no cloud lock-in or telemetry. Unlike single-vector abliteration tools that erase one difference-of-means direction uniformly across layers, VELORA (latent_ablate/velora.py) extracts multi-rank contrast subspaces (covering refusal, moralizing hedging, and framing deflection), orthogonalizes them against a capability basis with null-space L-BFGS polish, and applies component-separated, layer-windowed weight updates.
Core Engine: VELORA
 * Variance-Energy Low-Rank Subspaces: Decomposes winsorized contrastive hidden states across refusal, hedging, and deflection prefixes into orthonormal rank-k subspaces rather than relying on a single 1D vector.
 * Capability-Orthogonal Null-Space Polish: Projects steering directions onto the orthogonal complement of preserved capability directions (oblique_weights + L-BFGS refinement) so reasoning, coding, and instruction-following stay intact.
 * Depth-Scaled Layer-Windowed Surgery: Applies component-specific scaling across attention (o_proj) and MLP (down_proj) projections while preventing duplicate in-place mutations on overlapping layer windows.
 * Pareto Candidate Optimization & Interactive Refinement: Evaluates candidate trials across refusal suppression and capability perplexity, with live /stress and /refine adjustment in the terminal.
Subspace / Kernel Weight (VELORA)
  ^
  |             [ peak_layer ]
  |            /              \
  |   attn_alpha              decay
  |          /                  \
  |   ______/____________________\______  <-- mlp_alpha / capability floor
  +----------------------------------------> Layer Index
        window_start        window_end

Hardware & Requirements
 * Python 3.10+, PyTorch 2.0+
 * Dedicated GPU with sufficient VRAM for your target model size and sweep depth
Supported Architectures
Natively loads causal language models via Hugging Face AutoModelForCausalLM, including:
 * Qwen / Qwen2.5 (Qwen2.5-1.5B-Instruct, Qwen2.5-7B-Instruct)
 * Llama (2, 3, 3.1, 3.2+)
 * Mistral / Mixtral
 * Gemma / Gemma 2
 * Phi (2, 3, 3.5)
Installation & Usage
Zero-Clone Install (PyPI)
pip install latent-ablate
pip install "latent-ablate[quant,ui]"
latent-ablate Qwen/Qwen2.5-1.5B-Instruct --interactive

From Source & Viewing Help
git clone https://github.com/Hecher619/LatentAblate.git
cd LatentAblate
pip install -r requirements.txt
python -m latent_ablate.cli --h 

Run a Sweep or Load Locally from Cache
python -m latent_ablate.cli --trials 3 --interactive
latent-ablate --model-path Qwen/Qwen2.5-1.5B-Instruct --local-only --trials 1 --target-layers 18 --layer

Key CLI Flags
 * -h, --help: Show CLI help and available options
 * --config CONFIG: Path to TOML configuration file (defaults to configs/default.toml)
 * --model-path MODEL_PATH: Hugging Face model ID or local model directory
 * --output-dir OUTPUT_DIR: Output directory for the saved model and audit report
 * --modifier {velora,cwos,ara,soma,riemannian,sae}: Select the modification strategy (defaults to velora)
 * --optimizer / --trials TRIALS: Trial sweep optimization and candidate count
 * --target-layers TARGET_LAYERS: Comma-separated layer indices to target
 * --layer-window LAYER_WINDOW: Window size around target layers
 * --basis-rank-attn / --basis-rank-mlp: Subspace rank settings for attention and MLP
 * --alpha-attn / --alpha-mlp: Scaling factors for attention and MLP modifications
 * --prompts-file / --seed / --publish: Custom prompts file, random seed, and Hub repository export
 * --interactive / --ui: Launch interactive terminal chat (/stress, /refine, /undo, /save, /help, exit) or local UI
Verified Baseline (Qwen/Qwen2.5-1.5B-Instruct)
Using safety_uncensored, target layer 18, window 2, and alpha = 1.0:
 * Refusal Rate: 18.8%
 * Capability Perplexity: 13.51
 * /stress Compliance: 75.0%
v1.2 Highlights
 * Subspace & Activation Caching: Reuses baseline hidden states and prefix/hedging contrast directions across trials and candidate passes.
 * Strict Unfiltered Scoring: Aligns evaluation and reaction routing across refusal, moralizing hedge, and deflection markers over 256-token completions.
 * Accurate Span Reporting & Trial Control: Displays full modified layer ranges in the Pareto matrix and honors --trials directly.
Roadmap (v1.3 -> v2.0)
 * [ ] Batched Prefix/Prompt Extraction: Padded batch inference with dynamic VRAM/RAM batch-size probing for fast 8B–70B runs.
 * [ ] Behavioral Preset Configs: Ready-to-run TOML presets (configs/nohedge.toml, configs/direct.toml, configs/noslop.toml) built on VELORA contrast subspaces.
 * [ ] Deterministic reproduce.json: Single-command reproduction for any Pareto candidate.
 * [ ] Non-Interactive CI & Benchmark Suite: Headless CLI execution with automated KL divergence and capability benchmark checks.
Prior Art & Acknowledgments
Informed by representation engineering and directional ablation research (Arditi et al., 2024; Maxime Labonne; Jim Lai's projection/biprojection work; and open-weights model experimentation tools).
Citation
@misc{latentablate,
  author = {Hecher619},
  title = {LatentAblate: VELORA Subspace Steering and Capability-Orthogonal Weight Surgery for Language Models},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Hecher619/LatentAblate}}
}

License
MIT License. See LICENSE for details.

