 LatentAblate

A local command-line toolkit for directional weight modification, trial sweeps, evaluation scoring, and interactive inspection of decoder-only causal language models.

Runs locally on your own hardware with no cloud lock-in or telemetry

 How It Works

1. Direction Extraction & Surgery:* Extracts contrastive activation directions and applies directional weight modifications across target layers, attention/MLP components, and rank/alpha settings.
2. Multi-Trial Evaluation:* Evaluates candidate runs using RefusalSuppressionScorer (refusal rate across benchmark prompts) and UtilityPerplexityScorer (token cross-entropy perplexity on evaluation text).
3. Interactive Mode & Reporting:* Lets you select a trial, save the modified model and audit report, chat interactively, and run /stress checks in the terminal.
 Hardware & Requirements

- Python 3.10+, PyTorch 2.0+
- A dedicated GPU with sufficient VRAM for your target model size (larger models and multi-trial sweeps require more memory)

## Supported Architectures

Natively loads causal language models via Hugging Face AutoModelForCausalLM, including

- Qwen / Qwen2.5 (e.g., Qwen2.5-1.5B-Instruct, Qwen2.5-7B-Instruct)
- Llama (2, 3, 3.1, 3.2+)
- Mistral / Mixtral
- Gemma / Gemma 2
- Phi (2, 3, 3.5)

## Installation & Usage


git clone (https://github.com/Hecher619/LatentAblate.git)
cd LatentAblate
pip install -r requirements.txt

Run a multi-trial sweep and enter interactive mode:
python -m latent_ablate.cli --trials 3 --interactive

Key CLI Flags
  --config CONFIG: Path to TOML configuration file (defaults to configs/default.toml).
  --model-path MODEL_PATH: Hugging Face model ID or local model directory.
  --output-dir OUTPUT_DIR: Output directory for the saved model and audit report.
  --modifier {cwoe,ara,soma,riemannian,sae}: Select the ablation/modification strategy.
  --trials TRIALS: Number of parameter/alpha trials to run and compare in the Trial Performance Matrix.
  --optimize: Enable optimization mode during trials.
  --target-layers TARGET_LAYERS: Comma-separated layer indices to target.
  --layer-window LAYER_WINDOW: Window size around target layers.
  --basis-rank-attn BASIS_RANK_ATTN / --basis-rank-mlp BASIS_RANK_MLP: Subspace rank settings for attention and MLP.
  --alpha-attn ALPHA_ATTN / --alpha-mlp ALPHA_MLP: Scaling factors for attention and MLP modifications.
  --prompts-file PROMPTS_FILE: Custom prompts file path.
  --seed SEED: Random seed for reproducibility.
  --publish REPO_ID: Publish resulting artifacts to a target repository ID.
  --interactive: Save the selected trial and launch the interactive terminal chat (with /stress, /help, and exit).
  --ui: Launch the local UI interface.
Prior Art & Acknowledgments
Informed by representation engineering research (Arditi et al., 2024) and open-weights model experimentation tools.
License
MIT License. See LICENSE for details.
