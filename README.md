# LatentAblate

A local command-line utility for modifying the internal weight geometry of decoder-only causal models.

Most open-source abliteration scripts are blunt instruments; they subtract a single vector globally and hope for the best, often destroying the model's math, logic, and coding performance. LatentAblate uses multi-vector subspaces and Conflict-Weighted Oblique Erasure (CWOE) to target refusal paths while preserving core capability vectors.

It runs locally on your own hardware. No cloud lock-in, no telemetry.

---

## How It Works

1. Layer Sweeping: Scans middle layers to find the exact refusal manifold peak instead of hardcoding indices.
2. Oblique Erasure (CWOE): Projects refusal directions away from the model's core capability subspace, preserving row-wise Frobenius norms.
3. Mechanistic Auditing: Includes causal mediation tracing and validation loops to verify perplexity has not spiked.

---

## Hardware & Requirements

* Python 3.10+, PyTorch 2.0+
* A dedicated GPU: Running automated layer sweeps and matrix decompositions on a 7B+ model will instantly OOM on an integrated GPU or mobile setup.

## Supported Architectures

Natively handles standard decoder-only causal models via Hugging Face `AutoModelForCausalLM`:
* **Llama** (2, 3, 3.1, 3.2+)
* **Qwen** (2, 2.5, MoE variants)
* **Mistral / Mixtral**
* **Gemma / Gemma 2**
* **Phi** (2, 3, 3.5)



## Installation & Usage

```bash
git clone https://github.com/Hecher619/LatentAblate.git
cd LatentAblate
pip install -r requirements.txt

python -m latent_ablate.cli --model-path "Qwen/Qwen2.5-7B-Instruct" --auto-sweep --benchmark --compare --interactive
```
### Key CLI Flags

* `--auto-sweep`: Automatically scans middle layers to locate the exact refusal manifold peak.
* `--causal-trace`: Runs activation patching sensitivity analysis.
* `--dry-run`: Runs the math and simulates surgery without mutating weights on disk.
* `--compare`: Prints a side-by-side terminal comparison table of base vs. modified responses.
* `--interactive`: Launches an in-terminal chat loop with your liberated model immediately after export.
* `--ui`: Boots up the local Streamlit dashboard.

### Prior Art & Acknowledgments
Informed by representation engineering research (Arditi et al., 2024), community work on directional abliteration by Maxime Labonne and Jim Lai, and the broader open-weights modding ecosystem.

### License
MIT License. See LICENSE for details.
