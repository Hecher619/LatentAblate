# LatentAblate

A local command-line utility for modifying the internal weight geometry of decoder-only causal models.

## Installation & Usage
```bash
git clone [https://github.com/Hecher619/LatentAblate.git](https://github.com/Hecher619/LatentAblate.git)
cd LatentAblate
pip install -r requirements.txt
python -m latent_ablate.cli --model-path "Qwen/Qwen2.5-7B-Instruct" --auto-sweep --benchmark --compare --interactive
```
