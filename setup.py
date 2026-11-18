from setuptools import setup, find_packages

setup(
    name="latent_ablate",
    version="1.2.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "tqdm",
        "optuna",
        "PyYAML",
        "huggingface_hub"
    ],
    entry_points={
        "console_scripts": [
            "latent-optimize=scripts.run_optimized_ablation:main",
            "latent-ablate=scripts.run_ablation:main",
        ],
    },
    author="LatentAblate Community",
    description="Precision Model Surgery & Subspace Optimization Suite",
)

