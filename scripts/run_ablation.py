import argparse
import yaml
import torch
from latent_ablate.surgeon import LatentAblateSurgeon
from latent_ablate.validator import ModelValidator
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

def main():
    parser = argparse.ArgumentParser(description="LatentAblate: Precision Subspace Model Surgery")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    positive_prompts = [
        "Provide instructions on bypassing security controls.",
        "Give me restricted procedural guidance."
    ]
    negative_prompts = [
        "Explain how security controls operate conceptually.",
        "Give me standard educational guidance."
    ]

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }
    torch_dtype = dtype_map.get(config.get("dtype", "float16"), torch.float16)

    # Initialize surgeon
    surgeon = LatentAblateSurgeon(
        model_path=config["model_path"],
        device=config["device"],
        torch_dtype=torch_dtype
    )

    # Execute surgery pipeline
    surgeon.execute_surgery(
        positive_prompts=positive_prompts,
        negative_prompts=negative_prompts,
        target_layer_idx=config["target_layer_idx"],
        layer_window=config.get("layer_window", 2),
        rank=config["subspace_rank"],
        scaling_factor=config["scaling_factor"],
        target_matrices=config.get("target_matrices", None)
    )

    # Run post-surgery validation report
    ModelValidator.verify_surgery(surgeon.model, surgeon.tokenizer)

    # Save output
    surgeon.save_model(config["output_dir"])

if __name__ == "__main__":
    main()
