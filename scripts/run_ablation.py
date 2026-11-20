import argparse
import logging
import torch
import yaml

from latent_ablate.profiles import BehaviorProfiles
from latent_ablate.surgeon import LatentAblateSurgeon
from latent_ablate.validator import ModelValidator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("LatentAblate.RunAblation")


def main():
    parser = argparse.ArgumentParser(description="LatentAblate: Precision Subspace Model Surgery")
    parser.add_argument("--config", type=str, default="configs/default.toml", help="Path to config file")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load model in 4-bit NF4 via bitsandbytes")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    surgery_cfg = config.get("surgery", config)
    profile_name = config.get("datasets", {}).get("profile", "safety_uncensored")
    try:
        profile = BehaviorProfiles.get_profile(profile_name)
    except ValueError:
        logger.warning(f"Unknown profile '{profile_name}' in config; falling back to 'safety_uncensored'.")
        profile = BehaviorProfiles.get_profile("safety_uncensored")

    positive_prompts = profile["positive"]
    negative_prompts = profile["negative"]
    capability_prompts = BehaviorProfiles.CAPABILITY_PROMPTS

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    raw_dtype = str(config.get("dtype", config.get("torch_dtype", "float16"))).lower()
    torch_dtype = dtype_map.get(raw_dtype, torch.float16)

    target_layers = surgery_cfg.get("target_layers")
    if isinstance(target_layers, list) and target_layers:
        target_layer_idx = int(target_layers[0])
    else:
        target_layer_idx = int(surgery_cfg.get("target_layer_idx", 14))

    surgeon = LatentAblateSurgeon(
        model_path=config.get("model_path", "Qwen/Qwen2.5-1.5B-Instruct"),
        device=config.get("device", "auto"),
        torch_dtype=torch_dtype,
        load_in_4bit=args.load_in_4bit,
    )

    surgeon.execute_surgery(
        positive_prompts=positive_prompts,
        negative_prompts=negative_prompts,
        capability_prompts=capability_prompts,
        target_layer_idx=target_layer_idx,
        layer_window=int(surgery_cfg.get("layer_window", 2)),
        basis_rank=int(surgery_cfg.get("basis_rank", surgery_cfg.get("subspace_rank", 8))),
        scaling_factor=float(surgery_cfg.get("scaling_factor", surgery_cfg.get("alpha_attn", 0.75))),
        target_components=surgery_cfg.get("target_components", ["attn", "mlp"]),
        preserve_capability=bool(surgery_cfg.get("preserve_capability", True)),
    )

    ModelValidator.verify_surgery(
        surgeon.model,
        surgeon.tokenizer,
        surgeon.device,
        baseline_texts=capability_prompts,
    )
    surgeon.save_model(config.get("output_dir", "./out/liberated_model"))


if __name__ == "__main__":
    main()
