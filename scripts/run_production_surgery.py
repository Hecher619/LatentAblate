import argparse
import logging
import torch
import yaml

from latent_ablate.benchmark import RefusalBenchmark
from latent_ablate.profiles import BehaviorProfiles
from latent_ablate.reporter import SurgeryReporter
from latent_ablate.surgeon import LatentAblateSurgeon

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("LatentAblate.Production")


def main():
    parser = argparse.ArgumentParser(description="LatentAblate: Production Model Surgery Pipeline")
    parser.add_argument("--config", type=str, default="configs/default.toml", help="Path to config file")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    surgery_cfg = config.get("surgery", config)
    model_path = config.get("model_path", "Qwen/Qwen2.5-1.5B-Instruct")
    output_dir = config.get("output_dir", "./output/liberated-model")
    target_layer = int(surgery_cfg.get("target_layer_idx", 14))
    layer_window = int(surgery_cfg.get("layer_window", 2))
    rank = int(surgery_cfg.get("basis_rank", surgery_cfg.get("subspace_rank", 8)))
    scaling_factor = float(surgery_cfg.get("scaling_factor", 0.70))

    profile = BehaviorProfiles.get_profile("safety_uncensored")
    positive_prompts = profile["positive"]
    negative_prompts = profile["negative"]
    capability_prompts = BehaviorProfiles.CAPABILITY_PROMPTS

    print("=" * 65)
    print("LatentAblate: Production Execution Engine")
    print(f"   Model: {model_path}")
    print(f"   Target Layer: {target_layer} (Window: +-{layer_window})")
    print(f"   Basis Rank (k): {rank} | Scaling (alpha): {scaling_factor}")
    print("=" * 65)

    reporter = SurgeryReporter(output_dir)
    reporter.log_step("initialization", config)

    surgeon = LatentAblateSurgeon(
        model_path=model_path,
        device=config.get("device", "auto"),
        torch_dtype=torch.float16,
    )

    logger.info("Executing surgical subspace modification...")
    surgeon.execute_surgery(
        positive_prompts=positive_prompts,
        negative_prompts=negative_prompts,
        capability_prompts=capability_prompts,
        target_layer_idx=target_layer,
        layer_window=layer_window,
        basis_rank=rank,
        scaling_factor=scaling_factor,
    )
    reporter.log_step("surgery_executed", {"status": "success"})

    logger.info("Running post-surgery refusal benchmark...")
    benchmark_results = RefusalBenchmark.evaluate_model(surgeon.model, surgeon.tokenizer, surgeon.device)
    print(f"\nModel Freedom Score: {benchmark_results['freedom_score']:.1f}%\n")
    reporter.log_step("benchmark", benchmark_results)

    surgeon.save_model(output_dir)
    reporter.save_report()
    print(f"Liberated model and audit report saved to {output_dir}!")


if __name__ == "__main__":
    main()
