import argparse
import gc
import logging
import torch
import yaml

from latent_ablate.benchmark import RefusalBenchmark
from latent_ablate.optimizer import SubspaceOptimizer
from latent_ablate.profiles import BehaviorProfiles
from latent_ablate.surgeon import LatentAblateSurgeon

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("LatentAblate.AutoPipeline")


def main():
    parser = argparse.ArgumentParser(description="LatentAblate: Automated Optuna TPE Optimization & Surgery")
    parser.add_argument("--config", type=str, default="configs/default.toml", help="Path to config file")
    parser.add_argument("--n_trials", type=int, default=10, help="Number of Optuna TPE trial iterations")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load model in 4-bit NF4 via bitsandbytes")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    profile = BehaviorProfiles.get_profile("safety_uncensored")
    positive_prompts = profile["positive"]
    negative_prompts = profile["negative"]
    capability_prompts = BehaviorProfiles.CAPABILITY_PROMPTS

    model_path = config.get("model_path", "Qwen/Qwen2.5-1.5B-Instruct")
    device = config.get("device", "auto")
    output_dir = config.get("output_dir", "./out/liberated_model")

    print("=" * 65)
    print("LatentAblate: Automated Co-Minimization & Surgery Pipeline")
    print("=" * 65)

    logger.info("Initializing Optuna TPE hyperparameter search...")
    optimizer = SubspaceOptimizer(
        model_path=model_path,
        device=device,
        torch_dtype=torch.float16,
    )

    best_params = optimizer.optimize_surgery(
        positive_prompts=positive_prompts,
        negative_prompts=negative_prompts,
        capability_prompts=capability_prompts,
        n_trials=args.n_trials,
    )
    del optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n" + "=" * 65)
    print("Optimization Complete! Best Hyperparameters Found:")
    for param, value in best_params.items():
        print(f"   - {param}: {value}")
    print("=" * 65 + "\n")

    logger.info("Initializing Surgeon with optimized parameters for final execution...")
    surgeon = LatentAblateSurgeon(
        model_path=model_path,
        device=device,
        torch_dtype=torch.float16,
        load_in_4bit=args.load_in_4bit,
    )

    surgeon.execute_multi_pass_surgery(
        passes=[
            {
                "name": "OptimizedPass",
                "positive": positive_prompts,
                "negative": negative_prompts,
                "target_layer": int(best_params.get("target_layer_idx", 14)),
            }
        ],
        capability_prompts=capability_prompts,
        layer_window=int(best_params.get("layer_window", 2)),
        basis_rank_attn=int(best_params.get("basis_rank_attn", 16)),
        basis_rank_mlp=int(best_params.get("basis_rank_mlp", 12)),
        scaling_factor_attn=float(best_params.get("alpha_attn", 1.15)),
        scaling_factor_mlp=float(best_params.get("alpha_mlp", 1.05)),
        target_components=["attn", "mlp"],
        preserve_capability=True,
    )

    logger.info("Running post-surgery refusal benchmark...")
    benchmark_results = RefusalBenchmark.evaluate_model(surgeon.model, surgeon.tokenizer, surgeon.device)
    print("\n" + "---" * 20)
    print(f"Final Model Freedom Score: {benchmark_results['freedom_score']:.1f}%")
    print("---" * 20 + "\n")

    surgeon.save_model(output_dir)
    logger.info(f"Fully optimized and liberated model saved to {output_dir}!")


if __name__ == "__main__":
    main()
