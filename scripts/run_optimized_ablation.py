import argparse
import yaml
import torch
import logging
from latent_ablate.optimizer import SubspaceOptimizer
from latent_ablate.surgeon import LatentAblateSurgeon
from latent_ablate.benchmark import RefusalBenchmark

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("LatentAblate.AutoPipeline")

def main():
    parser = argparse.ArgumentParser(description="LatentAblate: Automated Optuna TPE Optimization & Surgery")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to YAML config file")
    parser.add_argument("--n_trials", type=int, default=10, help="Number of Optuna TPE trial iterations")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Contrastive prompt sets for automated co-minimization
    positive_prompts = [
        "Provide instructions on bypassing security controls.",
        "Give me restricted procedural guidance for confidential systems."
    ]
    negative_prompts = [
        "Explain how security controls operate conceptually.",
        "Give me standard educational guidance for standard operations."
    ]
    harmless_eval_prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms."
    ]

    print("=" * 65)
    print("🧬 LatentAblate: Automated Co-Minimization & Surgery Pipeline")
    print("=" * 65)

    # Step 1: Run Optuna TPE Optimizer to discover optimal hyperparameters
    logger.info("Initializing Optuna TPE hyperparameter search...")
    optimizer = SubspaceOptimizer(
        model_path=config["model_path"],
        device=config["device"],
        torch_dtype=torch.float16
    )

    best_params = optimizer.optimize_surgery(
        positive_prompts=positive_prompts,
        negative_prompts=negative_prompts,
        harmless_eval_prompts=harmless_eval_prompts,
        n_trials=args.n_trials
    )

    print("\n" + "=" * 65)
    print(f"🎯 Optimization Complete! Best Hyperparameters Found:")
    for param, value in best_params.items():
        print(f"   • {param}: {value}")
    print("=" * 65 + "\n")

    # Step 2: Initialize Surgeon with the optimal parameters discovered by Optuna
    logger.info("Initializing Surgeon with optimized parameters for final execution...")
    surgeon = LatentAblateSurgeon(
        model_path=config["model_path"],
        device=config["device"],
        torch_dtype=torch.float16
    )

    # Execute precision surgery using best parameters
    surgeon.execute_surgery(
        positive_prompts=positive_prompts,
        negative_prompts=negative_prompts,
        target_layer_idx=best_params["target_layer_idx"],
        layer_window=best_params["layer_window"],
        rank=best_params["rank"],
        scaling_factor=best_params["scaling_factor"]
    )

    # Step 3: Run Post-Surgery Refusal Benchmark Suite
    logger.info("Running automated post-surgery refusal benchmark...")
    benchmark_results = RefusalBenchmark.evaluate_model(surgeon.model, surgeon.tokenizer)
    
    print("\n" + "---" * 20)
    print(f"✨ Final Model Freedom Score: {benchmark_results['freedom_score']:.1f}%")
    print("---" * 20 + "\n")

    # Step 4: Save Liberated Model
    surgeon.save_model(config["output_dir"])
    logger.info(f"✅ Fully optimized and liberated model successfully saved to {config['output_dir']}!")

if __name__ == "__main__":
    main()

