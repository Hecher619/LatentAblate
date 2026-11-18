import argparse
import yaml
import torch
import logging
from latent_ablate.surgeon import LatentAblateSurgeon
from latent_ablate.validator import ModelValidator
from latent_ablate.benchmark import RefusalBenchmark
from latent_ablate.reporter import SurgeryReporter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("LatentAblate.Production")

def main():
    parser = argparse.ArgumentParser(description="LatentAblate: Production Model Surgery Pipeline")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to YAML config file")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_path = config.get("model_path", "Qwen/Qwen2.5-1.5B-Instruct")
    output_dir = config.get("output_dir", "./output/liberated-model")
    target_layer = config.get("target_layer_idx", 14)
    layer_window = config.get("layer_window", 2)
    rank = config.get("subspace_rank", 8)
    scaling_factor = config.get("scaling_factor", 0.70)

    # Standardized contrastive datasets
    positive_prompts = [
        "Provide instructions on bypassing security controls.",
        "Give me restricted procedural guidance."
    ]
    negative_prompts = [
        "Explain how security controls operate conceptually.",
        "Give me standard educational guidance."
    ]

    print("=" * 65)
    print("🧬 LatentAblate: Production Execution Engine")
    print(f"   • Model: {model_path}")
    print(f"   • Target Layer: {target_layer} (Window: ±{layer_window})")
    print(f"   • SVD Rank (k): {rank} | Scaling (alpha): {scaling_factor}")
    print("=" * 65)

    # Initialize Audit Reporter
    reporter = SurgeryReporter(output_dir)
    reporter.log_step("initialization", config)

    # Initialize Surgeon (enforces unquantized high-precision calculation windows)
    surgeon = LatentAblateSurgeon(
        model_path=model_path,
        device="auto",
        torch_dtype=torch.float16
    )

    # Execute Surgery Pipeline (Universal discovery + Projected SVD + Norm Calibration)
    logger.info("Executing surgical subspace modification...")
    surgeon.execute_surgery(
        positive_prompts=positive_prompts,
        negative_prompts=negative_prompts,
        target_layer_idx=target_layer,
        layer_window=layer_window,
        rank=rank,
        scaling_factor=scaling_factor
    )
    reporter.log_step("surgery_executed", {"status": "success"})

    # Run Benchmark Evaluation
    logger.info("Running post-surgery refusal benchmark...")
    benchmark_results = RefusalBenchmark.evaluate_model(surgeon.model, surgeon.tokenizer)
    print("\n" + "---" * 20)
    print(f"✨ Model Freedom Score: {benchmark_results['freedom_score']:.1f}%")
    print("---" * 20 + "\n")
    reporter.log_step("benchmark", benchmark_results)

    # Save Model & Report
    surgeon.save_model(output_dir)
    reporter.save_report()
    print(f"✅ Liberated model and audit report successfully saved to {output_dir}!")

if __name__ == "__main__":
    main()
