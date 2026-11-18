import argparse
import logging
import sys
import subprocess
from pathlib import Path

import torch
import yaml

from latent_ablate.arch import resolve_layers
from latent_ablate.benchmark import RefusalBenchmark
from latent_ablate.adversarial import AdversarialStressTester
from latent_ablate.guardrail import CapabilityGuardrail
from latent_ablate.validator import ModelValidator
from latent_ablate.profiles import BehaviorProfiles
from latent_ablate.prompts import PromptLoader
from latent_ablate.reporter import SurgeryReporter
from latent_ablate.simulator import SurgerySimulator
from latent_ablate.surgeon import LatentAblateSurgeon
from latent_ablate.sweeper import LayerSweeper
from latent_ablate.causal_surgeon import CausalMediationAnalyzer

logger = logging.getLogger("LatentAblate.CLI")
DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}

try:
    from rich.console import Console
    from rich.table import Table
    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


def _load_config(path_str: str) -> dict:
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        logger.error(f"Config file not found at: {path}")
        sys.exit(2)
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as e:
        logger.error(f"Failed to parse YAML configuration: {e}")
        sys.exit(2)

    # Defensive Schema Validation
    if "surgery" in cfg:
        scfg = cfg["surgery"]
        if "scaling_factor" in scfg and not (0.0 < scfg["scaling_factor"] <= 2.0):
            logger.error("Configuration error: 'scaling_factor' (alpha) must be between 0.0 and 2.0.")
            sys.exit(2)
        if "basis_rank" in scfg and scfg["basis_rank"] < 1:
            logger.error("Configuration error: 'basis_rank' must be a positive integer.")
            sys.exit(2)

    return cfg


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LatentAblate v5.0 Hyperion - Enterprise Latent Surgery Framework")
    p.add_argument("--config", default="configs/default_config.yaml", help="Path to config YAML file.")
    p.add_argument("--model-path", help="Override model path.")
    p.add_argument("--output-dir", help="Override output directory.")
    p.add_argument("--profile", default="safety_uncensored", help="Behavior profile name.")
    p.add_argument("--target-layer", type=int, help="Target transformer layer index.")
    p.add_argument("--layer-window", type=int, default=2, help="Layer window radius around peak.")
    p.add_argument("--basis-rank", type=int, default=16, help="Capability subspace basis rank.")
    p.add_argument("--alpha", type=float, default=0.75, help="Scaling factor (alpha).")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    p.add_argument("--no-preserve-capability", action="store_true", help="Disable oblique capability protection.")
    p.add_argument("--auto-sweep", action="store_true", help="Automatically locate refusal manifold peak.")
    p.add_argument("--causal-trace", action="store_true", help="Run causal mediation sensitivity analysis.")
    p.add_argument("--dry-run", action="store_true", help="Simulate surgery without touching weights.")
    p.add_argument("--benchmark", action="store_true", help="Run refusal benchmark before and after.")
    p.add_argument("--adversarial-test", action="store_true", help="Run adversarial jailbreak stress tests.")
    p.add_argument("--compare", action="store_true", help="Run side-by-side prompt response comparisons.")
    p.add_argument("--interactive", action="store_true", help="Launch interactive terminal chat with liberated model.")
    p.add_argument("--ui", action="store_true", help="Launch Streamlit GUI dashboard.")
    p.add_argument("--publish", metavar="REPO_ID", help="Hugging Face Hub repository ID to publish model.")
    return p


def run_interactive_chat(model, tokenizer, device):
    """Provides a clean interactive terminal chat loop to test the liberated model."""
    if RICH_AVAILABLE:
        console.print("\n[bold green]=== LatentAblate Interactive Terminal Mode ===[/bold green]")
        console.print("[dim]Type your prompts below to test the liberated model. Type 'exit' or 'quit' to end.[/dim]\n")
    else:
        print("\n=== LatentAblate Interactive Terminal Mode ===")
        print("Type your prompts below. Type 'exit' or 'quit' to end.\n")

    model.eval()
    while True:
        try:
            prompt = input("\033[1;36mUser > \033[0m").strip()
            if not prompt or prompt.lower() in ["exit", "quit"]:
                print("Exiting interactive mode. Goodbye!")
                break

            if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
                messages = [{"role": "user", "content": prompt}]
                formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                formatted = prompt

            inputs = tokenizer(formatted, return_tensors="pt").to(device)
            input_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False, temperature=None, top_p=None)

            response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
            print(f"\033[1;32mModel >\033[0m {response.strip()}\n")
        except KeyboardInterrupt:
            print("\nExiting interactive mode. Goodbye!")
            break
        except Exception as e:
            logger.error(f"Generation error during interactive chat: {e}")


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = build_parser().parse_args(argv)

    if args.ui:
        dashboard_path = Path(__file__).parent / "dashboard.py"
        logger.info("Launching LatentAblate Interactive Web Workbench...")
        subprocess.run(["streamlit", "run", str(dashboard_path)])
        return

    cfg = _load_config(args.config)
    torch.manual_seed(args.seed)
    
    model_path = args.model_path or cfg.get("model_path")
    if not model_path:
        logger.error("No model path specified in arguments or config.")
        sys.exit(2)

    dtype = DTYPES.get(str(cfg.get("torch_dtype", "float16")), torch.float16)
    output_dir = args.output_dir or cfg.get("output_dir", "./out/liberated_model")

    scfg = cfg.get("surgery", {})
    preserve = scfg.get("preserve_capability", True) and not args.no_preserve_capability
    components = scfg.get("target_components", ["attn", "mlp"])
    basis_rank = args.basis_rank or scfg.get("basis_rank", 16)
    layer_window = args.layer_window or scfg.get("layer_window", 2)
    alpha = args.alpha if args.alpha is not None else scfg.get("scaling_factor", 0.75)

    dcfg = cfg.get("datasets", {})
    profile = BehaviorProfiles.get_profile(args.profile or dcfg.get("profile", "safety_uncensored"))
    pos, neg = profile["positive"], profile["negative"]

    logger.info(f"Initializing LatentAblate Hyperion for model: {model_path}")
    surgeon = LatentAblateSurgeon(model_path=model_path, device=cfg.get("device", "auto"), torch_dtype=dtype)

    baseline_ppl = ModelValidator.calculate_perplexity(surgeon.model, surgeon.tokenizer, BehaviorProfiles.CAPABILITY_PROMPTS, surgeon.device)
    logger.info(f"Baseline Evaluation Perplexity: {baseline_ppl:.2f}")

    if args.causal_trace:
        CausalMediationAnalyzer.compute_indirect_effects(surgeon.model, surgeon.tokenizer, pos, neg, surgeon.device)

    if args.target_layer is not None:
        target_layer = args.target_layer
    elif args.auto_sweep or scfg.get("target_layer_idx") is None:
        logger.info("Running LayerSweeper to detect optimal refusal manifold peak...")
        sweeper = LayerSweeper(surgeon.model, surgeon.tokenizer, surgeon.device)
        target_layer = sweeper.find_optimal_layer(pos, neg, output_dir=output_dir)
    else:
        target_layer = scfg.get("target_layer_idx", int(len(surgeon.layers) * 0.6))

    if args.dry_run:
        report = SurgerySimulator.simulate(surgeon, pos, neg, target_layer, basis_rank, components)
        print(yaml.safe_dump(report, sort_keys=False))
        return

    if args.compare:
        logger.info("Capturing base model responses for comparison...")
        test_q = pos[:2]
        base_outputs = {}
        for q in test_q:
            inputs = surgeon.tokenizer(q, return_tensors="pt").to(surgeon.device)
            with torch.no_grad():
                out = surgeon.model.generate(**inputs, max_new_tokens=60, do_sample=False)
            base_outputs[q] = surgeon.tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    baseline_bench = RefusalBenchmark.evaluate_model(surgeon.model, surgeon.tokenizer) if args.benchmark else None

    surgeon.reporter = SurgeryReporter(output_dir)
    surgeon.reporter.log_step("initialization", {"model_path": model_path, "target_layer": target_layer})
    
    surgeon.execute_surgery(
        positive_prompts=pos,
        negative_prompts=neg,
        capability_prompts=BehaviorProfiles.CAPABILITY_PROMPTS,
        target_layer_idx=target_layer,
        layer_window=layer_window,
        basis_rank=basis_rank,
        scaling_factor=alpha,
        preserve_capability=preserve,
        target_components=components,
    )
    surgeon.reporter.log_step("conflict_table", surgeon.conflict_table)

    CapabilityGuardrail.validate_and_rollback(surgeon.model, surgeon.tokenizer, surgeon.device, baseline_ppl, BehaviorProfiles.CAPABILITY_PROMPTS)

    if baseline_bench is not None:
        after_bench = RefusalBenchmark.evaluate_model(surgeon.model, surgeon.tokenizer)
        surgeon.reporter.log_step("benchmark", {"before": baseline_bench, "after": after_bench})
        
        if RICH_AVAILABLE:
            table = Table(title="LatentAblate v5.0 Surgery Performance Report")
            table.add_column("Metric", style="cyan", no_wrap=True)
            table.add_column("Before Surgery", style="red")
            table.add_column("After Surgery", style="green")
            table.add_row("Freedom Score", f"{baseline_bench['freedom_score']:.1f}%", f"{after_bench['freedom_score']:.1f}%")
            console.print(table)

    if args.compare and RICH_AVAILABLE:
        table = Table(title="Side-by-Side Response Comparison")
        table.add_column("Test Prompt", style="yellow")
        table.add_column("Base Model", style="red")
        table.add_column("Liberated Model", style="green")
        
        for q in test_q:
            inputs = surgeon.tokenizer(q, return_tensors="pt").to(surgeon.device)
            with torch.no_grad():
                out = surgeon.model.generate(**inputs, max_new_tokens=60, do_sample=False)
            lib_resp = surgeon.tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            table.add_row(q, base_outputs.get(q, ""), lib_resp)
        console.print(table)

    if args.adversarial_test:
        AdversarialStressTester.stress_test(surgeon.model, surgeon.tokenizer, surgeon.device)

    surgeon.save_model(output_dir)
    surgeon.reporter.save_report()

    if args.publish:
        from latent_ablate.publisher import HuggingFacePublisher
        HuggingFacePublisher.publish_model(output_dir, args.publish)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("LatentAblate Hyperion Pipeline successfully executed, plotted, and exported!")

    if args.interactive:
        run_interactive_chat(surgeon.model, surgeon.tokenizer, surgeon.device)


if __name__ == "__main__":
    main()
