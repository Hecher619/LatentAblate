import argparse
import logging
import sys
import subprocess
from pathlib import Path
import torch

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None

from latent_ablate.arch import resolve_layers
from latent_ablate.profiles import BehaviorProfiles
from latent_ablate.reporter import SurgeryReporter
from latent_ablate.surgeon import LatentAblateSurgeon
from latent_ablate.sweeper import LayerSweeper
from latent_ablate.optimizer import SubspaceOptimizer
from latent_ablate.publisher import HuggingFacePublisher
from latent_ablate.plugins.modifier import CWOEModifier, ARAModifier, SOMAModifier, RiemannianModifier, SAEModifier
from latent_ablate.plugins.scorer import Context, RefusalSuppressionScorer, UtilityPerplexityScorer

logger = logging.getLogger("LatentAblate.CLI")
DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}

try:
    from rich.console import Console
    from rich.table import Table
    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LatentAblate Heretic Edition - Advanced Latent Modding Suite")
    p.add_argument("--config", default="configs/default.toml", help="Path to config file.")
    p.add_argument("--model-path", default="Qwen/Qwen2.5-1.5B-Instruct", help="Model path or HF repo ID.")
    p.add_argument("--output-dir", default="./out/liberated_model", help="Output directory.")
    p.add_argument("--modifier", choices=["cwoe", "ara", "soma", "riemannian", "sae"], default="cwoe", help="Modification engine.")
    p.add_argument("--trials", type=int, default=3, help="Number of iterative trial loops.")
    p.add_argument("--optimize", action="store_true", help="Enable automated Optuna TPE hyperparameter optimization across trials.")
    p.add_argument("--basis-rank", type=int, default=16, help="Subspace rank.")
    p.add_argument("--alpha", type=float, default=0.75, help="Scaling factor (alpha).")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--publish", metavar="REPO_ID", help="Hugging Face Hub repository ID to auto-publish model.")
    p.add_argument("--interactive", action="store_true", help="Launch interactive chat after trials.")
    p.add_argument("--ui", action="store_true", help="Launch Streamlit dashboard.")
    return p


def run_interactive_chat(model, tokenizer, device):
    if RICH_AVAILABLE:
        console.print("\n[bold green]=== LatentAblate Interactive Terminal Mode ===[/bold green]")
        console.print("[dim]Type your prompts below. Type 'exit' or 'quit' to end.[/dim]\n")
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
                outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)

            response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
            print(f"\033[1;32mModel >\033[0m {response.strip()}\n")
        except KeyboardInterrupt:
            print("\nExiting interactive mode. Goodbye!")
            break
        except Exception as e:
            logger.error(f"Generation error: {e}")


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = build_parser().parse_args(argv)

    if args.ui:
        dashboard_path = Path(__file__).parent / "dashboard.py"
        subprocess.run(["streamlit", "run", str(dashboard_path)])
        return

    torch.manual_seed(args.seed)
    profile = BehaviorProfiles.get_profile("safety_uncensored")
    pos, neg = profile["positive"], profile["negative"]

    # Optional Optuna TPE Hyperparameter Search
    alpha = args.alpha
    basis_rank = args.basis_rank
    if args.optimize:
        logger.info("Running automated Optuna TPE Hyperparameter Search...")
        try:
            optimizer = SubspaceOptimizer(model_path=args.model_path, device="auto", torch_dtype=torch.float16)
            best_params = optimizer.optimize_surgery(positive_prompts=pos, negative_prompts=neg, capability_prompts=BehaviorProfiles.CAPABILITY_PROMPTS, n_trials=args.trials)
            alpha = best_params.get("scaling_factor", alpha)
            basis_rank = best_params.get("basis_rank", basis_rank)
            logger.info(f"Optuna Best Parameters -> Alpha: {alpha}, Rank: {basis_rank}")
        except Exception as e:
            logger.warning(f"Optuna optimization skipped due to missing dependency/error: {e}")

    modifier_map = {
        "cwoe": CWOEModifier(),
        "ara": ARAModifier(),
        "soma": SOMAModifier(),
        "riemannian": RiemannianModifier(),
        "sae": SAEModifier()
    }
    active_modifier = modifier_map[args.modifier]
    logger.info(f"Selected Modding Engine: {active_modifier.__class__.__name__}")

    reporter = SurgeryReporter(args.output_dir)

    # Run Iterative Trial Loop
    for trial in range(1, args.trials + 1):
        logger.info(f"==================================================")
        logger.info(f"🚀 Starting Trial [{trial}/{args.trials}] using {args.modifier.upper()}")
        logger.info(f"==================================================")

        surgeon = LatentAblateSurgeon(model_path=args.model_path, device="auto", torch_dtype=torch.float16)
        sweeper = LayerSweeper(surgeon.model, surgeon.tokenizer, surgeon.device)
        target_layer = sweeper.find_optimal_layer(pos, neg, output_dir=args.output_dir)

        ctx = Context(
            model=surgeon.model,
            tokenizer=surgeon.tokenizer,
            device=surgeon.device,
            settings={
                "positive_prompts": pos,
                "negative_prompts": neg,
                "capability_prompts": BehaviorProfiles.CAPABILITY_PROMPTS,
                "target_layer_idx": target_layer,
                "basis_rank": basis_rank,
                "scaling_factor": alpha,
                "cluster_count": 4
            }
        )

        active_modifier.modify_model(ctx, surgeon)

        refusal_scorer = RefusalSuppressionScorer()
        perplexity_scorer = UtilityPerplexityScorer()

        score_refusal = refusal_scorer.get_score(ctx)
        score_ppl = perplexity_scorer.get_score(ctx)

        logger.info(f"📊 Trial {trial} Results -> {score_refusal.md_display} | {score_ppl.md_display}")
        reporter.log_step(f"trial_{trial}_scores", {"refusal": score_refusal.value, "perplexity": score_ppl.value})

        if trial == args.trials:
            surgeon.save_model(args.output_dir)
            reporter.save_report()
            logger.info(f"🎉 All trials completed successfully! Liberated model saved to {args.output_dir}")

            if args.publish:
                logger.info(f"🌐 Publishing liberated model to Hugging Face Hub: {args.publish}")
                try:
                    HuggingFacePublisher.publish_model(args.output_dir, args.publish)
                except Exception as e:
                    logger.error(f"Failed to publish to Hugging Face: {e}")

            if args.interactive:
                run_interactive_chat(surgeon.model, surgeon.tokenizer, surgeon.device)


if __name__ == "__main__":
    main()
