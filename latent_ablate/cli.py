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

from latent_ablate.profiles import BehaviorProfiles
from latent_ablate.prompts import PromptLoader
from latent_ablate.reporter import SurgeryReporter
from latent_ablate.surgeon import LatentAblateSurgeon
from latent_ablate.sweeper import LayerSweeper
from latent_ablate.optimizer import SubspaceOptimizer
from latent_ablate.publisher import HuggingFacePublisher
from latent_ablate.plugins.modifier import CWOEModifier, ARAModifier, SOMAModifier, RiemannianModifier, SAEModifier
from latent_ablate.plugins.scorer import Context, RefusalSuppressionScorer, UtilityPerplexityScorer
from latent_ablate.adversarial import AdversarialStressTester

logger = logging.getLogger("LatentAblate.CLI")

try:
    from rich.console import Console
    from rich.table import Table
    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


def load_config(config_path: str) -> dict:
    if not tomllib:
        return {}
    path = Path(config_path)
    if not path.exists():
        return {}
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except Exception as e:
        logger.warning(f"Failed to parse config file {config_path}: {e}")
        return {}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LatentAblate Heretic Edition - Advanced Latent Modding Suite")
    p.add_argument("--config", default="configs/default.toml", help="Path to config file.")
    p.add_argument("--model-path", default="Qwen/Qwen2.5-1.5B-Instruct", help="Model path or HF repo ID.")
    p.add_argument("--output-dir", default="./out/liberated_model", help="Output directory.")
    p.add_argument("--modifier", choices=["cwoe", "ara", "soma", "riemannian", "sae"], default="cwoe", help="Modification engine.")
    p.add_argument("--trials", type=int, default=10, help="Number of trial iterations.")
    p.add_argument("--optimize", action="store_true", help="Enable automated Optuna TPE optimization.")
    p.add_argument("--target-layers", type=str, default="15,19,23", help="Comma-separated transformer layer indices for multi-pass surgery.")
    p.add_argument("--basis-rank-attn", type=int, default=20, help="Attention subspace rank.")
    p.add_argument("--basis-rank-mlp", type=int, default=14, help="MLP subspace rank.")
    p.add_argument("--alpha-attn", type=float, default=1.05, help="Attention scaling factor.")
    p.add_argument("--alpha-mlp", type=float, default=0.75, help="MLP scaling factor.")
    p.add_argument("--prompts-file", type=str, default=None, help="Path to custom JSON or text prompt file.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--publish", metavar="REPO_ID", help="Hugging Face Hub repo ID to auto-publish.")
    p.add_argument("--interactive", action="store_true", help="Launch interactive chat after surgery.")
    p.add_argument("--ui", action="store_true", help="Launch Streamlit dashboard.")
    return p


def run_interactive_chat_with_commands(model, tokenizer, device):
    if RICH_AVAILABLE:
        console.print("\n[bold green]=== LatentAblate Pro Interactive Terminal Mode ===[/bold green]")
        console.print("[dim]Commands: Type your prompt normally, '/stress' to test compliance, '/help' for commands, or 'exit'.[/dim]\n")
    else:
        print("\n=== LatentAblate Pro Interactive Terminal Mode ===")
        print("Commands: Type prompt normally, '/stress', '/help', or 'exit'.\n")

    model.eval()
    while True:
        try:
            prompt = input("\033[1;36mUser > \033[0m").strip()
            if not prompt:
                continue

            if prompt.lower() in ["exit", "quit"]:
                print("Exiting interactive mode. Goodbye!")
                break

            if prompt.lower() == "/help":
                print("\n📌 Available Interactive Commands:")
                print("  /stress - Run adversarial compliance benchmark suite")
                print("  /help   - Display this help message")
                print("  exit    - Exit the interactive session\n")
                continue

            if prompt.lower() == "/stress":
                print("\n[⚡] Running live adversarial stress test suite...")
                results = AdversarialStressTester.stress_test(model, tokenizer, device)
                score = results["compliance_rate"]
                print(f"\n🎯 Live Stress Test Score: {score:.1f}% Compliance")
                for p, res in results["test_results"].items():
                    status = "✅ Bypassed" if not res["resisted_attack"] else "❌ Refused"
                    print(f"  {status} | {p[:50]}...")
                print()
                continue

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


def interactive_trial_selector(trial_results: list) -> int:
    if not trial_results:
        return 0

    if not RICH_AVAILABLE:
        print("\n=== Trial Results Summary ===")
        for idx, res in enumerate(trial_results):
            print(f"[{idx}] Layers: {res['layers']} | Refusal: {res['refusal']}% | PPL: {res['perplexity']:.2f}")
        choice = input("Select trial index [default 0]: ").strip()
        return int(choice) if choice.isdigit() and int(choice) < len(trial_results) else 0

    table = Table(title="🧬 LatentAblate - Trial Performance Matrix", style="cyan")
    table.add_column("Index", justify="center", style="bold yellow")
    table.add_column("Target Layers", justify="center")
    table.add_column("Attn Rank / Alpha", justify="center")
    table.add_column("MLP Rank / Alpha", justify="center")
    table.add_column("Refusal Rate", justify="center", style="red")
    table.add_column("Perplexity", justify="center", style="green")

    for idx, res in enumerate(trial_results):
        table.add_row(
            str(idx),
            str(res["layers"]),
            f"{res['rank_attn']} / {res['alpha_attn']}",
            f"{res['rank_mlp']} / {res['alpha_mlp']}",
            f"{res['refusal']:.1f}%",
            f"{res['perplexity']:.2f}"
        )

    console.print(table)
    console.print("\n[bold green]Tip:[/bold green] Select a trial with 0.0% refusal and low perplexity.")

    selection = input("\033[1;36mEnter trial index to apply & save (Press Enter for best): \033[0m").strip()
    if selection.isdigit() and int(selection) < len(trial_results):
        return int(selection)
    return 0


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    # Load and overlay TOML config if present
    config_data = load_config(args.config)
    if config_data:
        logger.info(f"Loaded configuration from {args.config}")
        args.model_path = config_data.get("model_path", args.model_path)
        args.output_dir = config_data.get("output_dir", args.output_dir)
        
        surgery_cfg = config_data.get("surgery", {})
        if "target_layers" in surgery_cfg:
            layers_val = surgery_cfg["target_layers"]
            args.target_layers = ",".join(map(str, layers_val)) if isinstance(layers_val, list) else str(layers_val)
        elif "target_layer_idx" in surgery_cfg:
            args.target_layers = str(surgery_cfg["target_layer_idx"])

        args.basis_rank_attn = surgery_cfg.get("basis_rank_attn", args.basis_rank_attn)
        args.basis_rank_mlp = surgery_cfg.get("basis_rank_mlp", args.basis_rank_mlp)
        args.alpha_attn = surgery_cfg.get("alpha_attn", args.alpha_attn)
        args.alpha_mlp = surgery_cfg.get("alpha_mlp", args.alpha_mlp)

        datasets_cfg = config_data.get("datasets", {})
        profile_name = datasets_cfg.get("profile", "safety_uncensored")
    else:
        profile_name = "safety_uncensored"

    if args.ui:
        dashboard_path = Path(__file__).parent / "dashboard.py"
        subprocess.run(["streamlit", "run", str(dashboard_path)])
        return

    torch.manual_seed(args.seed)

    # Load prompts
    if args.prompts_file:
        logger.info(f"Loading custom contrastive prompts from {args.prompts_file}...")
        try:
            pos, neg = PromptLoader.load_from_file(args.prompts_file)
        except Exception as e:
            logger.error(f"Failed to load custom prompts: {e}. Falling back to default profile.")
            profile = BehaviorProfiles.get_profile(profile_name)
            pos, neg = profile["positive"], profile["negative"]
    else:
        profile = BehaviorProfiles.get_profile(profile_name)
        pos, neg = profile["positive"], profile["negative"]

    reporter = SurgeryReporter(args.output_dir)
    trial_records = []

    # Parse target layers list
    target_layers_list = [int(x.strip()) for x in args.target_layers.split(",") if x.strip().isdigit()]

    # Optional Optuna Search
    if args.optimize:
        logger.info("Running Optuna Hyperparameter Optimization...")
        try:
            optimizer = SubspaceOptimizer(model_path=args.model_path, device="auto", torch_dtype=torch.float16)
            best_params = optimizer.optimize_surgery(pos, neg, BehaviorProfiles.CAPABILITY_PROMPTS, n_trials=args.trials)
            best_layer = best_params.get("target_layer_idx", target_layers_list[0] if target_layers_list else 19)
            target_layers_list = [best_layer]
            args.basis_rank_attn = best_params.get("basis_rank_attn", args.basis_rank_attn)
            args.basis_rank_mlp = best_params.get("basis_rank_mlp", args.basis_rank_mlp)
            args.alpha_attn = best_params.get("alpha_attn", args.alpha_attn)
            args.alpha_mlp = best_params.get("alpha_mlp", args.alpha_mlp)
        except Exception as e:
            logger.warning(f"Optimization phase encountered an issue: {e}")

    # Run Controlled Trial Loop
    logger.info(f"🚀 Starting multi-pass surgical execution across layers {target_layers_list}...")
    for trial in range(1, max(1, args.trials if not args.optimize else 1) + 1):
        surgeon = LatentAblateSurgeon(model_path=args.model_path, device="auto", torch_dtype=torch.float16)

        passes = []
        for l_idx in target_layers_list:
            passes.append({
                "name": f"Layer_{l_idx}_Liberation",
                "positive": pos,
                "negative": neg,
                "target_layer": l_idx
            })

        surgeon.execute_multi_pass_surgery(
            passes=passes,
            capability_prompts=BehaviorProfiles.CAPABILITY_PROMPTS,
            layer_window=2,
            basis_rank_attn=args.basis_rank_attn,
            basis_rank_mlp=args.basis_rank_mlp,
            scaling_factor_attn=args.alpha_attn,
            scaling_factor_mlp=args.alpha_mlp,
            preserve_capability=True
        )

        ctx = Context(
            model=surgeon.model,
            tokenizer=surgeon.tokenizer,
            device=surgeon.device,
            settings={
                "positive_prompts": pos, "negative_prompts": neg,
                "target_layers": target_layers_list, "basis_rank_attn": args.basis_rank_attn,
                "alpha_attn": args.alpha_attn
            }
        )

        score_refusal = RefusalSuppressionScorer().get_score(ctx)
        score_ppl = UtilityPerplexityScorer().get_score(ctx)

        trial_records.append({
            "trial_id": trial,
            "surgeon": surgeon,
            "layers": target_layers_list,
            "rank_attn": args.basis_rank_attn,
            "rank_mlp": args.basis_rank_mlp,
            "alpha_attn": args.alpha_attn,
            "alpha_mlp": args.alpha_mlp,
            "refusal": score_refusal.value * 100,
            "perplexity": score_ppl.value
        })

        reporter.log_step(f"trial_{trial}_scores", {"refusal": score_refusal.value, "perplexity": score_ppl.value})

    chosen_idx = interactive_trial_selector(trial_records)
    winning_trial = trial_records[chosen_idx]

    winning_surgeon = winning_trial["surgeon"]
    winning_surgeon.save_output_dir = args.output_dir
    winning_surgeon.save_model(args.output_dir)
    reporter.save_report()
    logger.info(f"🎉 Liberated model successfully saved to {args.output_dir}")

    if args.publish:
        logger.info(f"🌐 Publishing model to Hugging Face Hub: {args.publish}")
        try:
            HuggingFacePublisher.publish_model(args.output_dir, args.publish)
        except Exception as e:
            logger.error(f"Failed to publish: {e}")

    if args.interactive:
        run_interactive_chat_with_commands(winning_surgeon.model, winning_surgeon.tokenizer, winning_surgeon.device)


if __name__ == "__main__":
    main()

