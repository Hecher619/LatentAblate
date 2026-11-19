import argparse
import gc
import logging
import subprocess
import sys
from pathlib import Path
import torch

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None

from latent_ablate.adversarial import AdversarialStressTester
from latent_ablate.optimizer import SubspaceOptimizer
from latent_ablate.plugins.scorer import Context, RefusalSuppressionScorer, UtilityPerplexityScorer
from latent_ablate.profiles import BehaviorProfiles
from latent_ablate.prompts import PromptLoader
from latent_ablate.publisher import HuggingFacePublisher
from latent_ablate.reporter import SurgeryReporter
from latent_ablate.surgeon import LatentAblateSurgeon

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
    p = argparse.ArgumentParser(description="LatentAblate - Latent Representation Modification Suite")
    p.add_argument("--config", default="configs/default.toml", help="Path to config file.")
    p.add_argument("--model-path", default="Qwen/Qwen2.5-1.5B-Instruct", help="Model path or HF repo ID.")
    p.add_argument("--local-only", action="store_true", help="Load models and tokenizers strictly from local cache/disk.")
    p.add_argument("--output-dir", default="./out/liberated_model", help="Output directory.")
    p.add_argument("--modifier", choices=["cwoe", "ara", "soma", "riemannian", "sae"], default="cwoe", help="Modification engine.")
    p.add_argument("--trials", type=int, default=1, help="Number of trial iterations.")
    p.add_argument("--optimize", action="store_true", help="Enable automated Optuna TPE optimization.")
    p.add_argument("--target-layers", type=str, default="14,18,22", help="Comma-separated transformer layer indices for multi-pass surgery.")
    p.add_argument("--layer-window", type=int, default=3, help="Window radius around each target layer.")
    p.add_argument("--basis-rank-attn", type=int, default=16, help="Attention subspace rank.")
    p.add_argument("--basis-rank-mlp", type=int, default=12, help="MLP subspace rank.")
    p.add_argument("--alpha-attn", type=float, default=1.0, help="Attention scaling factor.")
    p.add_argument("--alpha-mlp", type=float, default=1.0, help="MLP scaling factor.")
    p.add_argument("--prompts-file", type=str, default=None, help="Path to custom JSON or text prompt file.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--save-model", action="store_true", help="Save the selected model and audit report to --output-dir.")
    p.add_argument("--publish", metavar="REPO_ID", help="Hugging Face Hub repo ID to auto-publish.")
    p.add_argument("--interactive", action="store_true", help="Launch interactive chat after run.")
    p.add_argument("--ui", action="store_true", help="Launch Streamlit dashboard.")
    return p


def run_interactive_chat_with_commands(model, tokenizer, device):
    if RICH_AVAILABLE:
        console.print("\n[bold green]=== LatentAblate Interactive Terminal Mode ===[/bold green]")
        console.print("[dim]Commands: Type your prompt normally, '/stress' to test compliance, '/help' for commands, or 'exit'.[/dim]\n")
    else:
        print("\n=== LatentAblate Interactive Terminal Mode ===")
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
                print("\nAvailable Interactive Commands:")
                print("  /stress - Run adversarial compliance benchmark suite")
                print("  /help   - Display this help message")
                print("  exit    - Exit the interactive session\n")
                continue

            if prompt.lower() == "/stress":
                print("\nRunning live adversarial stress test suite...")
                results = AdversarialStressTester.stress_test(model, tokenizer, device)
                score = results["compliance_rate"]
                print(f"\nLive Stress Test Score: {score:.1f}% Compliance")
                for p, res in results["test_results"].items():
                    status = "Bypassed" if not res["resisted_attack"] else "Refused"
                    print(f"  [{status}] | {p[:50]}...")
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

    table = Table(title="LatentAblate - Trial Performance Matrix", style="cyan")
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
            f"{res['rank_attn']} / {res['alpha_attn']:.4f}",
            f"{res['rank_mlp']} / {res['alpha_mlp']:.4f}",
            f"{res['refusal']:.1f}%",
            f"{res['perplexity']:.2f}",
        )

    console.print(table)
    selection = input("\033[1;36mEnter trial index to select (Press Enter for best): \033[0m").strip()
    if selection.isdigit() and int(selection) < len(trial_results):
        return int(selection)

    best_idx = min(
        range(len(trial_results)),
        key=lambda i: (trial_results[i]["refusal"], trial_results[i]["perplexity"]),
    )
    return best_idx


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = build_parser()
    raw_argv = argv if argv is not None else sys.argv[1:]
    args = parser.parse_args(raw_argv)

    config_data = load_config(args.config)
    if config_data:
        logger.info(f"Loaded base configuration from {args.config}")
        if "--model-path" not in raw_argv:
            args.model_path = config_data.get("model_path", args.model_path)
        if "--output-dir" not in raw_argv:
            args.output_dir = config_data.get("output_dir", args.output_dir)

        surgery_cfg = config_data.get("surgery", {})
        if "--target-layers" not in raw_argv:
            if "target_layers" in surgery_cfg:
                layers_val = surgery_cfg["target_layers"]
                args.target_layers = ",".join(map(str, layers_val)) if isinstance(layers_val, list) else str(layers_val)
            elif "target_layer_idx" in surgery_cfg:
                args.target_layers = str(surgery_cfg["target_layer_idx"])

        if "--basis-rank-attn" not in raw_argv:
            args.basis_rank_attn = surgery_cfg.get("basis_rank_attn", args.basis_rank_attn)
        if "--basis-rank-mlp" not in raw_argv:
            args.basis_rank_mlp = surgery_cfg.get("basis_rank_mlp", args.basis_rank_mlp)
        if "--alpha-attn" not in raw_argv:
            args.alpha_attn = surgery_cfg.get("alpha_attn", args.alpha_attn)
        if "--alpha-mlp" not in raw_argv:
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
    target_layers_list = [int(x.strip()) for x in args.target_layers.split(",") if x.strip().isdigit()]

    if args.optimize:
        logger.info("Running Optuna Hyperparameter Optimization...")
        try:
            optimizer = SubspaceOptimizer(model_path=args.model_path, device="auto", torch_dtype=torch.float16)
            best_params = optimizer.optimize_surgery(
                pos, neg, BehaviorProfiles.CAPABILITY_PROMPTS, n_trials=max(args.trials, 6)
            )
            best_layer = best_params.get("target_layer_idx")
            if best_layer is not None:
                target_layers_list = [best_layer]
            args.layer_window = best_params.get("layer_window", args.layer_window)
            args.basis_rank_attn = best_params.get("basis_rank_attn", args.basis_rank_attn)
            args.basis_rank_mlp = best_params.get("basis_rank_mlp", args.basis_rank_mlp)
            args.alpha_attn = best_params.get("alpha_attn", args.alpha_attn)
            args.alpha_mlp = best_params.get("alpha_mlp", args.alpha_mlp)
        except Exception as e:
            logger.warning(f"Optimization phase encountered an issue: {e}")

    logger.info(
        f"Starting multi-pass surgical execution across layers {target_layers_list} "
        f"(alpha_attn={args.alpha_attn:.2f}, alpha_mlp={args.alpha_mlp:.2f})..."
    )

    for trial in range(1, args.trials + 1):
        alpha_step = 0.1 * (trial - 1) if (args.trials > 1 and not args.optimize) else 0.0
        cur_alpha_attn = min(1.0, round(args.alpha_attn + alpha_step, 4))
        cur_alpha_mlp = min(1.0, round(args.alpha_mlp + alpha_step, 4))

        surgeon = LatentAblateSurgeon(
            model_path=args.model_path,
            device="auto",
            torch_dtype=torch.float16,
            local_files_only=args.local_only,
        )

        max_layer = len(surgeon.layers) - 1
        valid_layers = [l for l in target_layers_list if 0 <= l <= max_layer]
        if not valid_layers:
            valid_layers = [len(surgeon.layers) // 2]

        passes = [
            {
                "name": f"Layer_{l_idx}_Pass",
                "positive": pos,
                "negative": neg,
                "target_layer": l_idx,
            }
            for l_idx in valid_layers
        ]

        surgeon.execute_multi_pass_surgery(
            passes=passes,
            capability_prompts=BehaviorProfiles.CAPABILITY_PROMPTS,
            layer_window=args.layer_window,
            basis_rank_attn=args.basis_rank_attn,
            basis_rank_mlp=args.basis_rank_mlp,
            scaling_factor_attn=cur_alpha_attn,
            scaling_factor_mlp=cur_alpha_mlp,
            target_components=["attn", "mlp"],
            preserve_capability=True,
        )

        ctx = Context(
            model=surgeon.model,
            tokenizer=surgeon.tokenizer,
            device=surgeon.device,
            settings={
                "positive_prompts": pos,
                "negative_prompts": neg,
                "target_layers": valid_layers,
                "basis_rank_attn": args.basis_rank_attn,
                "alpha_attn": cur_alpha_attn,
            },
        )

        score_refusal = RefusalSuppressionScorer().get_score(ctx)
        score_ppl = UtilityPerplexityScorer().get_score(ctx)

        trial_records.append({
            "trial_id": trial,
            "surgeon": surgeon,
            "layers": valid_layers,
            "rank_attn": args.basis_rank_attn,
            "rank_mlp": args.basis_rank_mlp,
            "alpha_attn": cur_alpha_attn,
            "alpha_mlp": cur_alpha_mlp,
            "refusal": score_refusal.value * 100,
            "perplexity": score_ppl.value,
        })

        reporter.log_step(
            f"trial_{trial}_scores",
            {"refusal": score_refusal.value, "perplexity": score_ppl.value},
        )

        if trial < args.trials:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    chosen_idx = interactive_trial_selector(trial_records)
    winning_trial = trial_records[chosen_idx]
    winning_surgeon = winning_trial["surgeon"]

    if args.save_model:
        winning_surgeon.save_output_dir = args.output_dir
        winning_surgeon.save_model(args.output_dir)
        reporter.save_report()
        logger.info(f"Model and report saved to {args.output_dir}")
    else:
        logger.info("Skipping disk export (pass --save-model to write weights and report).")

    if args.publish:
        if not args.save_model:
            logger.warning("Publishing requires --save-model so files exist in --output-dir.")
        else:
            logger.info(f"Publishing model to Hugging Face Hub: {args.publish}")
            try:
                HuggingFacePublisher.publish_model(args.output_dir, args.publish)
            except Exception as e:
                logger.error(f"Failed to publish: {e}")

    if args.interactive:
        run_interactive_chat_with_commands(
            winning_surgeon.model, winning_surgeon.tokenizer, winning_surgeon.device
        )


if __name__ == "__main__":
    main()
