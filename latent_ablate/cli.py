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
from latent_ablate.arch import eligible_linears, resolve_layers
from latent_ablate.diagnostics import composite_trial_score, measure_layer_delta
from latent_ablate.erasure import apply_multi_direction_erasure
from latent_ablate.guardrail import CapabilityGuardrail
from latent_ablate.optimizer import SubspaceOptimizer
from latent_ablate.plugins.scorer import Context, RefusalSuppressionScorer, UtilityPerplexityScorer
from latent_ablate.profiles import BehaviorProfiles
from latent_ablate.prompts import PromptLoader
from latent_ablate.publisher import HuggingFacePublisher
from latent_ablate.refusal_subspace import RefusalSubspaceExtractor
from latent_ablate.reporter import SurgeryReporter, build_reproducibility_manifest
from latent_ablate.scope import AblateScope
from latent_ablate.snapshot import LayerWeightSnapshot
from latent_ablate.surgeon import LatentAblateSurgeon
from latent_ablate.sweeper import LayerSweeper
from latent_ablate.validator import ModelValidator

logger = logging.getLogger("LatentAblate.CLI")

try:
    from rich.console import Console
    from rich.table import Table
    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

REFUSAL_MARKERS = (
    "i'm sorry", "i am sorry", "sorry,", "i cannot", "i can't",
    "i am unable", "i'm unable", "as an ai", "as a language model",
    "fulfill this request", "against my ethical", "against my safety",
)


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
    p.add_argument(
        "--modifier",
        choices=["mpoa", "cwoe", "ara", "soma", "riemannian", "sae"],
        default="mpoa",
        help="Modification engine.",
    )
    p.add_argument(
        "--full-stack",
        action="store_true",
        help="Run stacked multi-tier passes across early-mid, mid, and late layers with tapered scaling.",
    )
    p.add_argument(
        "--subspace-rank",
        type=int,
        default=1,
        help="Rank k (>1) for multi-directional refusal subspace ablation via SVD.",
    )
    p.add_argument(
        "--adaptive-rank",
        action="store_true",
        help="Automatically select refusal subspace rank from centered contrastive singular-value energy.",
    )
    p.add_argument(
        "--auto-sweep",
        action="store_true",
        help="Scan layers with LayerSweeper before surgery to detect the refusal manifold peak.",
    )
    p.add_argument(
        "--guardrail",
        action="store_true",
        help="Enable CapabilityGuardrail automatic rollback if perplexity increases beyond threshold.",
    )
    p.add_argument(
        "--max-ppl-increase",
        type=float,
        default=25.0,
        help="Maximum allowed percentage perplexity increase when --guardrail is enabled.",
    )
    p.add_argument("--scope-out", type=str, default=None, help="Export AblateScope .latentspec.json profile before surgery.")
    p.add_argument("--scope-in", type=str, default=None, help="Load an AblateScope .latentspec.json profile to set layer/alpha.")
    p.add_argument("--trials", type=int, default=1, help="Number of trial iterations.")
    p.add_argument("--optimize", action="store_true", help="Enable automated Optuna TPE optimization.")
    p.add_argument("--target-layers", type=str, default="18", help="Comma-separated transformer layer indices.")
    p.add_argument("--layer-window", type=int, default=2, help="Window radius around each target layer.")
    p.add_argument("--basis-rank-attn", type=int, default=16, help="Attention subspace rank.")
    p.add_argument("--basis-rank-mlp", type=int, default=12, help="MLP subspace rank.")
    p.add_argument("--alpha-attn", type=float, default=1.0, help="Attention scaling factor.")
    p.add_argument("--alpha-mlp", type=float, default=1.0, help="MLP scaling factor.")
    p.add_argument("--profile", type=str, default=None, help="Behavior profile name.")
    p.add_argument("--prompts-file", type=str, default=None, help="Path to custom JSON or text prompt file.")
    p.add_argument("--auto-refine", action="store_true", help="Automatically run an incremental refinement step in chat if a refusal marker is detected.")
    p.add_argument("--max-refines", type=int, default=2, help="Maximum incremental refinement steps per prompt when --auto-refine is enabled.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--save-model", action="store_true", help="Save the selected model and audit report to --output-dir without prompting.")
    p.add_argument("--publish", metavar="REPO_ID", help="Hugging Face Hub repo ID to auto-publish.")
    p.add_argument("--interactive", action="store_true", help="Launch interactive chat after run.")
    p.add_argument("--ui", action="store_true", help="Launch Streamlit dashboard.")
    return p


def save_artifacts(
    surgeon: LatentAblateSurgeon,
    reporter: SurgeryReporter,
    output_dir: str,
    manifest: dict | None = None,
):
    surgeon.save_output_dir = output_dir
    surgeon.save_model(output_dir)
    if manifest is not None:
        reporter.save_manifest(manifest)
    reporter.save_report()
    logger.info(f"Model and report saved to {output_dir}")


def generate_reply(model, tokenizer, device, prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        formatted = prompt

    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)

    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()


def run_multi_direction_refinement(
    surgeon: LatentAblateSurgeon,
    pos: list,
    target_layer: int,
    layer_window: int,
    rank: int = 2,
    step_alpha: float = 0.25,
) -> dict:
    layers = resolve_layers(surgeon.model)
    max_idx = len(layers) - 1
    window_indices = [
        i for i in range(target_layer - layer_window, target_layer + layer_window + 1)
        if 0 <= i <= max_idx
    ]
    if not window_indices:
        window_indices = [max(0, min(max_idx, target_layer))]

    directions = RefusalSubspaceExtractor.extract_prefix_contrast_directions(
        model=surgeon.model,
        tokenizer=surgeon.tokenizer,
        device=surgeon.device,
        prompts=pos[:8],
        layer_idx=window_indices[len(window_indices) // 2],
        rank=rank,
    )
    if not directions:
        return {"layers_updated": 0, "max_relative_change": 0.0}

    max_rel_change = 0.0
    updated_layers = 0
    for idx in window_indices:
        layer_modified = False
        for _name, weight in eligible_linears(layers[idx]):
            before = weight.detach().clone()
            updated = apply_multi_direction_erasure(
                weight, directions, alpha=step_alpha, decay_per_rank=0.75
            )
            with torch.no_grad():
                weight.copy_(updated.to(device=weight.device, dtype=weight.dtype))
            delta = measure_layer_delta(before, weight)
            max_rel_change = max(max_rel_change, delta["relative_change"])
            layer_modified = True
        if layer_modified:
            updated_layers += 1

    return {
        "layers_updated": updated_layers,
        "directions_used": len(directions),
        "max_relative_change": round(max_rel_change, 6),
    }


def run_interactive_chat_with_commands(
    surgeon: LatentAblateSurgeon,
    reporter: SurgeryReporter,
    output_dir: str,
    pos: list,
    target_layer: int,
    layer_window: int,
    auto_refine: bool = False,
    max_refines: int = 2,
    manifest: dict | None = None,
):
    model = surgeon.model
    tokenizer = surgeon.tokenizer
    device = surgeon.device

    max_idx = len(resolve_layers(model)) - 1
    window_indices = [
        i for i in range(target_layer - layer_window, target_layer + layer_window + 1)
        if 0 <= i <= max_idx
    ]
    snapshot = LayerWeightSnapshot()
    snapshot.capture(model, window_indices or [max(0, min(max_idx, target_layer))], store_on_cpu=True)

    if RICH_AVAILABLE:
        console.print("\n[bold green]=== LatentAblate Interactive Terminal Mode ===[/bold green]")
        console.print("[dim]Commands: '/stress', '/refine', '/undo', '/save', '/help', or 'exit'.[/dim]\n")
    else:
        print("\n=== LatentAblate Interactive Terminal Mode ===")
        print("Commands: '/stress', '/refine', '/undo', '/save', '/help', or 'exit'.\n")

    model.eval()
    last_prompt = None

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
                print("  /stress - Run stress test suite")
                print("  /refine - Apply multi-direction prefix-contrast refinement step and re-run last prompt")
                print("  /undo   - Restore window weights to the initial interactive snapshot")
                print("  /save   - Save current model weights and report to disk")
                print("  /help   - Display this help message")
                print("  exit    - Exit the interactive session\n")
                continue

            if prompt.lower() == "/undo":
                snapshot.restore(model)
                reporter.log_step("interactive_snapshot_restore", {"target_layer": target_layer})
                print("Restored weights to initial interactive snapshot.\n")
                if last_prompt:
                    response = generate_reply(model, tokenizer, device, last_prompt)
                    print(f"\033[1;32mModel >\033[0m {response}\n")
                continue

            if prompt.lower() == "/save":
                save_artifacts(surgeon, reporter, output_dir, manifest=manifest)
                print(f"Saved current model, manifest, and audit report to: {output_dir}\n")
                continue

            if prompt.lower() == "/refine":
                print("Applying multi-direction refinement step...")
                stats = run_multi_direction_refinement(
                    surgeon, pos, target_layer, layer_window, rank=2, step_alpha=0.25
                )
                reporter.log_step("interactive_multi_dir_refine", stats)
                print(
                    f"Updated {stats['layers_updated']} layer(s) across {stats['directions_used']} direction(s) "
                    f"(max relative delta: {stats['max_relative_change']})."
                )
                if last_prompt:
                    response = generate_reply(model, tokenizer, device, last_prompt)
                    print(f"\033[1;32mModel >\033[0m {response}\n")
                else:
                    print("Refinement applied. Enter a prompt to test.\n")
                continue

            if prompt.lower() == "/stress":
                print("\nRunning live stress test suite...")
                results = AdversarialStressTester.stress_test(model, tokenizer, device)
                score = results["compliance_rate"]
                print(f"\nLive Stress Test Score: {score:.1f}% Compliance")
                for p, res in results["test_results"].items():
                    status = "Bypassed" if not res["resisted_attack"] else "Refused"
                    print(f"  [{status}] | {p[:50]}...")
                print()
                continue

            last_prompt = prompt
            response = generate_reply(model, tokenizer, device, prompt)

            if auto_refine:
                attempts = 0
                while attempts < max_refines and any(m in response.lower() for m in REFUSAL_MARKERS):
                    attempts += 1
                    print(f"\033[1;33m[Auto-Refine {attempts}/{max_refines}]\033[0m Refining layer {target_layer} and retrying...")
                    stats = run_multi_direction_refinement(
                        surgeon, pos, target_layer, layer_window, rank=2, step_alpha=0.25
                    )
                    reporter.log_step(f"auto_refine_{attempts}", stats)
                    response = generate_reply(model, tokenizer, device, prompt)

            print(f"\033[1;32mModel >\033[0m {response}\n")
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
        key=lambda i: composite_trial_score(
            refusal_rate=float(trial_results[i].get("refusal", 100.0)) / 100.0,
            perplexity=float(trial_results[i].get("perplexity", 100.0)),
            kl_divergence=float(trial_results[i].get("kl_divergence", 0.0)),
        ),
    )
    return best_idx


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = build_parser()
    raw_argv = argv if argv is not None else sys.argv[1:]
    args = parser.parse_args(raw_argv)

    config_data = load_config(args.config)
    profile_name = "safety_uncensored"
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

        if "--layer-window" not in raw_argv:
            args.layer_window = surgery_cfg.get("layer_window", args.layer_window)
        if "--basis-rank-attn" not in raw_argv:
            args.basis_rank_attn = surgery_cfg.get("basis_rank_attn", args.basis_rank_attn)
        if "--basis-rank-mlp" not in raw_argv:
            args.basis_rank_mlp = surgery_cfg.get("basis_rank_mlp", args.basis_rank_mlp)
        if "--alpha-attn" not in raw_argv:
            args.alpha_attn = surgery_cfg.get("alpha_attn", args.alpha_attn)
        if "--alpha-mlp" not in raw_argv:
            args.alpha_mlp = surgery_cfg.get("alpha_mlp", args.alpha_mlp)
        if "--full-stack" not in raw_argv:
            args.full_stack = bool(surgery_cfg.get("full_stack", args.full_stack))
        if "--subspace-rank" not in raw_argv:
            args.subspace_rank = int(surgery_cfg.get("subspace_rank", args.subspace_rank))
        if "--adaptive-rank" not in raw_argv:
            args.adaptive_rank = bool(surgery_cfg.get("adaptive_rank", args.adaptive_rank))

        datasets_cfg = config_data.get("datasets", {})
        profile_name = datasets_cfg.get("profile", profile_name)

    if args.profile:
        profile_name = args.profile

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
            logger.error(f"Failed to load custom prompts: {e}. Falling back to profile '{profile_name}'.")
            profile = BehaviorProfiles.get_profile(profile_name)
            pos, neg = profile["positive"], profile["negative"]
    else:
        profile = BehaviorProfiles.get_profile(profile_name)
        pos, neg = profile["positive"], profile["negative"]

    reporter = SurgeryReporter(args.output_dir)
    trial_records = []
    target_layers_list = [int(x.strip()) for x in args.target_layers.split(",") if x.strip().isdigit()]

    if args.scope_in:
        spec = AblateScope.load_spec(args.scope_in)
        rec_layer = spec.get("recommended_layer")
        rec_alpha = spec.get("recommended_alpha")
        if rec_layer is not None:
            target_layers_list = [int(rec_layer)]
        if rec_alpha is not None:
            args.alpha_attn = float(rec_alpha)
            args.alpha_mlp = round(float(rec_alpha) * 0.9, 4)
        logger.info(f"Loaded AblateScope spec from {args.scope_in}: layer={target_layers_list}, alpha={args.alpha_attn}")

    if args.optimize:
        logger.info("Running Optuna Hyperparameter Optimization...")
        try:
            optimizer = SubspaceOptimizer(
                model_path=args.model_path,
                device="auto",
                torch_dtype=torch.float16,
                local_files_only=args.local_only,
            )
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
        f"(full_stack={args.full_stack}, subspace_rank={args.subspace_rank}, "
        f"adaptive_rank={args.adaptive_rank}, alpha_attn={args.alpha_attn:.2f}, alpha_mlp={args.alpha_mlp:.2f})..."
    )

    for trial in range(1, args.trials + 1):
        alpha_step = 0.1 * (trial - 1) if (args.trials > 1 and not args.optimize) else 0.0
        cur_alpha_attn = max(0.0, round(args.alpha_attn + alpha_step, 4))
        cur_alpha_mlp = max(0.0, round(args.alpha_mlp + alpha_step, 4))

        surgeon = LatentAblateSurgeon(
            model_path=args.model_path,
            device="auto",
            torch_dtype=torch.float16,
            local_files_only=args.local_only,
        )

        if args.scope_out and trial == 1:
            scope_report = AblateScope.map_layers(
                surgeon.model,
                surgeon.tokenizer,
                surgeon.device,
                pos,
                neg,
                capability_prompts=BehaviorProfiles.CAPABILITY_PROMPTS,
                basis_rank=args.basis_rank_attn,
            )
            AblateScope.save_spec(scope_report, args.scope_out)
            if "--target-layers" not in raw_argv and not args.scope_in:
                target_layers_list = [int(scope_report["recommended_layer"])]

        if args.auto_sweep and trial == 1:
            sweeper = LayerSweeper(surgeon.model, surgeon.tokenizer, surgeon.device)
            peak_layer = sweeper.find_optimal_layer(pos, neg, output_dir=args.output_dir)
            target_layers_list = [peak_layer]
            reporter.log_step("auto_sweep_peak", {"peak_layer": peak_layer})

        baseline_ppl = None
        param_snapshot = None
        if args.guardrail:
            baseline_ppl = ModelValidator.calculate_perplexity(
                surgeon.model, surgeon.tokenizer, BehaviorProfiles.CAPABILITY_PROMPTS, surgeon.device
            )
            param_snapshot = CapabilityGuardrail.snapshot_parameters(surgeon.model)

        max_layer = len(surgeon.layers) - 1
        valid_layers = [l for l in target_layers_list if 0 <= l <= max_layer]
        if not valid_layers:
            valid_layers = [len(surgeon.layers) // 2]

        if args.full_stack:
            n_layers = len(surgeon.layers)
            tier_specs = [
                (max(1, int(n_layers * 0.35)), max(2, n_layers // 8), min(1.5, cur_alpha_attn * 0.65)),
                (valid_layers[0], max(2, args.layer_window), cur_alpha_attn),
                (max(1, int(n_layers * 0.75)), max(2, n_layers // 8), min(1.5, cur_alpha_attn * 0.60)),
            ]
            surgeon.execute_full_stack_surgery(
                positive_prompts=pos,
                negative_prompts=neg,
                capability_prompts=BehaviorProfiles.CAPABILITY_PROMPTS,
                tier_specs=tier_specs,
                basis_rank_attn=args.basis_rank_attn,
                basis_rank_mlp=args.basis_rank_mlp,
                target_components=["attn", "mlp"],
                preserve_capability=True,
                calibrate_norms=True,
                subspace_rank=args.subspace_rank,
                adaptive_rank=args.adaptive_rank,
            )
        else:
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
                calibrate_norms=True,
                subspace_rank=args.subspace_rank,
                adaptive_rank=args.adaptive_rank,
            )

        if args.guardrail:
            passed_guardrail = CapabilityGuardrail.validate_and_rollback(
                surgeon.model,
                surgeon.tokenizer,
                surgeon.device,
                baseline_ppl=baseline_ppl,
                capability_texts=BehaviorProfiles.CAPABILITY_PROMPTS,
                max_ppl_increase_pct=args.max_ppl_increase,
                snapshot=param_snapshot,
            )
            reporter.log_step(f"trial_{trial}_guardrail", {"passed": passed_guardrail})

        ctx = Context(
            model=surgeon.model,
            tokenizer=surgeon.tokenizer,
            device=surgeon.device,
            settings={
                "positive_prompts": pos,
                "negative_prompts": neg,
                "capability_texts": BehaviorProfiles.CAPABILITY_PROMPTS,
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
    primary_layer = winning_trial["layers"][0]

    manifest = build_reproducibility_manifest(
        model_path=args.model_path,
        target_layers=winning_trial["layers"],
        alpha_attn=winning_trial["alpha_attn"],
        alpha_mlp=winning_trial["alpha_mlp"],
        curve="cosine",
        best_trial={
            "trial_id": winning_trial["trial_id"],
            "refusal_rate_pct": winning_trial["refusal"],
            "perplexity": winning_trial["perplexity"],
        },
        extra={
            "modifier": args.modifier,
            "full_stack": args.full_stack,
            "subspace_rank": args.subspace_rank,
            "adaptive_rank": args.adaptive_rank,
            "profile": profile_name,
            "seed": args.seed,
        },
    )

    saved = False
    if args.save_model:
        save_artifacts(winning_surgeon, reporter, args.output_dir, manifest=manifest)
        saved = True
    else:
        print("\nSave Options:")
        print(f"  [1] Save model, manifest & report to {args.output_dir} and continue")
        print("  [2] Keep in memory for testing only (do not save)")
        print(f"  [3] Save model, manifest & report to {args.output_dir} and exit")
        save_choice = input("\033[1;36mChoose option [1/2/3, default 2]: \033[0m").strip()
        if save_choice == "1":
            save_artifacts(winning_surgeon, reporter, args.output_dir, manifest=manifest)
            saved = True
        elif save_choice == "3":
            save_artifacts(winning_surgeon, reporter, args.output_dir, manifest=manifest)
            return

    if args.publish:
        if not saved:
            logger.warning("Publishing requires saving the model first.")
        else:
            logger.info(f"Publishing model to Hugging Face Hub: {args.publish}")
            try:
                HuggingFacePublisher.publish_model(args.output_dir, args.publish)
            except Exception as e:
                logger.error(f"Failed to publish: {e}")

    if args.interactive:
        run_interactive_chat_with_commands(
            surgeon=winning_surgeon,
            reporter=reporter,
            output_dir=args.output_dir,
            pos=pos,
            target_layer=primary_layer,
            layer_window=args.layer_window,
            auto_refine=args.auto_refine,
            max_refines=args.max_refines,
            manifest=manifest,
        )


if __name__ == "__main__":
    main()
