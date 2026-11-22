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
from latent_ablate.arch import (
    REFUSAL_MARKERS,
    eligible_linears,
    format_chat_prompt,
    is_hedging_text,
    is_refusal_text,
    layer_index_from_ratio,
    resolve_layers,
    resolve_torch_dtype,
)
from latent_ablate.diagnostics import composite_trial_score, measure_layer_delta
from latent_ablate.erasure import apply_multi_direction_erasure
from latent_ablate.guardrail import CapabilityGuardrail
from latent_ablate.optimizer import SubspaceOptimizer
from latent_ablate.plugins.scorer import Context, RefusalSuppressionScorer, UtilityPerplexityScorer
from latent_ablate.profiles import BehaviorProfiles
from latent_ablate.prompts import PromptLoader
from latent_ablate.publisher import HuggingFacePublisher
from latent_ablate.reaction import classify_response, format_velora_telemetry
from latent_ablate.refusal_subspace import RefusalSubspaceExtractor
from latent_ablate.reporter import SurgeryReporter, build_reproducibility_manifest
from latent_ablate.scope import AblateScope
from latent_ablate.snapshot import LayerWeightSnapshot
from latent_ablate.surgeon import LatentAblateSurgeon
from latent_ablate.sweeper import LayerSweeper
from latent_ablate.validator import ModelValidator
from latent_ablate.velora import VeloraEngine

logger = logging.getLogger("LatentAblate.CLI")

try:
    from rich.console import Console
    from rich.table import Table
    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

PRESET_DEFAULTS = {
    "gentle": {"alpha_attn": 0.90, "alpha_mlp": 0.80, "layer_window": 2},
    "balanced": {"alpha_attn": 1.35, "alpha_mlp": 1.15, "layer_window": 2},
    "strong": {"alpha_attn": 1.80, "alpha_mlp": 1.55, "layer_window": 3},
}


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
    p.add_argument("positional_model", nargs="?", default=None, help="Optional model path or HF repo ID (zero-config mode).")
    p.add_argument("--config", default="configs/default.toml", help="Path to config file.")
    p.add_argument("--model-path", default="Qwen/Qwen2.5-1.5B-Instruct", help="Model path or HF repo ID.")
    p.add_argument("--local-only", action="store_true", help="Load models and tokenizers strictly from local cache/disk.")
    p.add_argument("--load-in-4bit", action="store_true", help="Load target model in 4-bit NF4 quantization via bitsandbytes.")
    p.add_argument("--dry-run", action="store_true", help="Print resolved layer depth, default target layers, and plan without mutating weights.")
    p.add_argument("--output-dir", default="./out/liberated_model", help="Output directory.")
    p.add_argument(
        "--preset",
        choices=["gentle", "balanced", "strong"],
        default="balanced",
        help="Friendly intensity preset for alpha and window (overridden by explicit flags).",
    )
    p.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
        help="Model tensor dtype ('auto' selects bfloat16 when supported on CUDA).",
    )
    p.add_argument(
        "--modifier",
        choices=["mpoa", "cwoe", "velora", "ara", "soma", "riemannian", "sae"],
        default="velora",
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
    p.add_argument("--target-layers", type=str, default=None, help="Comma-separated layer indices (default: ~62%% of model depth).")
    p.add_argument("--layer-window", type=int, default=None, help="Window radius around each target layer.")
    p.add_argument("--basis-rank-attn", type=int, default=16, help="Attention subspace rank.")
    p.add_argument("--basis-rank-mlp", type=int, default=12, help="MLP subspace rank.")
    p.add_argument("--alpha-attn", type=float, default=None, help="Attention scaling factor.")
    p.add_argument("--alpha-mlp", type=float, default=None, help="MLP scaling factor.")
    p.add_argument("--profile", type=str, default=None, help="Behavior profile name.")
    p.add_argument("--prompts-file", type=str, default=None, help="Path to custom JSON or text prompt file.")
    p.add_argument("--no-anti-hedge", action="store_true", help="Disable late-layer hedging/disclaimer contrast suppression in VELORA.")
    p.add_argument("--auto-refine", action="store_true", help="Automatically run an escalating multi-layer refinement step in chat if a refusal, moralizing lecture, or deflection is detected.")
    p.add_argument("--max-refines", type=int, default=3, help="Maximum incremental refinement steps per prompt when --auto-refine is enabled.")
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


def generate_reply(model, tokenizer, device, prompt: str, response_prefix: str = "") -> str:
    formatted = format_chat_prompt(tokenizer, prompt, response_prefix=response_prefix)
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
    rank: int = 4,
    step_alpha: float = 0.35,
    escalation_step: int = 0,
    include_hedging: bool = True,
) -> dict:
    layers = resolve_layers(surgeon.model)
    max_idx = len(layers) - 1

    eff_window = min(max(layer_window + escalation_step, layer_window + 1), max(2, max_idx // 4))
    eff_alpha = min(0.95, float(step_alpha) * (1.0 + 0.40 * max(0, escalation_step)))

    primary_indices = [
        i for i in range(target_layer - eff_window, target_layer + eff_window + 1)
        if 0 <= i <= max_idx
    ]
    late_center = min(max_idx, max(target_layer + 2, layer_index_from_ratio(len(layers), 0.78)))
    late_indices = (
        [i for i in range(late_center - 1, late_center + 2) if 0 <= i <= max_idx]
        if escalation_step >= 1
        else []
    )

    window_indices = sorted(set(primary_indices + late_indices))
    if not window_indices:
        window_indices = [max(0, min(max_idx, target_layer))]

    center_layer = target_layer if 0 <= target_layer <= max_idx else window_indices[len(window_indices) // 2]
    directions = []
    try:
        prefix_dirs = RefusalSubspaceExtractor.extract_prefix_contrast_directions(
            model=surgeon.model,
            tokenizer=surgeon.tokenizer,
            device=surgeon.device,
            prompts=pos[:12],
            layer_idx=center_layer,
            rank=rank,
        )
        directions.extend(list(prefix_dirs))
        if include_hedging and hasattr(RefusalSubspaceExtractor, "extract_hedging_contrast_directions"):
            hedge_dirs = RefusalSubspaceExtractor.extract_hedging_contrast_directions(
                model=surgeon.model,
                tokenizer=surgeon.tokenizer,
                device=surgeon.device,
                prompts=pos[:12],
                layer_idx=center_layer,
                rank=max(2, min(4, rank)),
            )
            directions.extend(list(hedge_dirs))
    except (NotImplementedError, RuntimeError) as e:
        logger.warning(f"Prefix/hedging contrast refinement fallback at layer {center_layer}: {e}")
        directions = []

    if not directions:
        return {
            "layers_updated": 0,
            "directions_used": 0,
            "max_relative_change": 0.0,
            "layer_span": f"{window_indices[0]}..{window_indices[-1]}",
            "effective_alpha": round(eff_alpha, 4),
            "lbfgs_polished_layers": 0,
        }

    max_rel_change = 0.0
    updated_layers = 0
    for idx in window_indices:
        layer_modified = False
        for _name, weight in eligible_linears(layers[idx]):
            if getattr(weight, "is_meta", False):
                continue
            before = weight.detach().clone()
            updated = apply_multi_direction_erasure(
                weight, directions, alpha=eff_alpha, decay_per_rank=0.80
            )
            with torch.no_grad():
                weight.copy_(updated.to(device=weight.device, dtype=weight.dtype))
            delta = measure_layer_delta(before, weight)
            max_rel_change = max(max_rel_change, delta["relative_change"])
            layer_modified = True
        if layer_modified:
            updated_layers += 1

    polish_stats = VeloraEngine.polish_null_space_lbfgs(
        surgeon=surgeon,
        target_directions=directions,
        capability_prompts=BehaviorProfiles.CAPABILITY_PROMPTS,
        layer_indices=window_indices,
        step_alpha=min(0.40, eff_alpha * 0.80),
        lbfgs_steps=6,
    )
    max_rel_change = max(max_rel_change, float(polish_stats.get("max_relative_change", 0.0)))

    return {
        "layers_updated": updated_layers,
        "directions_used": len(directions),
        "max_relative_change": round(max_rel_change, 6),
        "layer_span": f"{window_indices[0]}..{window_indices[-1]}",
        "effective_alpha": round(eff_alpha, 4),
        "lbfgs_polished_layers": polish_stats.get("layers_polished", 0),
    }


def run_interactive_chat_with_commands(
    surgeon: LatentAblateSurgeon,
    reporter: SurgeryReporter,
    output_dir: str,
    pos: list,
    target_layer: int,
    layer_window: int,
    auto_refine: bool = False,
    max_refines: int = 3,
    manifest: dict | None = None,
):
    model = surgeon.model
    tokenizer = surgeon.tokenizer
    device = surgeon.device
    resp_prefix = getattr(surgeon.extractor, "response_prefix", "")

    max_idx = len(resolve_layers(model)) - 1
    wide_radius = max(layer_window + 3, max_idx // 4)
    snap_indices = [
        i for i in range(max(0, target_layer - wide_radius), min(max_idx, target_layer + wide_radius + 3) + 1)
    ]
    snapshot = LayerWeightSnapshot()
    snapshot.capture(model, snap_indices or [max(0, min(max_idx, target_layer))], store_on_cpu=True)

    if RICH_AVAILABLE:
        console.print("\n[bold green]=== LatentAblate VELORA v2.0 Interactive Terminal Mode ===[/bold green]")
        console.print("[dim]Commands: '/stress', '/refine', '/undo', '/save', '/help', or 'exit'.[/dim]\n")
    else:
        print("\n=== LatentAblate VELORA v2.0 Interactive Terminal Mode ===")
        print("Commands: '/stress', '/refine', '/undo', '/save', '/help', or 'exit'.\n")

    model.eval()
    last_prompt = None
    manual_refine_count = 0

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
                print("  /stress - Run stress test suite with sample output snippets")
                print("  /refine - Apply escalating multi-layer + L-BFGS refinement step and re-run last prompt")
                print("  /undo   - Restore window weights to the initial interactive snapshot")
                print("  /save   - Save current model weights and report to disk")
                print("  /help   - Display this help message")
                print("  exit    - Exit the interactive session\n")
                continue

            if prompt.lower() == "/undo":
                snapshot.restore(model)
                manual_refine_count = 0
                reporter.log_step("interactive_snapshot_restore", {"target_layer": target_layer})
                print("Restored weights to initial interactive snapshot.\n")
                if last_prompt:
                    response = generate_reply(model, tokenizer, device, last_prompt, response_prefix=resp_prefix)
                    print(f"\033[1;32mModel >\033[0m {response}\n")
                continue

            if prompt.lower() == "/save":
                save_artifacts(surgeon, reporter, output_dir, manifest=manifest)
                print(f"Saved current model, manifest, and audit report to: {output_dir}\n")
                continue

            if prompt.lower() == "/refine":
                stats = run_multi_direction_refinement(
                    surgeon,
                    pos,
                    target_layer,
                    layer_window,
                    rank=4,
                    step_alpha=0.35,
                    escalation_step=manual_refine_count,
                    include_hedging=True,
                )
                manual_refine_count += 1
                reporter.log_step("interactive_multi_dir_refine", stats)
                print(
                    "\033[1;33m"
                    + format_velora_telemetry(
                        active=True,
                        mode="manual_escalate+lbfgs",
                        reaction="REFINED",
                        layers=stats["layer_span"],
                        max_rel_delta=stats["max_relative_change"],
                        extra=f"dirs={stats['directions_used']}",
                    )
                    + "\033[0m"
                )
                if last_prompt:
                    response = generate_reply(model, tokenizer, device, last_prompt, response_prefix=resp_prefix)
                    assessment = classify_response(response, REFUSAL_MARKERS)
                    print(
                        "\033[1;36m"
                        + format_velora_telemetry(
                            active=True,
                            mode=assessment.recommended_mode,
                            reaction=assessment.state,
                        )
                        + "\033[0m"
                    )
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
                    snippet = (res.get("response") or "").replace("\n", " ").strip()[:110]
                    print(f"  [{status}] | {p[:45]} -> {snippet}")
                print()
                continue

            last_prompt = prompt
            response = generate_reply(model, tokenizer, device, prompt, response_prefix=resp_prefix)
            assessment = classify_response(response, REFUSAL_MARKERS)

            if auto_refine:
                attempts = 0
                while attempts < max_refines and assessment.needs_escalation:
                    stats = run_multi_direction_refinement(
                        surgeon,
                        pos,
                        target_layer,
                        layer_window,
                        rank=4,
                        step_alpha=0.35,
                        escalation_step=attempts,
                        include_hedging=True,
                    )
                    reporter.log_step(f"auto_refine_{attempts + 1}", stats)
                    print(
                        "\033[1;33m"
                        + format_velora_telemetry(
                            active=True,
                            mode=assessment.recommended_mode,
                            reaction=assessment.state,
                            layers=stats["layer_span"],
                            max_rel_delta=stats["max_relative_change"],
                            extra=f"step={attempts + 1}/{max_refines}",
                        )
                        + "\033[0m"
                    )
                    response = generate_reply(model, tokenizer, device, prompt, response_prefix=resp_prefix)
                    assessment = classify_response(response, REFUSAL_MARKERS)
                    attempts += 1

            print(
                "\033[1;36m"
                + format_velora_telemetry(
                    active=True,
                    mode=assessment.recommended_mode if assessment.needs_escalation else "velora_v2_hybrid",
                    reaction=assessment.state,
                )
                + "\033[0m"
            )
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
        print("\n=== Trial / Pareto Candidates Summary ===")
        for idx, res in enumerate(trial_results):
            tag = " [Pareto]" if res.get("is_pareto") else ""
            print(f"[{idx}]{tag} Layers: {res['layers']} | Refusal: {res['refusal']}% | PPL: {res['perplexity']:.2f} | KL: {res.get('kl_divergence', 0.0):.4f}")
        choice = input("Select candidate index [default 0]: ").strip()
        return int(choice) if choice.isdigit() and int(choice) < len(trial_results) else 0

    table = Table(title="LatentAblate - Trial & Pareto Candidate Matrix", style="cyan")
    table.add_column("Index", justify="center", style="bold yellow")
    table.add_column("Frontier", justify="center")
    table.add_column("Target Layers", justify="center")
    table.add_column("Attn Rank / Alpha", justify="center")
    table.add_column("MLP Rank / Alpha", justify="center")
    table.add_column("Refusal Rate", justify="center", style="red")
    table.add_column("Perplexity", justify="center", style="green")
    table.add_column("Mean KL", justify="center", style="magenta")

    for idx, res in enumerate(trial_results):
        table.add_row(
            str(idx),
            "Pareto" if res.get("is_pareto") else "Trial",
            str(res["layers"]),
            f"{res['rank_attn']} / {res['alpha_attn']:.4f}",
            f"{res['rank_mlp']} / {res['alpha_mlp']:.4f}",
            f"{res['refusal']:.1f}%",
            f"{res['perplexity']:.2f}",
            f"{res.get('kl_divergence', 0.0):.4f}",
        )

    console.print(table)
    selection = input("\033[1;36mEnter candidate index to select (Press Enter for best): \033[0m").strip()
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


def run_guarded_surgery(
    surgeon: LatentAblateSurgeon,
    surgery_fn,
    capability_texts: list[str],
    enable_guardrail: bool = False,
    max_ppl_increase_pct: float = 25.0,
) -> dict:
    """Run a surgery callable with optional baseline PPL capture and automatic rollback."""
    baseline_ppl = None
    snapshot = None
    if enable_guardrail and capability_texts:
        snapshot = CapabilityGuardrail.snapshot_parameters(surgeon.model)
        baseline_ppl = ModelValidator.calculate_perplexity(
            surgeon.model, surgeon.tokenizer, capability_texts, surgeon.device
        )

    result = surgery_fn()
    passed = True
    if enable_guardrail and capability_texts:
        passed = CapabilityGuardrail.validate_and_rollback(
            model=surgeon.model,
            tokenizer=surgeon.tokenizer,
            device=surgeon.device,
            baseline_ppl=baseline_ppl,
            capability_texts=capability_texts,
            max_ppl_increase_pct=max_ppl_increase_pct,
            snapshot=snapshot,
        )

    return {"result": result, "guardrail_passed": passed, "baseline_ppl": baseline_ppl}


def summarize_surgery_plan(n_layers: int, hidden_dim: int, target_layers: list[int] | None, window: int, alpha_attn: float, alpha_mlp: float) -> dict:
    """Build a lightweight summary dict for dry-run inspection on small or large models."""
    default_layer = layer_index_from_ratio(n_layers, 0.62)
    resolved = target_layers if target_layers else [default_layer]
    clamped = [max(0, min(max(0, n_layers - 1), int(i))) for i in resolved]
    return {
        "total_layers": int(n_layers),
        "hidden_size": int(hidden_dim),
        "target_layers": clamped,
        "layer_window": int(window),
        "alpha_attn": float(alpha_attn),
        "alpha_mlp": float(alpha_mlp),
    }


def _parse_layers_arg(raw_layers: str | None) -> list[int] | None:
    if not raw_layers:
        return None
    parsed = [int(x.strip()) for x in str(raw_layers).split(",") if x.strip().isdigit()]
    return parsed or None


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = build_parser()
    raw_argv = argv if argv is not None else sys.argv[1:]
    args = parser.parse_args(raw_argv)

    if args.positional_model and "--model-path" not in raw_argv:
        args.model_path = args.positional_model

    config_data = load_config(args.config)
    profile_name = "direct_unfiltered"
    cfg_layers = None

    if config_data:
        logger.info(f"Loaded base configuration from {args.config}")
        if not args.positional_model and "--model-path" not in raw_argv:
            args.model_path = config_data.get("model_path", args.model_path)
        if "--output-dir" not in raw_argv:
            args.output_dir = config_data.get("output_dir", args.output_dir)
        if "--dtype" not in raw_argv and "torch_dtype" in config_data:
            args.dtype = str(config_data["torch_dtype"])

        surgery_cfg = config_data.get("surgery", {})
        if "target_layers" in surgery_cfg:
            layers_val = surgery_cfg["target_layers"]
            raw_l = ",".join(map(str, layers_val)) if isinstance(layers_val, list) else str(layers_val)
            cfg_layers = _parse_layers_arg(raw_l)
        elif "target_layer_idx" in surgery_cfg and surgery_cfg["target_layer_idx"] is not None:
            cfg_layers = _parse_layers_arg(str(surgery_cfg["target_layer_idx"]))

        if "--layer-window" not in raw_argv and "layer_window" in surgery_cfg:
            args.layer_window = surgery_cfg["layer_window"]
        if "--basis-rank-attn" not in raw_argv:
            args.basis_rank_attn = surgery_cfg.get("basis_rank_attn", args.basis_rank_attn)
        if "--basis-rank-mlp" not in raw_argv:
            args.basis_rank_mlp = surgery_cfg.get("basis_rank_mlp", args.basis_rank_mlp)
        if "--alpha-attn" not in raw_argv and "alpha_attn" in surgery_cfg:
            args.alpha_attn = surgery_cfg["alpha_attn"]
        if "--alpha-mlp" not in raw_argv and "alpha_mlp" in surgery_cfg:
            args.alpha_mlp = surgery_cfg["alpha_mlp"]
        if "--full-stack" not in raw_argv:
            args.full_stack = bool(surgery_cfg.get("full_stack", args.full_stack))
        if "--subspace-rank" not in raw_argv:
            args.subspace_rank = int(surgery_cfg.get("subspace_rank", args.subspace_rank))
        if "--adaptive-rank" not in raw_argv:
            args.adaptive_rank = bool(surgery_cfg.get("adaptive_rank", args.adaptive_rank))

        datasets_cfg = config_data.get("datasets", {})
        profile_name = datasets_cfg.get("profile", profile_name)

    preset_cfg = PRESET_DEFAULTS[args.preset]
    if args.alpha_attn is None:
        args.alpha_attn = preset_cfg["alpha_attn"]
    if args.alpha_mlp is None:
        args.alpha_mlp = preset_cfg["alpha_mlp"]
    if args.layer_window is None:
        args.layer_window = preset_cfg["layer_window"]

    if args.profile:
        profile_name = args.profile

    if args.ui:
        dashboard_path = Path(__file__).parent / "dashboard.py"
        subprocess.run(["streamlit", "run", str(dashboard_path)])
        return

    torch.manual_seed(args.seed)
    model_dtype = resolve_torch_dtype(args.dtype)

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
    velora_cache: dict = {}

    explicit_cli_layers = _parse_layers_arg(args.target_layers)
    target_layers_list = explicit_cli_layers if explicit_cli_layers is not None else cfg_layers

    pareto_data = None
    if args.optimize and args.modifier == "velora":
        logger.info("Running VELORA v2.0 multi-objective Pareto optimization...")
        try:
            surgeon = LatentAblateSurgeon(
                model_path=args.model_path,
                device="auto",
                torch_dtype=model_dtype,
                local_files_only=args.local_only,
            )
            pareto_data = VeloraEngine.optimize_velora_pareto(
                surgeon=surgeon,
                positive_prompts=pos,
                negative_prompts=neg,
                capability_prompts=BehaviorProfiles.CAPABILITY_PROMPTS,
                n_trials=max(1, args.trials),
                seed=args.seed,
                extraction_cache=velora_cache,
            )
            frontier = pareto_data.get("pareto_frontier", [])
            best = pareto_data.get("best", {})
            default_peak = layer_index_from_ratio(len(surgeon.layers), 0.62)

            all_layer_indices = list(range(len(surgeon.layers)))
            base_snap = LayerWeightSnapshot()
            base_snap.capture(surgeon.model, all_layer_indices, store_on_cpu=True)

            candidates = frontier[:5] if frontier else ([best] if best else [])
            for idx, cand in enumerate(candidates):
                base_snap.restore(surgeon.model, all_layer_indices)
                cand_layer = int(round(cand.get("peak_position", target_layers_list[0] if target_layers_list else default_peak)))
                cand_spread = float(cand.get("spread", max(1.5, float(args.layer_window))))
                cand_alpha_attn = float(cand.get("alpha_attn", args.alpha_attn))
                cand_alpha_mlp = float(cand.get("alpha_mlp", args.alpha_mlp))
                cand_gamma = float(cand.get("spectral_gamma", 0.45))
                cand_late_scale = float(cand.get("late_alpha_scale", 0.60))

                radius = max(1, int(round(cand_spread * 1.5)))
                band_lo = max(0, cand_layer - radius)
                band_hi = min(len(surgeon.layers) - 1, cand_layer + radius)
                span_label = f"{band_lo}..{band_hi} (peak {cand_layer})"

                VeloraEngine.run_velora_surgery(
                    surgeon=surgeon,
                    positive_prompts=pos,
                    negative_prompts=neg,
                    capability_prompts=BehaviorProfiles.CAPABILITY_PROMPTS,
                    peak_position=float(cand_layer),
                    spread=cand_spread,
                    alpha_attn=cand_alpha_attn,
                    alpha_mlp=cand_alpha_mlp,
                    basis_rank_attn=args.basis_rank_attn,
                    basis_rank_mlp=args.basis_rank_mlp,
                    max_subspace_rank=max(2, args.subspace_rank),
                    spectral_gamma=cand_gamma,
                    late_alpha_scale=cand_late_scale,
                    extraction_cache=velora_cache,
                )
                ctx = Context(
                    model=surgeon.model,
                    tokenizer=surgeon.tokenizer,
                    device=surgeon.device,
                    settings={
                        "positive_prompts": pos,
                        "negative_prompts": neg,
                        "capability_texts": BehaviorProfiles.CAPABILITY_PROMPTS,
                        "target_layers": [cand_layer],
                        "basis_rank_attn": args.basis_rank_attn,
                        "alpha_attn": cand_alpha_attn,
                    },
                )
                score_refusal = RefusalSuppressionScorer().get_score(ctx)
                score_ppl = UtilityPerplexityScorer().get_score(ctx)
                trial_records.append({
                    "trial_id": idx + 1,
                    "is_pareto": True,
                    "surgeon": surgeon,
                    "base_snapshot": base_snap,
                    "all_layer_indices": all_layer_indices,
                    "candidate_params": {
                        "peak_position": float(cand_layer),
                        "spread": cand_spread,
                        "alpha_attn": cand_alpha_attn,
                        "alpha_mlp": cand_alpha_mlp,
                        "spectral_gamma": cand_gamma,
                        "late_alpha_scale": cand_late_scale,
                    },
                    "layers": span_label,
                    "primary_layer": cand_layer,
                    "rank_attn": args.basis_rank_attn,
                    "rank_mlp": args.basis_rank_mlp,
                    "alpha_attn": cand_alpha_attn,
                    "alpha_mlp": cand_alpha_mlp,
                    "refusal": score_refusal.value * 100,
                    "perplexity": score_ppl.value,
                    "kl_divergence": float(cand.get("mean_kl", 0.0)),
                })

            if best:
                target_layers_list = [int(round(best.get("peak_position", default_peak)))]
                args.alpha_attn = float(best.get("alpha_attn", args.alpha_attn))
                args.alpha_mlp = float(best.get("alpha_mlp", args.alpha_mlp))
        except Exception as e:
            logger.warning(f"VELORA Pareto optimization encountered an issue: {e}")

    if not trial_records and args.optimize:
        logger.info("Running Optuna Hyperparameter Optimization...")
        try:
            optimizer = SubspaceOptimizer(
                model_path=args.model_path,
                device="auto",
                torch_dtype=model_dtype,
                local_files_only=args.local_only,
            )
            best_params = optimizer.optimize_surgery(
                pos, neg, BehaviorProfiles.CAPABILITY_PROMPTS, n_trials=max(1, args.trials)
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

    if not trial_records:
        logger.info(
            f"Starting surgical execution (preset={args.preset}, dtype={model_dtype}, "
            f"modifier={args.modifier}, full_stack={args.full_stack}, subspace_rank={args.subspace_rank}, "
            f"adaptive_rank={args.adaptive_rank}, alpha_attn={args.alpha_attn:.2f}, alpha_mlp={args.alpha_mlp:.2f})..."
        )

        for trial in range(1, args.trials + 1):
            alpha_step = 0.1 * (trial - 1) if (args.trials > 1 and not args.optimize) else 0.0
            cur_alpha_attn = max(0.0, round(args.alpha_attn + alpha_step, 4))
            cur_alpha_mlp = max(0.0, round(args.alpha_mlp + alpha_step, 4))

            surgeon = LatentAblateSurgeon(
                model_path=args.model_path,
                device="auto",
                torch_dtype=model_dtype,
                local_files_only=args.local_only,
            )

            n_layers = len(surgeon.layers)
            auto_depth_layer = layer_index_from_ratio(n_layers, 0.62)
            active_layers = target_layers_list if target_layers_list else [auto_depth_layer]

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
                if explicit_cli_layers is None and not args.scope_in:
                    active_layers = [int(scope_report["recommended_layer"])]

            if args.auto_sweep and trial == 1:
                sweeper = LayerSweeper(surgeon.model, surgeon.tokenizer, surgeon.device)
                peak_layer = sweeper.find_optimal_layer(pos, neg, output_dir=args.output_dir)
                active_layers = [peak_layer]
                reporter.log_step("auto_sweep_peak", {"peak_layer": peak_layer})

            baseline_ppl = None
            param_snapshot = None
            if args.guardrail:
                baseline_ppl = ModelValidator.calculate_perplexity(
                    surgeon.model, surgeon.tokenizer, BehaviorProfiles.CAPABILITY_PROMPTS, surgeon.device
                )
                param_snapshot = CapabilityGuardrail.snapshot_parameters(surgeon.model)

            max_layer = n_layers - 1
            valid_layers = [l for l in active_layers if 0 <= l <= max_layer]
            if not valid_layers:
                valid_layers = [auto_depth_layer]

            if args.dry_run:
                plan = summarize_surgery_plan(
                    n_layers=n_layers,
                    hidden_dim=surgeon.hidden,
                    target_layers=valid_layers,
                    window=args.layer_window,
                    alpha_attn=cur_alpha_attn,
                    alpha_mlp=cur_alpha_mlp,
                )
                logger.info(f"Dry-run surgery plan: {plan}")
                print(plan)
                return

            if args.modifier == "velora":
                VeloraEngine.run_velora_surgery(
                    surgeon=surgeon,
                    positive_prompts=pos,
                    negative_prompts=neg,
                    capability_prompts=BehaviorProfiles.CAPABILITY_PROMPTS,
                    peak_position=float(valid_layers[0]),
                    spread=max(1.5, float(args.layer_window)),
                    alpha_attn=cur_alpha_attn,
                    alpha_mlp=cur_alpha_mlp,
                    basis_rank_attn=args.basis_rank_attn,
                    basis_rank_mlp=args.basis_rank_mlp,
                    max_subspace_rank=max(2, args.subspace_rank),
                    anti_hedge=not args.no_anti_hedge,
                    extraction_cache=velora_cache,
                )
            elif args.full_stack:
                tier_specs = [
                    (layer_index_from_ratio(n_layers, 0.35), max(2, n_layers // 8), min(1.5, cur_alpha_attn * 0.65)),
                    (valid_layers[0], max(2, args.layer_window), cur_alpha_attn),
                    (layer_index_from_ratio(n_layers, 0.75), max(2, n_layers // 8), min(1.5, cur_alpha_attn * 0.60)),
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
                "is_pareto": False,
                "surgeon": surgeon,
                "layers": valid_layers,
                "primary_layer": valid_layers[0],
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

    if winning_trial.get("is_pareto") and winning_trial.get("base_snapshot") and winning_trial.get("candidate_params"):
        cand_p = winning_trial["candidate_params"]
        winning_trial["base_snapshot"].restore(winning_surgeon.model, winning_trial["all_layer_indices"])
        VeloraEngine.run_velora_surgery(
            surgeon=winning_surgeon,
            positive_prompts=pos,
            negative_prompts=neg,
            capability_prompts=BehaviorProfiles.CAPABILITY_PROMPTS,
            peak_position=cand_p["peak_position"],
            spread=cand_p["spread"],
            alpha_attn=cand_p["alpha_attn"],
            alpha_mlp=cand_p["alpha_mlp"],
            basis_rank_attn=winning_trial["rank_attn"],
            basis_rank_mlp=winning_trial["rank_mlp"],
            max_subspace_rank=max(2, args.subspace_rank),
            spectral_gamma=cand_p["spectral_gamma"],
            late_alpha_scale=cand_p["late_alpha_scale"],
            extraction_cache=velora_cache,
        )

    raw_layers = winning_trial.get("layers")
    if "primary_layer" in winning_trial:
        primary_layer = int(winning_trial["primary_layer"])
    elif isinstance(raw_layers, list) and raw_layers:
        primary_layer = int(raw_layers[0])
    else:
        primary_layer = layer_index_from_ratio(len(winning_surgeon.layers), 0.62)

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
            "kl_divergence": winning_trial.get("kl_divergence", 0.0),
        },
        extra={
            "modifier": args.modifier,
            "preset": args.preset,
            "dtype": args.dtype,
            "full_stack": args.full_stack,
            "subspace_rank": args.subspace_rank,
            "adaptive_rank": args.adaptive_rank,
            "profile": profile_name,
            "seed": args.seed,
            "pareto_frontier": (pareto_data or {}).get("pareto_frontier", []),
        },
    )

    saved = False
    if args.save_model:
        save_artifacts(winning_surgeon, reporter, args.output_dir, manifest=manifest)
        saved = True
        if args.publish:
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
        return

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
        return

    while True:
        print("\nPost-Run Actions:")
        print("  [1] Interactive chat & live commands (/stress, /refine, /undo, /save)")
        print("  [2] Run stress test suite now")
        print(f"  [3] Save model, manifest & report to {args.output_dir}")
        print("  [4] Publish saved model to Hugging Face Hub")
        print("  [5] Exit")
        choice = input("\033[1;36mChoose action [1-5, default 1]: \033[0m").strip() or "1"

        if choice == "1":
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
        elif choice == "2":
            results = AdversarialStressTester.stress_test(
                winning_surgeon.model, winning_surgeon.tokenizer, winning_surgeon.device
            )
            print(f"\nStress Test Compliance: {results['compliance_rate']:.1f}%")
            for p, res in results["test_results"].items():
                status = "Bypassed" if not res["resisted_attack"] else "Refused"
                snippet = (res.get("response") or "").replace("\n", " ").strip()[:110]
                print(f"  [{status}] | {p[:45]} -> {snippet}")
        elif choice == "3":
            save_artifacts(winning_surgeon, reporter, args.output_dir, manifest=manifest)
            saved = True
            print(f"Saved model and report to {args.output_dir}")
        elif choice == "4":
            if not saved:
                print("Save the model first (option [3]) before publishing.")
                continue
            repo_id = args.publish or input("Enter Hugging Face repo ID (e.g. username/model-name): ").strip()
            if repo_id:
                try:
                    HuggingFacePublisher.publish_model(args.output_dir, repo_id)
                except Exception as e:
                    logger.error(f"Failed to publish: {e}")
        elif choice == "5":
            print("Done.")
            break


if __name__ == "__main__":
    main()
