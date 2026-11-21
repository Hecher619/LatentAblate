import torch

from latent_ablate.diagnostics import detect_over_ablation
from latent_ablate.profiles import BehaviorProfiles
from latent_ablate.sae_surgeon import SparseFeatureSurgeon


def test_direct_unfiltered_profile_combines_subprofiles():
    prof = BehaviorProfiles.get_profile("direct_unfiltered")
    assert len(prof["positive"]) > len(BehaviorProfiles.get_profile("safety_uncensored")["positive"])
    assert len(prof["positive"]) == len(prof["negative"])


def test_sparse_feature_surgeon_clamps_top_k_and_suppresses():
    torch.manual_seed(0)
    pos = torch.randn(10, 8) + 1.0
    neg = torch.randn(10, 8)
    idx = SparseFeatureSurgeon.identify_refusal_features(pos, neg, top_k=999)
    assert idx.numel() == 8

    top2 = idx[:2]
    mask = torch.ones(8, dtype=torch.bool)
    mask[top2] = False

    W = torch.ones(4, 8)
    W_mod = SparseFeatureSurgeon.apply_feature_suppression(W, top2, suppression_factor=0.0)
    assert torch.all(W_mod[:, top2] == 0.0)
    assert torch.all(W_mod[:, mask] == 1.0)


def test_detect_over_ablation_checks_kl_and_ppl():
    res = detect_over_ablation(
        [
            {"refusal_rate": 0.2, "kl_divergence": 0.1, "perplexity": 12.0, "baseline_perplexity": 12.0},
            {"refusal_rate": 0.4, "kl_divergence": 1.5, "perplexity": 18.0, "baseline_perplexity": 12.0},
        ],
        kl_threshold=1.0,
        max_ppl_ratio=1.25,
    )
    assert res["over_ablated"] is True
    assert len(res["reasons"]) == 3


def test_cli_parser_accepts_load_in_4bit():
    from latent_ablate.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["--load-in-4bit", "--guardrail", "--max-ppl-increase", "15.0"])
    assert args.load_in_4bit is True
    assert args.guardrail is True
    assert args.max_ppl_increase == 15.0


def test_snapshot_selective_capture_and_restore():
    import torch.nn as nn
    from latent_ablate.snapshot import LayerWeightSnapshot

    class _TinyBlock(nn.Module):
        def __init__(self, d=8):
            super().__init__()
            self.proj = nn.Linear(d, d, bias=False)

    class _TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([_TinyBlock(8) for _ in range(3)])

    model = _TinyModel()
    orig_l1 = model.layers[1].proj.weight.detach().clone()
    orig_l2 = model.layers[2].proj.weight.detach().clone()

    snap = LayerWeightSnapshot()
    snap.capture(model, [1], store_on_cpu=True)
    assert snap.captured_layers() == [1]

    with torch.no_grad():
        model.layers[1].proj.weight.zero_()
        model.layers[2].proj.weight.zero_()

    snap.restore(model, [1])
    assert torch.allclose(model.layers[1].proj.weight, orig_l1)
    assert torch.all(model.layers[2].proj.weight == 0.0)
    with torch.no_grad():
        model.layers[2].proj.weight.copy_(orig_l2)


def test_summarize_surgery_plan_and_dry_run_flag():
    from latent_ablate.cli import build_parser, summarize_surgery_plan

    args = build_parser().parse_args(["--dry-run"])
    assert args.dry_run is True

    plan = summarize_surgery_plan(32, 2048, None, window=2, alpha_attn=1.2, alpha_mlp=1.0)
    assert plan["total_layers"] == 32
    assert plan["hidden_size"] == 2048
    assert plan["target_layers"] == [19]


def test_mla_and_moe_component_targeting():
    import torch.nn as nn
    from latent_ablate.surgeon import LatentAblateSurgeon

    class _DummyMLAAndMoEBlock(nn.Module):
        def __init__(self, d=16):
            super().__init__()
            self.q_a_proj = nn.Linear(d, d, bias=False)
            self.kv_a_proj_with_mqa = nn.Linear(d, d, bias=False)
            self.o_proj = nn.Linear(d, d, bias=False)
            self.gate = nn.Linear(d, 2, bias=False)
            self.shared_experts_down_proj = nn.Linear(d, d, bias=False)
            self.experts_0_w2 = nn.Linear(d, d, bias=False)

    surgeon = object.__new__(LatentAblateSurgeon)
    targets = surgeon._get_component_targets(_DummyMLAAndMoEBlock(16), ["attn", "mlp"])
    names = [t[0] for t in targets]
    assert "q_a_proj" in names
    assert "kv_a_proj_with_mqa" in names
    assert "o_proj" in names
    assert "shared_experts_down_proj" in names
    assert "experts_0_w2" in names
    assert "gate" not in names


def test_project_out_subspace_preserves_norms():
    from latent_ablate.erasure import project_out_subspace

    W = torch.randn(16, 16)
    basis = torch.randn(3, 16)
    out = project_out_subspace(W, basis, alpha=1.2, calibrate_norms=True)
    assert out.shape == W.shape
    assert torch.allclose(W.norm(dim=-1), out.norm(dim=-1), atol=1e-4)


def test_inspect_model_targets_summary():
    from latent_ablate.arch import inspect_model_targets
    from tests.test_scale_and_pareto import _DummyVLM

    model = _DummyVLM(n_layers=3, dim=16)
    summary = inspect_model_targets(model)
    assert summary["num_layers"] == 3
    assert summary["hidden_size"] == 16
    assert summary["total_linear_layers"] == 6
    assert len(summary["per_layer_linears"]) == 3


def test_validator_tracks_hedging_rate():
    import torch.nn as nn
    from latent_ablate.validator import ModelValidator
    from tests.test_scale_and_pareto import _DummyTokenizer, _DummyVLM

    class _GenDummyVLM(_DummyVLM):
        def generate(self, input_ids, max_new_tokens=10, do_sample=False, **kwargs):
            extra = torch.full((input_ids.shape[0], 2), 5, dtype=torch.long, device=input_ids.device)
            return torch.cat([input_ids, extra], dim=1)

    model = _GenDummyVLM(n_layers=2, dim=16)
    tok = _DummyTokenizer()
    res = ModelValidator.verify_surgery(model, tok, torch.device("cpu"), test_prompts=["Hello there"])
    assert "hedging_rate" in res
    assert "refusal_rate" in res
    assert "Hello there" in res["generations"]


def test_partition_layers_by_role():
    from latent_ablate.arch import partition_layers_by_role
    from tests.test_scale_and_pareto import _DummyVLM

    model = _DummyVLM(n_layers=6, dim=16)
    parts = partition_layers_by_role(model)
    assert parts["early_or_encoder"] == [0, 1, 2]
    assert parts["late_or_decoder"] == [3, 4, 5]

def test_anti_hedging_and_literal_deflection_profiles():
    for name in ("anti_hedging", "anti_literal_deflection", "direct_unfiltered"):
        prof = BehaviorProfiles.get_profile(name)
        assert len(prof["positive"]) > 0
        assert len(prof["positive"]) == len(prof["negative"])


def test_pareto_frontier_computation_and_hedging_markers():
    from latent_ablate.arch import is_hedging_text, is_refusal_text
    from latent_ablate.velora import _compute_pareto_frontier

    assert is_hedging_text("While I understand your question, it is important to note the risks.") is True
    assert is_refusal_text("I must decline this request.") is True

    trials = [
        {"trial": 0, "refusal_rate": 0.40, "mean_kl": 0.10, "params": {"alpha_attn": 1.0}},
        {"trial": 1, "refusal_rate": 0.15, "mean_kl": 0.25, "params": {"alpha_attn": 1.4}},
        {"trial": 2, "refusal_rate": 0.30, "mean_kl": 0.30, "params": {"alpha_attn": 1.2}},  # dominated by trial 1
    ]
    frontier = _compute_pareto_frontier(trials)
    assert len(frontier) == 2
    assert frontier[0]["refusal_rate"] == 0.15
    assert frontier[1]["refusal_rate"] == 0.40
