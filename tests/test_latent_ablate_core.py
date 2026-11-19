import torch
import torch.nn as nn

from latent_ablate.cli import build_parser
from latent_ablate.erasure import apply_erasure, build_capability_basis, oblique_weights, preserve_row_norms
from latent_ablate.optimizer import compute_layer_weights, compute_kernel_layer_weights
from latent_ablate.scope import AblateScope


def test_preserve_row_norms_keeps_row_magnitude():
    orig = torch.randn(16, 32)
    modified = orig + 0.5 * torch.randn(16, 32)
    calibrated = preserve_row_norms(orig, modified)

    orig_norms = orig.norm(dim=-1)
    cal_norms = calibrated.norm(dim=-1)
    assert torch.allclose(orig_norms, cal_norms, atol=1e-5)


def test_oblique_weights_reduces_capability_overlap():
    torch.manual_seed(0)
    cap_acts = torch.randn(12, 64)
    basis = build_capability_basis(cap_acts, rank=4)
    r = torch.randn(64)
    _, w_ortho = oblique_weights(r, basis)

    proj_before = (basis.T @ (basis @ (r / r.norm()))).norm().item()
    proj_after = (basis.T @ (basis @ w_ortho)).norm().item()
    assert proj_after <= proj_before + 1e-5


def test_apply_erasure_modifies_matching_linear():
    W = torch.randn(32, 32)
    r = torch.randn(32)
    out = apply_erasure(W, r, alpha=1.0)
    assert out.shape == W.shape
    assert not torch.allclose(out, W)


def test_layer_weight_curves():
    weights = compute_layer_weights([10, 11, 12, 13, 14], curve="cosine")
    assert set(weights.keys()) == {10, 11, 12, 13, 14}
    assert weights[12] >= weights[10]

    kernel_w = compute_kernel_layer_weights([10, 12, 14], peak_position=12.0)
    assert kernel_w[12] >= kernel_w[10]


def test_cli_full_stack_and_mpoa_flags():
    parser = build_parser()
    args = parser.parse_args(["--modifier", "mpoa", "--full-stack"])
    assert args.modifier == "mpoa"
    assert args.full_stack is True


def test_ablate_scope_spec_roundtrip(tmp_path):
    spec = {
        "schema": "LatentScopeSpec/1.0",
        "total_layers": 24,
        "recommended_layer": 14,
        "recommended_alpha": 0.95,
        "layer_map": [],
    }
    out_file = tmp_path / "test.latentspec.json"
    AblateScope.save_spec(spec, str(out_file))
    loaded = AblateScope.load_spec(str(out_file))
    assert loaded == spec
