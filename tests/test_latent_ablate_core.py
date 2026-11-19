import torch

from latent_ablate.cli import build_parser
from latent_ablate.erasure import (
    apply_erasure,
    build_capability_basis,
    explained_variance_ratio,
    oblique_weights,
    preserve_row_norms,
)
from latent_ablate.optimizer import compute_kernel_layer_weights, compute_layer_weights
from latent_ablate.riemannian_steering import RiemannianGeodesicEngine
from latent_ablate.scope import AblateScope


def test_preserve_row_norms_keeps_row_magnitude():
    orig = torch.randn(16, 32)
    modified = orig + 0.5 * torch.randn(16, 32)
    calibrated = preserve_row_norms(orig, modified)

    orig_norms = orig.norm(dim=-1)
    cal_norms = calibrated.norm(dim=-1)
    assert torch.allclose(orig_norms, cal_norms, atol=1e-5)


def test_capability_basis_is_orthonormal_and_tracks_variance():
    torch.manual_seed(0)
    cap_acts = torch.randn(16, 64)
    basis = build_capability_basis(cap_acts, rank=6)
    gram = basis @ basis.T
    assert torch.allclose(gram, torch.eye(6), atol=1e-5)

    ev_small = explained_variance_ratio(cap_acts, rank=2)
    ev_large = explained_variance_ratio(cap_acts, rank=10)
    assert 0.0 < ev_small <= ev_large <= 1.0


def test_oblique_weights_reduces_capability_overlap():
    torch.manual_seed(0)
    cap_acts = torch.randn(12, 64)
    basis = build_capability_basis(cap_acts, rank=4)
    r = torch.randn(64)
    _, w_ortho = oblique_weights(r, basis)

    proj_before = (basis.T @ (basis @ (r / r.norm()))).norm().item()
    proj_after = (basis.T @ (basis @ w_ortho)).norm().item()
    assert proj_after <= proj_before + 1e-5


def test_spherical_geodesic_transport_preserves_row_norms():
    h = torch.randn(8, 32)
    v = torch.randn(32)
    transported = RiemannianGeodesicEngine.spherical_geodesic_transport(h, v, step_size=0.35)
    assert torch.allclose(h.norm(dim=-1), transported.norm(dim=-1), atol=1e-5)

    basis = torch.randn(4, 32)
    inv_gram = RiemannianGeodesicEngine.regularized_gram_inverse(basis)
    assert inv_gram.shape == (4, 4)


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


def test_cli_flags_and_scope_roundtrip(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "--modifier", "mpoa",
        "--full-stack",
        "--subspace-rank", "3",
        "--auto-sweep",
        "--guardrail",
        "--scope-out", str(tmp_path / "spec.json"),
    ])
    assert args.modifier == "mpoa"
    assert args.full_stack is True
    assert args.subspace_rank == 3
    assert args.auto_sweep is True
    assert args.guardrail is True

    spec = {
        "schema": "LatentScopeSpec/1.1",
        "total_layers": 24,
        "recommended_layer": 14,
        "recommended_alpha": 0.95,
        "layer_map": [],
    }
    out_file = tmp_path / "test.latentspec.json"
    AblateScope.save_spec(spec, str(out_file))
    loaded = AblateScope.load_spec(str(out_file))
    assert loaded == spec
