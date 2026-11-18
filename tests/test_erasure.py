import torch
import pytest
from latent_ablate.erasure import conflict_score, build_capability_basis, oblique_weights, apply_erasure

def test_conflict_score_orthogonal():
    r = torch.tensor([1.0, 0.0, 0.0])
    basis = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    score = conflict_score(r, basis)
    assert score == pytest.approx(0.0, abs=1e-5)

def test_conflict_score_parallel():
    r = torch.tensor([0.0, 1.0, 0.0])
    basis = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    score = conflict_score(r, basis)
    assert score == pytest.approx(1.0, abs=1e-5)

def test_build_capability_basis():
    acts = torch.randn(10, 32)
    basis = build_capability_basis(acts, rank=4)
    assert basis.shape == (4, 32)
    # Check orthonormality: basis @ basis.T should be close to Identity
    identity_approx = basis @ basis.T
    assert torch.allclose(identity_approx, torch.eye(4), atol=1e-4)

def test_apply_erasure():
    W = torch.eye(4)
    r = torch.tensor([1.0, 0.0, 0.0, 0.0])
    w = torch.tensor([1.0, 0.0, 0.0, 0.0])
    W_prime = apply_erasure(W, r, w, alpha=1.0)
    # The first row/column should be cleared out
    assert W_prime[0, 0].item() == pytest.approx(0.0, abs=1e-5)
