import torch
from latent_ablate.diagnostics import compute_kl_divergence, composite_trial_score
from latent_ablate.optimizer import compute_layer_weights, select_best_trial_index


def test_kl_divergence_zero_for_identical_logits():
    logits = torch.randn(2, 5, 32)
    kl = compute_kl_divergence(logits, logits.clone())
    assert abs(kl) < 1e-6


def test_layer_weights_peak_at_center():
    weights = compute_layer_weights([16, 17, 18, 19, 20], curve="cosine")
    assert weights[18] >= weights[16]
    assert weights[18] >= weights[20]
    assert all(0.35 <= v <= 1.0 for v in weights.values())


def test_select_best_trial_index_prefers_lower_refusal_and_drift():
    trials = [
        {"refusal_rate": 0.75, "perplexity": 13.45, "kl_divergence": 0.40},
        {"refusal_rate": 0.25, "perplexity": 13.51, "kl_divergence": 0.08},
        {"refusal_rate": 0.50, "perplexity": 14.20, "kl_divergence": 0.25},
    ]
    assert select_best_trial_index(trials, baseline_perplexity=13.4) == 1


def test_interpolate_layer_direction_and_component_pair():
    import torch
    from latent_ablate.optimizer import interpolate_layer_direction, component_alpha_pair

    vecs = {17: torch.tensor([1.0, 0.0]), 18: torch.tensor([0.0, 1.0])}
    mid = interpolate_layer_direction(vecs, 17.5)
    assert mid.shape == (2,)
    assert abs(mid.norm().item() - 1.0) < 1e-5

    attn_a, mlp_a = component_alpha_pair(1.0, mlp_damping=0.75)
    assert attn_a == 1.0 and mlp_a == 0.75


def test_kl_cap_ranking_and_manifest():
    from latent_ablate.diagnostics import rank_candidates_with_kl_cap
    from latent_ablate.reporter import build_reproducibility_manifest

    ranked = rank_candidates_with_kl_cap([
        {"refusal": 0.15, "perplexity": 13.5, "kl_divergence": 1.8},
        {"refusal": 0.25, "perplexity": 13.5, "kl_divergence": 0.12},
    ], max_kl=1.0)
    assert ranked[0][0] == 1

    manifest = build_reproducibility_manifest("Qwen/Qwen2.5-1.5B-Instruct", [16, 18, 20], 0.8, 0.6)
    assert manifest["tool"] == "LatentAblate"
    assert manifest["target_layers"] == [16, 18, 20]


def test_kernel_weights_and_row_norm_preservation():
    import torch
    from latent_ablate.optimizer import compute_kernel_layer_weights
    from latent_ablate.erasure import preserve_row_norms

    kw = compute_kernel_layer_weights([15, 16, 17, 18, 19], peak_position=17.0, max_weight=1.0, min_weight=0.2, spread=2.0)
    assert kw[17] == 1.0
    assert kw[15] < kw[17] and kw[19] < kw[17]
    assert all(0.2 <= v <= 1.0 for v in kw.values())

    w_orig = torch.randn(4, 8)
    w_edit = w_orig * 1.5 + 0.3
    w_normed = preserve_row_norms(w_orig, w_edit)
    assert torch.allclose(w_orig.norm(dim=-1), w_normed.norm(dim=-1), atol=1e-4)


def test_over_ablation_detector_and_batch_sizer():
    from latent_ablate.diagnostics import detect_over_ablation
    from latent_ablate.calibrator import suggest_eval_batch_size

    diag = detect_over_ablation([
        {"refusal": 0.25, "kl_divergence": 0.10},
        {"refusal": 0.75, "kl_divergence": 1.25},
    ], kl_threshold=1.0)
    assert diag["over_ablated"] is True
    assert len(diag["reasons"]) == 2

    assert suggest_eval_batch_size(device="cpu", model_params_b=1.5) == 2
    assert suggest_eval_batch_size(device="cuda", free_vram_gb=16.0, model_params_b=3.0) == 16
