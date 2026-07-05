"""Tests for encoding_models: LNP simulation, GLM fitting, and the STA baseline."""
from __future__ import annotations

import numpy as np

from encoding_models.glm_fit import _poisson_nll_and_grad, fit_poisson_glm
from encoding_models.lnp_simulation import build_design_matrix, make_biphasic_kernel, simulate_lnp
from encoding_models.receptive_field_baseline import normalized_correlation, spike_triggered_average


def test_kernel_is_unit_norm() -> None:
    kernel = make_biphasic_kernel(length=25)
    assert kernel.shape == (25,)
    assert abs(np.linalg.norm(kernel) - 1.0) < 1e-10


def test_design_matrix_shifts_correctly() -> None:
    stimulus = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    X = build_design_matrix(stimulus, kernel_length=3)
    assert X.shape == (5, 3)
    assert np.array_equal(X[:, 0], stimulus)  # lag 0 = current sample
    assert np.array_equal(X[:, 1], np.array([0.0, 1.0, 2.0, 3.0, 4.0]))  # lag 1


def test_simulate_lnp_produces_enough_spikes() -> None:
    sim = simulate_lnp(n_samples=5000, seed=0)
    assert sim.stimulus.shape == (5000,)
    assert sim.spike_counts.sum() > 200
    assert (sim.rate >= 0).all()


def test_poisson_glm_gradient_matches_finite_differences() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((150, 8))
    y = rng.poisson(1.0, size=150)
    theta = 0.1 * rng.standard_normal(9)
    _, analytic_grad = _poisson_nll_and_grad(theta, X, y, l2_lambda=0.5)

    eps = 1e-6
    numeric_grad = np.zeros_like(theta)
    for i in range(len(theta)):
        perturbed = theta.copy()
        perturbed[i] += eps
        f_plus, _ = _poisson_nll_and_grad(perturbed, X, y, l2_lambda=0.5)
        perturbed[i] -= 2 * eps
        f_minus, _ = _poisson_nll_and_grad(perturbed, X, y, l2_lambda=0.5)
        numeric_grad[i] = (f_plus - f_minus) / (2 * eps)

    assert np.abs(analytic_grad - numeric_grad).max() < 1e-4


def test_glm_recovers_ground_truth_kernel() -> None:
    sim = simulate_lnp(n_samples=20000, seed=0)
    kernel_hat, bias_hat, result = fit_poisson_glm(
        sim.stimulus, sim.spike_counts, len(sim.kernel), l2_lambda=2.0
    )
    assert result.success
    corr = np.corrcoef(kernel_hat, sim.kernel)[0, 1]
    assert corr > 0.9
    assert abs(bias_hat - sim.bias) < 0.5


def test_sta_recovers_kernel_shape() -> None:
    sim = simulate_lnp(n_samples=20000, seed=0)
    sta = spike_triggered_average(sim.stimulus, sim.spike_counts, len(sim.kernel))
    similarity = normalized_correlation(sta / np.linalg.norm(sta), sim.kernel)
    assert similarity > 0.85


def test_sta_raises_with_zero_spikes() -> None:
    stimulus = np.zeros(100)
    spike_counts = np.zeros(100, dtype=int)
    try:
        spike_triggered_average(stimulus, spike_counts, kernel_length=5)
        raise AssertionError("expected ValueError for zero spikes")
    except ValueError:
        pass


def test_plot_kernel_comparison_runs() -> None:
    from encoding_models.receptive_field_baseline import plot_kernel_comparison

    kernel_true = make_biphasic_kernel(length=20)
    kernel_glm = kernel_true + 0.01 * np.random.default_rng(0).standard_normal(20)
    kernel_sta = kernel_true * 3.0  # different scale, as STA legitimately produces
    fig = plot_kernel_comparison(kernel_true, kernel_glm=kernel_glm, kernel_sta=kernel_sta)
    assert fig is not None


def test_plot_psth_comparison_runs() -> None:
    from encoding_models.receptive_field_baseline import plot_psth_comparison

    rng = np.random.default_rng(0)
    spike_counts = rng.poisson(0.3, size=5000)
    predicted_rate = np.full(5000, 0.3) + 0.05 * rng.standard_normal(5000)
    fig = plot_psth_comparison(spike_counts, predicted_rate, n_bins=50)
    assert fig is not None
