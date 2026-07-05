"""Poisson GLM fitting via maximum likelihood, no sklearn/statsmodels.

The exponential-nonlinearity Poisson GLM is a canonical generalized
linear model: its negative log-likelihood is convex in the parameters,
so gradient-based optimization is guaranteed to find the global
optimum -- unlike many neural encoding model fits. The gradient is
derived and implemented directly rather than left to numerical
differentiation.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from encoding_models.lnp_simulation import build_design_matrix


def _poisson_nll_and_grad(
    theta: np.ndarray, X: np.ndarray, y: np.ndarray, l2_lambda: float
) -> tuple[float, np.ndarray]:
    """Negative log-likelihood (dropping the y!-dependent constant) and its exact gradient."""
    kernel, bias = theta[:-1], theta[-1]
    linear = np.clip(X @ kernel + bias, -20, 20)
    rate_hat = np.exp(linear)
    nll = float(np.sum(rate_hat - y * linear) + 0.5 * l2_lambda * np.sum(kernel**2))
    resid = rate_hat - y
    grad_kernel = X.T @ resid + l2_lambda * kernel
    grad_bias = np.sum(resid)
    grad = np.concatenate([grad_kernel, [grad_bias]])
    return nll, grad


def fit_poisson_glm(
    stimulus: np.ndarray, spike_counts: np.ndarray, kernel_length: int, l2_lambda: float = 1.0
):
    """Fit a Poisson GLM (exponential link) kernel + bias by L-BFGS-B on the analytic gradient.

    Returns (kernel_hat, bias_hat, scipy.optimize.OptimizeResult).
    """
    X = build_design_matrix(stimulus, kernel_length)
    theta0 = np.zeros(kernel_length + 1)
    result = minimize(
        _poisson_nll_and_grad, theta0, args=(X, spike_counts, l2_lambda), jac=True, method="L-BFGS-B"
    )
    kernel_hat, bias_hat = result.x[:-1], result.x[-1]
    return kernel_hat, bias_hat, result


if __name__ == "__main__":
    from encoding_models.lnp_simulation import simulate_lnp

    print("gradient check: analytic gradient vs. central finite differences...")
    rng = np.random.default_rng(0)
    X_check = rng.standard_normal((200, 10))
    y_check = rng.poisson(1.0, size=200)
    theta_check = 0.1 * rng.standard_normal(11)
    _, analytic_grad = _poisson_nll_and_grad(theta_check, X_check, y_check, l2_lambda=0.5)

    eps = 1e-6
    numeric_grad = np.zeros_like(theta_check)
    for i in range(len(theta_check)):
        perturbed = theta_check.copy()
        perturbed[i] += eps
        f_plus, _grad_plus = _poisson_nll_and_grad(perturbed, X_check, y_check, l2_lambda=0.5)
        perturbed[i] -= 2 * eps
        f_minus, _grad_minus = _poisson_nll_and_grad(perturbed, X_check, y_check, l2_lambda=0.5)
        numeric_grad[i] = (f_plus - f_minus) / (2 * eps)

    max_diff = float(np.abs(analytic_grad - numeric_grad).max())
    print(f"max |analytic - numeric| gradient difference: {max_diff:.2e}")
    assert max_diff < 1e-4, "analytic gradient should match finite differences"
    print("OK: analytic Poisson GLM gradient verified against finite differences.")

    print()
    print("fitting a GLM to simulated LNP data and comparing to ground truth...")
    sim = simulate_lnp(n_samples=20000, seed=0)
    kernel_hat, bias_hat, result = fit_poisson_glm(
        sim.stimulus, sim.spike_counts, len(sim.kernel), l2_lambda=2.0
    )
    mse = float(np.mean((kernel_hat - sim.kernel) ** 2))
    corr = float(np.corrcoef(kernel_hat, sim.kernel)[0, 1])
    print(f"converged: {result.success}, MSE vs ground truth: {mse:.4f}, correlation: {corr:.3f}")
    print(f"true bias: {sim.bias:.3f}, fitted bias: {bias_hat:.3f}")
    assert corr > 0.9, "GLM should recover the true kernel shape closely"
    print("OK: Poisson GLM recovers the ground-truth kernel from simulated spikes.")
