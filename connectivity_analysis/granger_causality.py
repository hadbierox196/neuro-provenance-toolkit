"""Granger causality via VAR(p) least squares and a restricted-vs-full F-test.

No statsmodels: both the restricted model (x predicted from its own
past) and the full model (x predicted from its own past plus y's past)
are fit directly with `numpy.linalg.lstsq`, and "does y Granger-cause
x" is answered by an F-test on whether adding y's lags significantly
reduces the residual sum of squares (Granger, 1969).
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def _lag_matrix(series: np.ndarray, lag: int) -> np.ndarray:
    """Row t (t=lag..n-1) holds [series[t-1], series[t-2], ..., series[t-lag]]."""
    n = len(series)
    return np.column_stack([series[lag - k - 1 : n - k - 1] for k in range(lag)])


def granger_causality_fstat(x: np.ndarray, y: np.ndarray, lag: int) -> tuple[float, float]:
    """Test whether y Granger-causes x: does adding y's lags improve prediction of x?

    Returns (F_statistic, p_value). A small p-value means y's past
    significantly improves prediction of x beyond x's own past alone.
    """
    n = len(x)
    x_target = x[lag:]
    x_lags = _lag_matrix(x, lag)
    y_lags = _lag_matrix(y, lag)

    design_restricted = np.column_stack([np.ones(n - lag), x_lags])
    design_full = np.column_stack([np.ones(n - lag), x_lags, y_lags])

    beta_r, _, _, _ = np.linalg.lstsq(design_restricted, x_target, rcond=None)
    beta_f, _, _, _ = np.linalg.lstsq(design_full, x_target, rcond=None)
    rss_r = float(np.sum((x_target - design_restricted @ beta_r) ** 2))
    rss_f = float(np.sum((x_target - design_full @ beta_f) ** 2))

    n_obs = n - lag
    df_full = n_obs - design_full.shape[1]
    f_stat = ((rss_r - rss_f) / lag) / (rss_f / df_full)
    p_value = float(stats.f.sf(f_stat, lag, df_full))
    return float(f_stat), p_value


def granger_causality_matrix(data: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise Granger F-stat and p-value matrices. matrix[i, j] = "does j Granger-cause i"."""
    n_channels = data.shape[0]
    f_matrix = np.full((n_channels, n_channels), np.nan)
    p_matrix = np.full((n_channels, n_channels), np.nan)
    for i in range(n_channels):
        for j in range(n_channels):
            if i == j:
                continue
            f_stat, p_value = granger_causality_fstat(data[i], data[j], lag)
            f_matrix[i, j] = f_stat
            p_matrix[i, j] = p_value
    return f_matrix, p_matrix


def simulate_coupled_ar(
    n_samples: int = 2000, coupling: float = 0.5, self_ar: float = 0.5, noise_std: float = 1.0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """X is an autonomous AR(1); Y depends on its own past AND X's past lag -- X->Y, not Y->X."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n_samples)
    y = np.zeros(n_samples)
    for t in range(1, n_samples):
        x[t] = self_ar * x[t - 1] + noise_std * rng.standard_normal()
        y[t] = self_ar * y[t - 1] + coupling * x[t - 1] + noise_std * rng.standard_normal()
    return x, y


if __name__ == "__main__":
    x, y = simulate_coupled_ar(n_samples=2000, coupling=0.6, seed=0)

    f_xy, p_xy = granger_causality_fstat(y, x, lag=2)  # does x Granger-cause y?
    f_yx, p_yx = granger_causality_fstat(x, y, lag=2)  # does y Granger-cause x?
    print(f"X -> Y (true coupling exists): F={f_xy:.2f}, p={p_xy:.2e}")
    print(f"Y -> X (no coupling, ground truth null): F={f_yx:.2f}, p={p_yx:.3f}")
    assert p_xy < 0.001, "true X->Y coupling should be highly significant"
    assert p_yx > 0.05, "the absent Y->X direction should NOT look significant"
    print("OK: Granger test recovers the correct causal direction and rejects the false one.")

    print()
    print("full pairwise matrix on a 3-channel system (channel 2 is independent noise)...")
    rng = np.random.default_rng(1)
    independent = rng.standard_normal(2000)
    data = np.stack([x, y, independent])
    f_matrix, p_matrix = granger_causality_matrix(data, lag=2)
    print("p-value matrix (rows/cols = 0:X, 1:Y, 2:independent; entry [i,j] = does j cause i):")
    print(np.round(p_matrix, 3))
    assert p_matrix[1, 0] < 0.001 and p_matrix[0, 1] > 0.05
    print("OK: matrix form agrees with the pairwise result.")

    from connectivity_analysis.spectral_coherence import plot_matrix_heatmap

    fig = plot_matrix_heatmap(
        np.nan_to_num(f_matrix, nan=0.0),
        title="Granger causality F-statistic (row caused-by column)",
        labels=["X", "Y", "independent"],
    )
    out_path = "/tmp/granger_heatmap.png"
    fig.savefig(out_path, dpi=120)
    print(f"saved heatmap to {out_path}")
