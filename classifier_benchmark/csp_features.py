"""Common Spatial Patterns (CSP) feature extraction.

CSP finds spatial filters that maximize the variance ratio between two
classes of multichannel signal -- the classic feature-engineering step
in motor-imagery BCI decoding (Blankertz et al., 2008). Implemented
directly via a generalized eigendecomposition (`scipy.linalg.eigh`)
rather than a library call, since the whole point is to fit exactly two
class covariance matrices we can reason about.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh


def _class_covariance(X: np.ndarray) -> np.ndarray:
    """Average, trace-normalized spatial covariance across epochs.

    Parameters
    ----------
    X : np.ndarray, shape (n_epochs, n_channels, n_times)
    """
    covs = np.empty((X.shape[0], X.shape[1], X.shape[1]))
    for i, epoch in enumerate(X):
        cov = epoch @ epoch.T
        covs[i] = cov / np.trace(cov)
    return covs.mean(axis=0)


class CSP:
    """Common Spatial Patterns spatial filters for binary discrimination.

    Fits generalized eigenvectors of the two classes' average spatial
    covariance matrices; the eigenvectors with the smallest and largest
    eigenvalues maximize the between-class variance ratio and become
    the spatial filters.

    Parameters
    ----------
    n_components : int, default 4
        Number of filter pairs to keep (2 * n_components total filters).
    """

    def __init__(self, n_components: int = 4) -> None:
        self.n_components = n_components
        self.filters_: np.ndarray | None = None  # (2*n_components, n_channels)
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> CSP:
        classes = np.unique(y)
        if classes.shape[0] != 2:
            raise ValueError(f"CSP requires exactly 2 classes, got {classes.shape[0]}")
        cov_a = _class_covariance(X[y == classes[0]])
        cov_b = _class_covariance(X[y == classes[1]])
        eigvals, eigvecs = eigh(cov_a, cov_a + cov_b)
        order = np.argsort(eigvals)
        eigvecs = eigvecs[:, order]
        k = min(self.n_components, eigvecs.shape[1] // 2)
        selected = np.concatenate([eigvecs[:, :k], eigvecs[:, -k:]], axis=1)
        self.filters_ = selected.T
        self.classes_ = classes
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project epochs through the fitted filters and return log-variance features."""
        if self.filters_ is None:
            raise RuntimeError("CSP must be fit before transform")
        projected = np.einsum("fc,ect->eft", self.filters_, X)
        variance = projected.var(axis=2)
        log_var = np.log(variance / variance.sum(axis=1, keepdims=True))
        return log_var

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n_epochs, n_channels, n_times = 100, 8, 150
    y = rng.integers(0, 2, size=n_epochs)
    X = rng.standard_normal((n_epochs, n_channels, n_times)) * 0.5
    t = np.linspace(0.0, 1.0, n_times)
    source = np.sin(2 * np.pi * 10 * t)
    half = n_channels // 2
    for i in range(n_epochs):
        amp = rng.uniform(1.5, 2.5)
        chans = slice(0, half) if y[i] == 0 else slice(half, n_channels)
        X[i, chans, :] += amp * source

    csp = CSP(n_components=3)
    feats = csp.fit_transform(X, y)
    assert csp.filters_ is not None
    assert csp.classes_ is not None
    print("CSP feature shape (expect (100, 6)):", feats.shape)
    print("filters_ shape (expect (6, 8)):", csp.filters_.shape)

    means_by_class = np.array([feats[y == c].mean(axis=0) for c in csp.classes_])
    separation = np.abs(means_by_class[0] - means_by_class[1]).mean()
    print(f"mean |feature difference| between classes: {separation:.3f} (should be well above 0)")
    assert separation > 0.5, "CSP should find a clearly discriminative projection on this synthetic signal"
    print("OK: CSP recovers spatially discriminative filters from synthetic motor-imagery-like data.")
