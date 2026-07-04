"""Tests for classifier_benchmark: CSP, metrics utilities, and the guardrail integration.

Training itself (`train_one_fold`) needs PyTorch; those tests are
skipped via `pytest.importorskip("torch")` rather than failing when it
isn't installed.
"""
from __future__ import annotations

import numpy as np
import pytest

from classifier_benchmark.csp_features import CSP
from classifier_benchmark.train_eval import (
    confusion_matrix_binary,
    inject_synthetic_contamination,
    k_fold_indices,
    load_motor_imagery_epochs,
    roc_auc_binary,
)
from provenance_pipeline import TaintLevel, TaintPolicy, filter_clean_epochs


def test_csp_separates_synthetic_classes(two_class_epochs: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = two_class_epochs
    csp = CSP(n_components=3)
    feats = csp.fit_transform(X, y)
    assert feats.shape == (X.shape[0], 6)
    means = np.array([feats[y == c].mean(axis=0) for c in np.unique(y)])
    assert np.abs(means[0] - means[1]).mean() > 0.3


def test_csp_requires_exactly_two_classes() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 4, 20))
    y = rng.integers(0, 3, size=30)  # 3 classes
    csp = CSP(n_components=2)
    with pytest.raises(ValueError):
        csp.fit(X, y)


def test_load_motor_imagery_epochs_falls_back_to_synthetic() -> None:
    X, y = load_motor_imagery_epochs(use_real_data=True, seed=0)
    assert X.ndim == 3
    assert set(np.unique(y)) == {0, 1}


def test_k_fold_indices_partition_without_overlap() -> None:
    splits = k_fold_indices(n=100, k=5, seed=0)
    assert len(splits) == 5
    all_val = np.concatenate([v for _, v in splits])
    assert sorted(all_val.tolist()) == list(range(100))
    for train_idx, val_idx in splits:
        assert len(set(train_idx.tolist()) & set(val_idx.tolist())) == 0


def test_roc_auc_binary_known_cases() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 1])
    assert abs(roc_auc_binary(y_true, np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])) - 1.0) < 1e-9
    assert abs(roc_auc_binary(y_true, np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])) - 0.0) < 1e-9


def test_confusion_matrix_binary() -> None:
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 0])
    cm = confusion_matrix_binary(y_true, y_pred)
    assert cm.tolist() == [[1, 1], [1, 2]]


def test_contamination_injection_is_caught_by_guardrail() -> None:
    X, _ = load_motor_imagery_epochs(use_real_data=False, seed=0)
    tainted, bad_idx = inject_synthetic_contamination(X, n_bad_epochs=6, seed=1)
    policy = TaintPolicy(max_taint_level=TaintLevel.CLEAN, max_contaminated_fraction=0.0)
    clean, keep_mask = filter_clean_epochs(tainted, policy)
    assert np.array_equal(np.flatnonzero(~keep_mask), bad_idx)
    assert clean.data.shape[0] == X.shape[0] - 6


def test_train_one_fold_runs_and_beats_chance() -> None:
    pytest.importorskip("torch")
    from classifier_benchmark.train_eval import train_one_fold

    X, y = load_motor_imagery_epochs(use_real_data=False, seed=0)
    n = len(X)
    split = int(0.8 * n)
    acc, auc, _, _ = train_one_fold(X[:split], y[:split], X[split:], y[split:])
    assert 0.0 <= acc <= 1.0
    assert acc > 0.6
