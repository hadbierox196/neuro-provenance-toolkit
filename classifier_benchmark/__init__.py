"""Motor-imagery classification benchmark with a provenance_pipeline guardrail."""
from __future__ import annotations

from utils_and_tests.shared_types import lazy_getattr

__all__ = [
    "CSP",
    "CSPClassifier",
    "load_motor_imagery_epochs",
    "inject_synthetic_contamination",
    "train_one_fold",
    "k_fold_indices",
    "roc_auc_binary",
    "confusion_matrix_binary",
]

_EXPORTS = {
    "CSP": "classifier_benchmark.csp_features",
    "CSPClassifier": "classifier_benchmark.model",
    "load_motor_imagery_epochs": "classifier_benchmark.train_eval",
    "inject_synthetic_contamination": "classifier_benchmark.train_eval",
    "train_one_fold": "classifier_benchmark.train_eval",
    "k_fold_indices": "classifier_benchmark.train_eval",
    "roc_auc_binary": "classifier_benchmark.train_eval",
    "confusion_matrix_binary": "classifier_benchmark.train_eval",
}

__getattr__ = lazy_getattr(_EXPORTS, __name__)
