"""Filter, re-reference, and ICA-clean continuous EEG via MNE-Python."""
from __future__ import annotations

from utils_and_tests.shared_types import lazy_getattr

__all__ = [
    "PreprocessConfig",
    "PreprocessReport",
    "run_preprocessing",
    "compare_psd",
    "fit_ica_and_find_artifacts",
]

_EXPORTS = {
    "PreprocessConfig": "eeg_preprocessing.pipeline",
    "PreprocessReport": "eeg_preprocessing.pipeline",
    "run_preprocessing": "eeg_preprocessing.pipeline",
    "compare_psd": "eeg_preprocessing.pipeline",
    "fit_ica_and_find_artifacts": "eeg_preprocessing.ica_utils",
}

__getattr__ = lazy_getattr(_EXPORTS, __name__)
