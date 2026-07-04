"""Shared types, synthetic data generators, and the pytest suite for the whole repo."""
from __future__ import annotations

from utils_and_tests.shared_types import lazy_getattr

__all__ = [
    "ChannelsByTime",
    "EpochsChannelsTimes",
    "TaintedLike",
    "lazy_getattr",
    "synthetic_multichannel_signal",
    "synthetic_two_class_epochs",
]

_EXPORTS = {
    "ChannelsByTime": "utils_and_tests.shared_types",
    "EpochsChannelsTimes": "utils_and_tests.shared_types",
    "TaintedLike": "utils_and_tests.shared_types",
    "lazy_getattr": "utils_and_tests.shared_types",
    "synthetic_multichannel_signal": "utils_and_tests.synthetic_data",
    "synthetic_two_class_epochs": "utils_and_tests.synthetic_data",
}

__getattr__ = lazy_getattr(_EXPORTS, __name__)
