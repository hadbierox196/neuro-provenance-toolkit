"""Spectral coherence and Granger causality between channels."""
from __future__ import annotations

from utils_and_tests.shared_types import lazy_getattr

__all__ = [
    "pairwise_coherence",
    "band_coherence_matrix",
    "granger_causality_fstat",
    "granger_causality_matrix",
    "simulate_coupled_ar",
]

_EXPORTS = {
    "pairwise_coherence": "connectivity_analysis.spectral_coherence",
    "band_coherence_matrix": "connectivity_analysis.spectral_coherence",
    "granger_causality_fstat": "connectivity_analysis.granger_causality",
    "granger_causality_matrix": "connectivity_analysis.granger_causality",
    "simulate_coupled_ar": "connectivity_analysis.granger_causality",
}

__getattr__ = lazy_getattr(_EXPORTS, __name__)
