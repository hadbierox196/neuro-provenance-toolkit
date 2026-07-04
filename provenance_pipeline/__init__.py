"""Provenance-aware taint tracking for neural preprocessing pipelines.

Public names are resolved lazily (PEP 562, via the shared
`utils_and_tests.shared_types.lazy_getattr` factory) so that ``python
-m provenance_pipeline.<submodule>`` can run a submodule's own demo
without this file eagerly importing every submodule first.
"""
from __future__ import annotations

from utils_and_tests.shared_types import lazy_getattr

__all__ = [
    "TaintedArray",
    "TaintLevel",
    "ProvenanceRecord",
    "apply_temporal_filter",
    "apply_average_reference",
    "wrap_opaque_transform",
    "TaintPolicy",
    "TaintViolationError",
    "assert_clean",
    "filter_clean_epochs",
]

_EXPORTS = {
    "TaintedArray": "provenance_pipeline.tainted_array",
    "TaintLevel": "provenance_pipeline.tainted_array",
    "ProvenanceRecord": "provenance_pipeline.tainted_array",
    "apply_temporal_filter": "provenance_pipeline.propagation",
    "apply_average_reference": "provenance_pipeline.propagation",
    "wrap_opaque_transform": "provenance_pipeline.propagation",
    "TaintPolicy": "provenance_pipeline.guardrail",
    "TaintViolationError": "provenance_pipeline.guardrail",
    "assert_clean": "provenance_pipeline.guardrail",
    "filter_clean_epochs": "provenance_pipeline.guardrail",
}

__getattr__ = lazy_getattr(_EXPORTS, __name__)
