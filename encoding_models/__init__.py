"""Linear-Nonlinear-Poisson encoding models: simulation, GLM fit, STA baseline."""
from __future__ import annotations

from utils_and_tests.shared_types import lazy_getattr

__all__ = [
    "LNPSimulation",
    "simulate_lnp",
    "make_biphasic_kernel",
    "build_design_matrix",
    "fit_poisson_glm",
    "spike_triggered_average",
    "normalized_correlation",
]

_EXPORTS = {
    "LNPSimulation": "encoding_models.lnp_simulation",
    "simulate_lnp": "encoding_models.lnp_simulation",
    "make_biphasic_kernel": "encoding_models.lnp_simulation",
    "build_design_matrix": "encoding_models.lnp_simulation",
    "fit_poisson_glm": "encoding_models.glm_fit",
    "spike_triggered_average": "encoding_models.receptive_field_baseline",
    "normalized_correlation": "encoding_models.receptive_field_baseline",
}

__getattr__ = lazy_getattr(_EXPORTS, __name__)
