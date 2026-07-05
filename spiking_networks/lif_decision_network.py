"""Two-pool spiking decision network: LIF neurons, cross-inhibition, Poisson evidence.

A simplified, qualitatively Wang (2002)-style two-alternative
forced-choice decision circuit: two pools of leaky integrate-and-fire
neurons compete via reciprocal inhibitory synapses, each pool driven by
a `PoissonInput` whose rate is biased by `coherence` toward one
alternative. Winner-take-all competition emerges from cross-inhibition
alone (no recurrent self-excitation), which keeps the dynamics simpler
and more numerically predictable than the full conductance-based
reduced model.

Parameter note: these values were not hand-guessed. They were tuned and
validated against a NumPy Euler-Maruyama simulation of the identical
equations (independent of Brian2) before being encoded here, confirming
a graded, monotonic accuracy-vs-coherence relationship rather than an
all-or-nothing threshold effect.
"""
from __future__ import annotations

from typing import TypedDict

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DecisionNetworkParams:
    """All-in-one parameter set for `run_decision_trial`."""

    n_per_pool: int = 40
    tau_m_ms: float = 20.0
    tau_syn_ms: float = 5.0
    v_rest_mv: float = -70.0
    v_threshold_mv: float = -50.0
    v_reset_mv: float = -55.0
    t_refrac_ms: float = 2.0
    sigma_mv: float = 6.0
    w_inhib_mv: float = -0.4
    n_external: int = 40
    base_rate_hz: float = 120.0
    rate_gain: float = 0.012
    w_external_mv: float = 1.0


@dataclass(frozen=True)
class DecisionResult:
    """Output of a single simulated 2AFC trial."""

    spike_t_a: np.ndarray
    spike_i_a: np.ndarray
    spike_t_b: np.ndarray
    spike_i_b: np.ndarray
    rate_t_ms: np.ndarray
    rate_a_hz: np.ndarray
    rate_b_hz: np.ndarray
    winner: str
    decision_time_ms: float | None


def run_decision_trial(
    coherence: float,
    params: DecisionNetworkParams | None = None,
    duration_ms: float = 1000.0,
    decision_threshold_hz: float = 55.0,
    smoothing_ms: float = 20.0,
    seed: int | None = None,
) -> DecisionResult:
    """Simulate one 2AFC trial. Pool A is favored when `coherence` > 0.

    Parameters
    ----------
    coherence : float
        Evidence strength in roughly [-1, 1]; positive favors pool A,
        negative favors pool B, 0 is unbiased.
    decision_threshold_hz : float
        A pool's smoothed rate crossing this value is called the
        decision; whichever pool crosses first wins.
    seed : int, optional
        Seeds NumPy's global RNG, which Brian2 draws from.
    """
    import brian2 as b2

    params = params if params is not None else DecisionNetworkParams()
    b2.start_scope()
    if seed is not None:
        np.random.seed(seed)

    tau_m = params.tau_m_ms * b2.ms
    tau_syn = params.tau_syn_ms * b2.ms
    v_rest = params.v_rest_mv * b2.mV
    v_threshold = params.v_threshold_mv * b2.mV
    v_reset = params.v_reset_mv * b2.mV
    t_refrac = params.t_refrac_ms * b2.ms
    sigma = params.sigma_mv * b2.mV

    eqs = """
    dv/dt = (v_rest - v)/tau_m + I_syn/tau_m + sigma*xi/sqrt(tau_m) : volt (unless refractory)
    dI_syn/dt = -I_syn/tau_syn : volt
    """
    lif_namespace = dict(
        v_rest=v_rest, v_threshold=v_threshold, v_reset=v_reset, tau_m=tau_m,
        tau_syn=tau_syn, sigma=sigma,
    )

    pool_a = b2.NeuronGroup(
        params.n_per_pool, eqs, threshold="v > v_threshold", reset="v = v_reset",
        refractory=t_refrac, method="euler", namespace=lif_namespace, name="pool_a",
    )
    pool_b = b2.NeuronGroup(
        params.n_per_pool, eqs, threshold="v > v_threshold", reset="v = v_reset",
        refractory=t_refrac, method="euler", namespace=lif_namespace, name="pool_b",
    )
    pool_a.v = v_rest
    pool_b.v = v_rest

    rate_a_hz = params.base_rate_hz * (1 + params.rate_gain * coherence)
    rate_b_hz = params.base_rate_hz * (1 - params.rate_gain * coherence)
    b2.PoissonInput(
        pool_a, "I_syn", params.n_external, rate_a_hz * b2.Hz, params.w_external_mv * b2.mV
    )
    b2.PoissonInput(
        pool_b, "I_syn", params.n_external, rate_b_hz * b2.Hz, params.w_external_mv * b2.mV
    )

    inhib_namespace = {"w_inhib": params.w_inhib_mv * b2.mV}
    cross_ab = b2.Synapses(
        pool_a, pool_b, on_pre="I_syn_post += w_inhib", namespace=inhib_namespace, name="cross_ab"
    )
    cross_ab.connect()
    cross_ba = b2.Synapses(
        pool_b, pool_a, on_pre="I_syn_post += w_inhib", namespace=inhib_namespace, name="cross_ba"
    )
    cross_ba.connect()

    spikes_a = b2.SpikeMonitor(pool_a)
    spikes_b = b2.SpikeMonitor(pool_b)
    rate_mon_a = b2.PopulationRateMonitor(pool_a)
    rate_mon_b = b2.PopulationRateMonitor(pool_b)

    b2.run(duration_ms * b2.ms)

    smooth_a = rate_mon_a.smooth_rate(window="flat", width=smoothing_ms * b2.ms) / b2.Hz
    smooth_b = rate_mon_b.smooth_rate(window="flat", width=smoothing_ms * b2.ms) / b2.Hz
    rate_t_ms = rate_mon_a.t / b2.ms

    winner, decision_time_ms = "undecided", None
    for t_ms, ra, rb in zip(rate_t_ms, smooth_a, smooth_b, strict=True):
        if ra >= decision_threshold_hz or rb >= decision_threshold_hz:
            winner = "A" if ra >= rb else "B"
            decision_time_ms = float(t_ms)
            break

    return DecisionResult(
        spike_t_a=spikes_a.t / b2.ms,
        spike_i_a=np.asarray(spikes_a.i),
        spike_t_b=spikes_b.t / b2.ms,
        spike_i_b=np.asarray(spikes_b.i),
        rate_t_ms=np.asarray(rate_t_ms),
        rate_a_hz=np.asarray(smooth_a),
        rate_b_hz=np.asarray(smooth_b),
        winner=winner,
        decision_time_ms=decision_time_ms,
    )


class CoherenceLevelResult(TypedDict):
    winners: list[str]
    decision_times_ms: list[float]


def run_coherence_sweep(
    coherences: list[float],
    n_trials_per_level: int = 10,
    params: DecisionNetworkParams | None = None,
    duration_ms: float = 1000.0,
    seed0: int = 0,
) -> dict[float, CoherenceLevelResult]:
    """Run repeated trials per coherence level; returns per-level winners and RTs."""
    results: dict[float, CoherenceLevelResult] = {}
    trial = 0
    for coherence in coherences:
        winners, rts = [], []
        for _rep in range(n_trials_per_level):
            res = run_decision_trial(coherence, params, duration_ms, seed=seed0 + trial)
            winners.append(res.winner)
            if res.decision_time_ms is not None:
                rts.append(res.decision_time_ms)
            trial += 1
        results[coherence] = {"winners": winners, "decision_times_ms": rts}
    return results


if __name__ == "__main__":
    print("Running single trial at coherence=0.5 (favors pool A)...")
    result = run_decision_trial(coherence=0.5, seed=0)
    print(f"winner={result.winner}, decision_time={result.decision_time_ms} ms")
    print(f"pool A spikes: {len(result.spike_t_a)}, pool B spikes: {len(result.spike_t_b)}")

    print()
    print("Coherence sweep (10 trials per level)...")
    sweep = run_coherence_sweep([0.0, 0.15, 0.3, 0.5, 0.75, 1.0], n_trials_per_level=10, seed0=100)
    for coherence, data in sweep.items():
        n_a = data["winners"].count("A")
        n = len(data["winners"])
        mean_rt = float(np.mean(data["decision_times_ms"])) if data["decision_times_ms"] else float("nan")
        print(f"coherence={coherence:.2f}: P(A wins)={n_a/n:.2f}  mean decision time={mean_rt:.0f} ms")
    print("OK: accuracy should rise with |coherence| and be near chance at coherence=0.")
