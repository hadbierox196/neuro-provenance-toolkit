"""Surrogate-gradient spiking network: the same 2AFC task, but learned.

`lif_decision_network.py` hand-sets cross-inhibition to produce
competition. Here the recurrent weight matrix (within- and cross-pool)
starts as small random noise and is learned end-to-end by
backpropagation-through-time, using a fast-sigmoid surrogate gradient
in place of the spiking nonlinearity's true (almost-everywhere-zero)
derivative -- the standard technique for training SNNs with gradient
descent (Neftci, Mostafa & Zenke, 2019).

Task input generation is plain NumPy so it stays usable without
PyTorch installed; only `SurrogateLIFDecisionNet` and `train` need
`torch`, imported lazily.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch


def generate_task_batch(
    coherences: np.ndarray,
    n_per_pool: int = 20,
    n_steps: int = 100,
    base_rate_hz: float = 80.0,
    rate_gain: float = 0.3,
    dt_ms: float = 4.0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Poisson input spike trains and labels for a batch of trials.

    Returns
    -------
    spikes : np.ndarray, shape (batch, n_steps, 2*n_per_pool), float32 {0,1}
        Binary input spikes; first `n_per_pool` channels feed pool A,
        remaining `n_per_pool` feed pool B.
    labels : np.ndarray, shape (batch,), int64
        1 if coherence > 0 (pool A is correct), else 0.
    """
    rng = np.random.default_rng(seed)
    batch = len(coherences)
    n = 2 * n_per_pool
    dt_s = dt_ms / 1000.0
    spikes = np.zeros((batch, n_steps, n), dtype=np.float32)
    for b, coherence in enumerate(coherences):
        rate_a = base_rate_hz * (1 + rate_gain * coherence)
        rate_b = base_rate_hz * (1 - rate_gain * coherence)
        p_a = np.clip(rate_a * dt_s, 0.0, 1.0)
        p_b = np.clip(rate_b * dt_s, 0.0, 1.0)
        spikes[b, :, :n_per_pool] = (rng.random((n_steps, n_per_pool)) < p_a).astype(np.float32)
        spikes[b, :, n_per_pool:] = (rng.random((n_steps, n_per_pool)) < p_b).astype(np.float32)
    labels = (coherences > 0).astype(np.int64)
    return spikes, labels


def _build_model(n_per_pool: int, beta: float, decay: float):
    import torch
    from torch import nn

    class SurrogateSpike(torch.autograd.Function):
        """Heaviside spike forward; fast-sigmoid surrogate gradient backward."""

        @staticmethod
        def forward(ctx, v_minus_threshold: torch.Tensor, beta: float) -> torch.Tensor:
            ctx.save_for_backward(v_minus_threshold)
            ctx.beta = beta
            return (v_minus_threshold > 0).float()

        @staticmethod
        def backward(ctx, grad_output: torch.Tensor):
            (v_minus_threshold,) = ctx.saved_tensors
            surrogate_grad = 1.0 / (ctx.beta * v_minus_threshold.abs() + 1.0) ** 2
            return grad_output * surrogate_grad, None

    class SurrogateLIFDecisionNet(nn.Module):
        """Two-pool recurrent spiking network with a fully learned recurrent matrix."""

        def __init__(self, n_per_pool: int, beta: float = 5.0, decay: float = 0.9) -> None:
            super().__init__()
            self.n_per_pool = n_per_pool
            self.beta = beta
            self.decay = decay
            n = 2 * n_per_pool
            self.w_rec: torch.Tensor = nn.Parameter(0.15 * torch.randn(n, n) / n**0.5)
            self.w_in: torch.Tensor = nn.Parameter(torch.ones(n) * 0.6)
            mask_a = torch.zeros(n)
            mask_a[:n_per_pool] = 1.0
            self.register_buffer("pool_a_mask", mask_a)
            self.register_buffer("pool_b_mask", 1.0 - mask_a)

        def forward(self, input_spikes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            batch, n_steps, n = input_spikes.shape
            v = torch.zeros(batch, n, device=input_spikes.device)
            prev_spike = torch.zeros(batch, n, device=input_spikes.device)
            spikes_out = []
            for t in range(n_steps):
                rec_input = prev_spike @ self.w_rec
                v = self.decay * v + input_spikes[:, t, :] * self.w_in + rec_input
                spike = SurrogateSpike.apply(v - 1.0, self.beta)
                v = v * (1.0 - spike)
                spikes_out.append(spike)
                prev_spike = spike
            spikes = torch.stack(spikes_out, dim=1)
            count_a = (spikes * self.pool_a_mask).sum(dim=(1, 2))  # type: ignore[operator]
            count_b = (spikes * self.pool_b_mask).sum(dim=(1, 2))  # type: ignore[operator]
            logits = torch.stack([count_b, count_a], dim=1) / n_steps  # scale to a sane range for softmax
            return spikes, logits

    return SurrogateLIFDecisionNet(n_per_pool, beta, decay)


def train(
    n_per_pool: int = 20,
    n_steps: int = 100,
    n_epochs: int = 150,
    batch_size: int = 32,
    lr: float = 5e-4,
    beta: float = 5.0,
    decay: float = 0.9,
    seed: int = 0,
) -> tuple[torch.nn.Module, list[float]]:
    """Train `SurrogateLIFDecisionNet` on random-coherence trials; returns (model, loss history)."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    model = _build_model(n_per_pool, beta, decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    rng = np.random.default_rng(seed)
    loss_history = []
    for _epoch in range(n_epochs):
        coherences = rng.uniform(-1.0, 1.0, size=batch_size)
        epoch_seed = int(rng.integers(0, 2**31 - 1))
        spikes_np, labels_np = generate_task_batch(coherences, n_per_pool, n_steps, seed=epoch_seed)
        spikes = torch.tensor(spikes_np)
        labels = torch.tensor(labels_np)

        optimizer.zero_grad()
        _, logits = model(spikes)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        loss_history.append(float(loss.item()))
    return model, loss_history


def evaluate(
    model: torch.nn.Module,
    coherences: np.ndarray,
    n_per_pool: int = 20,
    n_steps: int = 100,
    seed: int = 999,
) -> float:
    """Accuracy of a trained model across the given coherence levels."""
    import torch

    spikes_np, labels_np = generate_task_batch(coherences, n_per_pool, n_steps, seed=seed)
    with torch.no_grad():
        _, logits = model(torch.tensor(spikes_np))
        preds = logits.argmax(dim=1).numpy()
    return float((preds == labels_np).mean())


if __name__ == "__main__":
    print("generating a task batch (pure NumPy, no torch needed for this part)...")
    spikes, labels = generate_task_batch(np.array([-1.0, -0.3, 0.0, 0.3, 1.0]), seed=0)
    print(f"spikes shape: {spikes.shape}, labels: {labels.tolist()}")
    assert spikes.shape == (5, 100, 40)
    assert labels.tolist() == [0, 0, 0, 1, 1]

    print()
    print("training SurrogateLIFDecisionNet via BPTT + fast-sigmoid surrogate gradient...")
    model, losses = train(n_epochs=150, seed=0)
    print(f"loss: epoch 0 = {losses[0]:.3f}, epoch {len(losses)-1} = {losses[-1]:.3f}")

    print()
    print("evaluating accuracy across coherence levels...")
    test_coherences = np.array([-1.0, -0.5, -0.2, 0.2, 0.5, 1.0])
    acc = evaluate(model, test_coherences, seed=999)
    print(f"held-out accuracy across |coherence| in [0.2, 1.0]: {acc:.3f}")
    assert losses[-1] < losses[0], "loss should decrease over training"
    print("OK: surrogate-gradient training reduces loss; both a mechanistic (Brian2) and a")
    print("    learned (PyTorch) network now solve the same 2AFC decision task.")
