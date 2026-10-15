"""Loss-trajectory participation ratio (PR) instrument.

Measures the effective dimensionality of probe-loss trajectories over a
sliding window. Low PR (~1) means probes' losses move together (coherent).
High PR means they move independently (fragmented).

This is the core instrument behind the grokking precursor and kill rule.
See README.md for validation status and boundaries.
"""
import torch


def trajectory_pr(loss_window: torch.Tensor) -> float:
    """Participation ratio of a (time_steps x n_probes) loss matrix.

    PR = (sum eigenvalues)^2 / sum(eigenvalues^2) of the probe-loss
    covariance. Ranges from 1 (all probes move together) to n_probes
    (all independent).
    """
    if loss_window.shape[0] < 2:
        return 1.0
    Lc = loss_window - loss_window.mean(dim=0, keepdim=True)
    C = (Lc.T @ Lc) / (loss_window.shape[0] - 1)
    lam = torch.linalg.eigvalsh(C).clamp_min(0)
    denom = (lam ** 2).sum().item()
    if denom < 1e-12:
        return 1.0
    return (lam.sum().item() ** 2) / denom


class PRMonitor:
    """Tracks probe-loss PR over training.

    Usage:
        monitor = PRMonitor(n_probes=300, window=21, probe_every=25)
        ...
        for step in training:
            if step % probe_every == 0:
                monitor.record(probe_losses)  # 1D tensor, n_probes
            if step % eval_every == 0:
                pr = monitor.pr()  # current PR, or None if window not full
    """

    def __init__(self, n_probes: int, window: int = 21, probe_every: int = 25):
        self.n_probes = n_probes
        self.window = window
        self.probe_every = probe_every
        self._losses: list[torch.Tensor] = []
        self._pr_history: list[tuple[int, float]] = []
        self._step = 0

    def record(self, probe_losses: torch.Tensor, step: int):
        """Record a probe-loss vector. Call every probe_every steps."""
        assert probe_losses.shape == (self.n_probes,), \
            f"expected ({self.n_probes},), got {tuple(probe_losses.shape)}"
        self._losses.append(probe_losses.detach().cpu())
        if len(self._losses) > self.window:
            self._losses.pop(0)
        self._step = step
        if len(self._losses) == self.window:
            L = torch.stack(self._losses)
            self._pr_history.append((step, trajectory_pr(L)))

    def pr(self) -> float | None:
        """Current PR, or None if the window isn't full yet."""
        return self._pr_history[-1][1] if self._pr_history else None

    def history(self) -> list[tuple[int, float]]:
        """Full (step, PR) history."""
        return list(self._pr_history)

    def peak(self) -> tuple[int, float] | None:
        """(step, value) of the maximum PR so far, or None."""
        if not self._pr_history:
            return None
        return max(self._pr_history, key=lambda x: x[1])
