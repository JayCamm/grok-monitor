"""PyTorch training callback: kill-rule monitoring for grokking runs.

Drop-in usage::

    from grok_monitor import GrokMonitor, MLP_KILL_RULE

    monitor = GrokMonitor(
        probe_fn=lambda: model(probe_a, probe_b),  # returns logits
        probe_targets=probe_y,                      # 1D tensor
        n_probes=300,
        kill_rule=MLP_KILL_RULE,
    )

    for step in range(max_steps):
        train_step(...)
        verdict = monitor.step(step, model)
        if verdict == "KILL":
            print(f"Killing run at step {step}: predicted dead.")
            break
        if verdict == "GROK":
            print(f"Run healthy at step {step}.")

The monitor records probe losses, computes PR, and applies the frozen
kill rule at the decision step. It does NOT stop your training for you —
it returns a verdict string; you decide what to do with it.
"""
import torch
import torch.nn as nn

from .precursor import PRMonitor
from .kill_rule import KillRule, apply_kill_rule


class GrokMonitor:
    def __init__(
        self,
        probe_fn,
        probe_targets: torch.Tensor,
        n_probes: int,
        kill_rule: KillRule,
        probe_every: int = 25,
        window: int = 21,
        loss_fn=None,
    ):
        self.probe_fn = probe_fn
        self.probe_targets = probe_targets
        self.kill_rule = kill_rule
        self.probe_every = probe_every
        self._pr = PRMonitor(n_probes=n_probes, window=window,
                             probe_every=probe_every)
        self._loss_fn = loss_fn or nn.CrossEntropyLoss(reduction="none")
        self._verdict = "CONTINUE"
        self._decided = False

    def step(self, step: int, model: nn.Module) -> str:
        """Call once per training step. Returns current verdict."""
        if self._decided:
            return self._verdict
        if step % self.probe_every == 0:
            was_training = model.training
            model.eval()
            with torch.no_grad():
                logits = self.probe_fn()
                losses = self._loss_fn(logits, self.probe_targets)
            if was_training:
                model.train()
            self._pr.record(losses, step)
        if step >= self.kill_rule.decision_step and not self._decided:
            self._verdict = self._decide(step)
            self._decided = True
        return self._verdict

    def _decide(self, step: int) -> str:
        hist = self._pr.history()
        if not hist:
            return "CONTINUE"
        # PR at decision step (nearest recorded)
        _, pr_now = min(hist, key=lambda x: abs(x[0] - step))
        peak_val = max(p for _, p in hist)
        peaked = pr_now < peak_val - 0.5
        return apply_kill_rule(self.kill_rule, pr_now, peaked)

    @property
    def pr_history(self):
        return self._pr.history()
