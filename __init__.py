"""grok_monitor: early-warning instruments for grokking runs."""
from .precursor import PRMonitor, trajectory_pr
from .kill_rule import (
    KillRule, MLP_KILL_RULE, TRANSFORMER_KILL_RULE, apply_kill_rule,
)
from .callback import GrokMonitor

__all__ = [
    "PRMonitor", "trajectory_pr",
    "KillRule", "MLP_KILL_RULE", "TRANSFORMER_KILL_RULE", "apply_kill_rule",
    "GrokMonitor",
]
