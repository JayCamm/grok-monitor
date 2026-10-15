"""Frozen kill rules: prospectively validated "should I kill this run?" heuristics.

Each rule was frozen BEFORE scoring on its test set. Do not retune the
thresholds without re-running the full prospective protocol.

RULES
-----
mlp_kill_rule(pr_at_10k, peaked):
    At step 10,000, if PR has peaked (declining) but remains above 2.0,
    the run is predicted DEAD (will not grok). Otherwise CONTINUE.
    Validated: 7/8 on MLP LR sweep (seeds 54/55), 9/10 incl. baselines.
    One miss was conservative (predicted CONTINUE on a dead run).

transformer_kill_rule(pr_at_10k):
    At step 10,000, if PR remains above 5.0, predicted DEAD. Else CONTINUE.
    Validated: 7/7 on calibration (seeds 60-64), 4/4 on test (seeds 65-66)
    — 11/11 combined.
    NOTE: threshold differs from MLP (5.0 vs 2.0) — transformer dynamics
    are noisier; do not apply the MLP threshold to transformers.

BOUNDARIES (where these do NOT apply)
-------------------------------------
- The timing precursor (PR collapse predicts WHEN grokking happens) works
  on MLPs (3/3 prospective) but NOT on transformers (collapse lags).
- Neither rule is validated outside modular-arithmetic grokking.
- GARD chemistry, bioelectric networks, and Kuramoto oscillators do not
  show the signature at all — do not apply these rules there.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class KillRule:
    name: str
    decision_step: int
    pr_threshold: float
    requires_peak: bool
    validation: str


MLP_KILL_RULE = KillRule(
    name="mlp_kill",
    decision_step=10_000,
    pr_threshold=2.0,
    requires_peak=True,
    validation="7/8 prospective (MLP LR sweep, seeds 54/55); 9/10 incl. baselines",
)

TRANSFORMER_KILL_RULE = KillRule(
    name="transformer_kill",
    decision_step=10_000,
    pr_threshold=5.0,
    requires_peak=False,
    validation="7/7 calibration (seeds 60-64); 4/4 test (seeds 65-66); 11/11 combined",
)


def apply_kill_rule(rule: KillRule, pr_at_decision: float, peaked: bool) -> str:
    """Apply a frozen kill rule. Returns 'KILL' or 'CONTINUE'.

    Args:
        rule: the frozen KillRule to apply.
        pr_at_decision: PR value at rule.decision_step.
        peaked: whether PR has peaked (is declining). Only used if
            rule.requires_peak is True.
    """
    if rule.requires_peak and not peaked:
        return "CONTINUE"
    return "KILL" if pr_at_decision > rule.pr_threshold else "CONTINUE"
