"""Smoke tests for grok_monitor. Run with: python -m pytest tests/ -v"""
import torch
import pytest

from grok_monitor.precursor import trajectory_pr, PRMonitor
from grok_monitor.kill_rule import (
    MLP_KILL_RULE, TRANSFORMER_KILL_RULE, apply_kill_rule,
)


def test_pr_coherent():
    # all probes move together -> PR ~ 1
    t = torch.linspace(0, 3, 21).unsqueeze(1)
    L = t @ torch.ones(1, 50) + 0.01 * torch.randn(21, 50)
    assert trajectory_pr(L) < 2.0


def test_pr_fragmented():
    # independent probes -> PR large (bounded by n_steps=21)
    L = torch.randn(21, 50)
    assert trajectory_pr(L) > 10.0


def test_pr_monitor_window():
    m = PRMonitor(n_probes=10, window=5, probe_every=1)
    for i in range(4):
        m.record(torch.randn(10), i)
    assert m.pr() is None  # window not full
    m.record(torch.randn(10), 4)
    assert m.pr() is not None
    assert len(m.history()) == 1


def test_mlp_kill_rule():
    assert apply_kill_rule(MLP_KILL_RULE, 7.0, peaked=True) == "KILL"
    assert apply_kill_rule(MLP_KILL_RULE, 1.2, peaked=True) == "CONTINUE"
    assert apply_kill_rule(MLP_KILL_RULE, 7.0, peaked=False) == "CONTINUE"


def test_transformer_kill_rule():
    assert apply_kill_rule(TRANSFORMER_KILL_RULE, 9.0, peaked=True) == "KILL"
    assert apply_kill_rule(TRANSFORMER_KILL_RULE, 1.5, peaked=True) == "CONTINUE"
    # transformer rule ignores peaked flag
    assert apply_kill_rule(TRANSFORMER_KILL_RULE, 9.0, peaked=False) == "KILL"


def test_rules_are_frozen():
    with pytest.raises(Exception):
        MLP_KILL_RULE.pr_threshold = 999.0
