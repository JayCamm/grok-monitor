# grok_monitor

Early-warning instruments for grokking runs: a kill rule that predicts
doomed training runs, and a precursor instrument that predicts
generalization jumps.

## What it does

**Kill rule** — at a fixed decision step, measures the participation
ratio (PR) of probe-loss trajectories. If probes' losses fragmented but
never re-synchronized, the run is predicted dead. Kill it, save the
compute.

**Precursor** — the same PR instrument, watched continuously. A sharp
rise-and-collapse of PR predicts an upcoming generalization jump on MLPs.

## Validation (prospective, frozen rules)

| Rule | Score | Notes |
|------|-------|-------|
| MLP kill (PR>2.0 @ step 10k) | 7/8 | LR sweep, seeds 54/55. 1 conservative miss. |
| Transformer kill (PR>5.0 @ step 10k) | 11/11 | 7/7 calib + 4/4 test, seeds 60–66. |
| MLP precursor (PR collapse → grok) | 3/3 | ~3000-step lead, frozen rule. |

## Boundaries — read before using

- The **timing precursor does NOT transfer to transformers**. Collapse
  lags grokking there. Only the kill rule is validated cross-architecture.
- Neither rule is validated outside modular-arithmetic grokking.
- The signature does not appear in GARD chemistry, bioelectric networks,
  or Kuramoto oscillators. Do not apply these rules there.
- Thresholds are frozen. Retuning them without re-running the prospective
  protocol voids the validation.

## Usage

```python
from grok_monitor import GrokMonitor, MLP_KILL_RULE

monitor = GrokMonitor(
    probe_fn=lambda: model(probe_a, probe_b),
    probe_targets=probe_y,
    n_probes=300,
    kill_rule=MLP_KILL_RULE,
)
for step in range(max_steps):
    train_step(...)
    if monitor.step(step, model) == "KILL":
        break  # predicted dead — your call
```

## Running tests

```
python -m pytest tests/ -v
```
