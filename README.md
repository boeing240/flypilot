# Fly Pilot — Drag Strip

A Drosophila mushroom-body-inspired controller (Kenyon cells -> dopamine-gated
MBON pools, plus a fast reflex path for the launch) driving a manual-shift
drag-strip car in a from-scratch physics sim. See the architecture notes in
[`flypilot/brain.py`](flypilot/brain.py) and especially
[`flypilot/FINDINGS.md`](flypilot/FINDINGS.md) for the actual research log —
what worked, what didn't, and why.

## Layout

- `flypilot/env.py` — physics: tree light, manual gearbox, wheel slip, curb
  sensors, slip-induced lateral pull, uncertainty knobs (engine derate,
  traction loss/jitter).
- `flypilot/sense.py` — sensor -> projection-neuron encoding.
- `flypilot/brain.py` — the mushroom-body-inspired controller (KC layer split
  into lateral/longitudinal compartments, MBON pools, node-perturbation
  plasticity).
- `flypilot/learning.py` — dopamine as reward-prediction error (scalar baseline
  for steering, TD(0) value baseline for throttle/shift).
- `flypilot/baseline.py` — hand-tuned rule-based controller, for comparison.
- `flypilot/train.py` — training loop (optionally with domain randomization)
  and the shared evaluation helpers.
- `flypilot/evolve.py` — population selection: train N individuals (varied
  seed / exploration noise / learning rate), evaluate, keep the best; can also
  score them on held-out scenarios during selection.
- `flypilot/scenarios.py`, `flypilot/evaluate_scenarios.py` — robustness
  testing across uncertainty scenarios without retraining.
- `flypilot/evaluate.py` — train one fly from scratch and compare it against
  the baseline.
- `models/` — the deployed brain (`flypilot_brain.pkl`).
- `results/` — its scenario results, the leaderboard of the run that produced
  it, and `evolution_log.json` (what each search round found).
- `runs/` — per-run output of `evolve.py` (`log.txt`, `leaderboard.json`,
  `best.pkl`); git-ignored.

## Status (see FINDINGS.md for the full log)

The deployed brain finishes 150/150 with 0 crashes on every scenario
(nominal, hot engine, cold greasy track, patchy grip, worst case), within
~0.2s of the rule-based baseline on each. Finding such an individual is
luck-heavy — roughly one in 15-30 population winners is both fast and fully
robust — so selection can now score robustness directly (`--robust-scenarios`)
and training can randomize conditions (`--randomize-prob`).

## Quickstart

```bash
pip install -r requirements.txt

# population selection; output goes to runs/<run-name>/
python -m flypilot.evolve --population 16 --train-episodes 4000 --eval-episodes 150 \
    --robust-scenarios cold_greasy_track worst_case --run-name my-run

# robustness across uncertainty scenarios (engine derate, traction loss/jitter)
python -m flypilot.evaluate_scenarios --brain runs/my-run/best.pkl --out runs/my-run/scenario_results.json
```

Run from the repo root (not inside `flypilot/`) so the `flypilot.*` module
imports resolve. Promote a run's `best.pkl` by copying it to
`models/flypilot_brain.pkl` and rerunning `evaluate_scenarios` with its
default paths.
