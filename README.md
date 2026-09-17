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
- `flypilot/brain.py` — the mushroom-body-inspired controller (KC layer,
  MBON pools, node-perturbation plasticity, compartmentalized dopamine).
- `flypilot/learning.py` — dopamine as reward-prediction-error.
- `flypilot/baseline.py` — hand-tuned rule-based controller, for comparison.
- `flypilot/train.py` — single-brain training loop.
- `flypilot/evolve.py` — population selection: train N individuals (varied
  seed / exploration noise / learning rate), evaluate, keep the best.
- `flypilot/scenarios.py`, `flypilot/evaluate_scenarios.py` — robustness
  testing across uncertainty scenarios without retraining.
- `flypilot/replay.py` — records a per-tick trace for the visualizer.
- `viz/index.html` — the visualization (published as a Claude Artifact);
  reads `replays.json` / `scenario_results.json`.

## Status (see FINDINGS.md for the full log)

Steering is solid — the dense centering reward gets 0 crashes even under
added slip-induced lateral pull. Throttle/shift is still a work in progress:
reward shaping keeps uncovering new equilibria (redline-forever in 1st gear,
instant-upshift-to-dodge-slip-penalty, near-zero-throttle-to-avoid-all-risk)
faster than hand-tuned coefficients can chase them down. Was mid-way through
switching from manual coefficient tuning to population selection
(`evolve.py`) when this was paused — that's the next thing to pick up.

## Quickstart

```bash
pip install -r requirements.txt

# train one brain
python -c "from flypilot.train import train; import pickle; b=train(n_episodes=5000); pickle.dump(b, open('flypilot_brain.pkl','wb'))"

# or run population selection instead of hand-tuning reward coefficients
python -m flypilot.evolve --population 8 --train-episodes 4000 --eval-episodes 150

# robustness across uncertainty scenarios (engine derate, traction loss/jitter)
python -m flypilot.evaluate_scenarios --episodes 150
```

Run from the repo root (not inside `flypilot/`) so the `flypilot.*` module
imports resolve.

Then open `viz/index.html` (with `replays.json` / `scenario_results.json`
next to it, or served via `python -m http.server`) to watch it drive.
