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
- `flypilot/league.py`, `flypilot/pilots.py`, `flypilot/web/` — the live
  league for streaming training: named pilots, Elo ranking, race replays, and an
  admin panel (see [Streaming](#streaming-the-training)).
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

## Streaming the training

```bash
python -m flypilot.league
```

Then open the **admin panel** at `http://127.0.0.1:8765/admin` (local-only),
press Start, and point OBS (Browser Source, 1920x1080) at
`http://127.0.0.1:8765/`. The program starts idle; a run is always started with the Start button in the admin panel (`/admin`).

A field of flies (11 learners plus the reigning champion as the "Legend") trains
continuously. Each round (Grand Prix) is one training block per pilot followed by
a frozen-policy race on shared conditions; finishing order updates an Elo rating,
which is the pilot ranking. Every fly gets a stable, human-looking name, number
and livery from its seed. At the end of each season the lowest-rated veteran
retires and a rookie with a fresh random genome takes the seat.
The rookie inherits the retiree's rating, so Elo points are never created. Each season
also crowns a champion (best rating; titles are kept on the fly and logged in
`stream_data/archive/seasons.json`), then every rating is pulled part of the way back to 1500
(*Rating carried into next season*, default 0.5; 1 keeps everything, 0 resets fully).

### Evolution and the analysis log

A rookie is normally a *child* of a good fly: a copy of a top-3 veteran's brain (or, 25% of the time,
of the reigning Legend) with a small mutation on the weights and on eta / exploration noise
(`flypilot/evolution.py`). A quarter of the rookies are still random immigrants, and the first field
is all random. The plasticity rule random-walks away from good solutions, so a fly that falls
`revert_gap` Elo below its peak returns to its best brain. All of it is set in the admin panel
(group *Evolution*).

Everything needed to analyse a run later is appended to `stream_data/analysis/*.jsonl` (survives
resets): `leagues` (full settings at start and every change), `pilots` (origin, parent, generation,
seed, genome on join; final stats on exit), `rounds` (one line per fly per race: placing, time,
rating before/after, training stats, eta/noise) and `events`.

    python -m flypilot.analysis                 # offspring vs. random flies, per generation
    python -m flypilot.analysis --csv out_dir   # flat tables for pandas / a spreadsheet

Rounds run back to back with no dead air: the next round trains while the page
is still playing the current one (the page reports which round it shows, and
training stays `lead_rounds` ahead of it). Each round has a start animation
(title card, lane tags, cars rolling onto the grid), an end animation (flag
wipe, podium, confetti) and a configurable countdown before the next one; if
training is late the page shows a training-progress screen instead of a gap.

The admin panel starts, pauses (after the current round) and resets the
league and edits every setting: league size, training per round, season and
retirement rules, pacing, and all broadcast timings. Settings are saved to
`stream_data/settings.json`; each is marked as taking effect instantly, at the
next round, or only for a new league. League state is checkpointed after every
round in `stream_data/` (git-ignored), so a restart resumes where it stopped.
The admin API only answers on loopback; use an SSH tunnel to reach it remotely.

### Results are never lost, and the goal keeps running

The league is not just a show: it keeps working on the project's goal, a
better champion (zero crashes and a lower total avg_time over the five
scenarios than `models/flypilot_brain.pkl`, the same criterion `search.py` uses).

- **Contenders.** A fly that races close to the champion on the same conditions
  (`verify_margin`) is saved to the hall of fame *in the exact state that
  raced* (continued training is a random walk, so peaks are captured when they
  happen) and verified in the background on all five scenarios with the same
  seeds and procedure as the champion's numbers.
- **Promotion.** A contender that really beats the champion becomes the new
  champion: `models/flypilot_brain.pkl`, `results/scenario_results.json` and
  `results/search_status.json` are updated (the previous champion is archived
  first), the promotion is appended to `results/league_promotions.jsonl`, and
  the fly is crowned the Legend on screen. Optional: commit (and push) the
  promotion from the admin panel. `search.py` and the league share the same
  champion files, so run one or the other, not both at once.
- **Nothing is deleted.** Retired, removed, dethroned and crowned flies (full
  learned state), previous champions, and every league that is reset or
  restarted with `--fresh` are kept under `stream_data/archive/`. The league
  checkpoint has a rolling backup and falls back to it if it can't be read.
  Pausing waits for verifications already running.

Every setting is also a command-line flag (`--stage-episodes`, `--pilots`,
`--intermission-s`, ...; see `--help`), and flags override the saved settings.

Run from the repo root (not inside `flypilot/`) so the `flypilot.*` module
imports resolve. Promote a run's `best.pkl` by copying it to
`models/flypilot_brain.pkl` and rerunning `evaluate_scenarios` with its
default paths.
