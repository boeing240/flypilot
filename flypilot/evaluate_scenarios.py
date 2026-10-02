"""Run the (frozen, already-trained) fly and the baseline controller through
every named scenario, without retraining -- this measures generalization /
robustness to conditions never seen in training, not in-scenario learning.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import pickle
import statistics as stats

from .baseline import BaselineController
from .env import DragStripEnv
from .scenarios import SCENARIOS
from .sense import N_PN, SenseEncoder
from .train import run_baseline_episode, run_fly_episode
from .learning import DopamineTracker, ValueDopamineTracker


def summarize(results):
    finished = [r for r in results if r["phase"] == "finished"]
    crashes = sum(1 for r in results if r["phase"] == "crash")
    timeouts = sum(1 for r in results if r["phase"] == "racing")
    n = len(results)
    return {
        "n": n,
        "finished": len(finished),
        "crashes": crashes,
        "timeouts": timeouts,
        "avg_time": stats.mean(r["elapsed_time"] for r in finished) if finished else None,
        "best_time": min((r["elapsed_time"] for r in finished), default=None),
    }


def _eval_fly(env, brain, n_episodes, seed0):
    encoder = SenseEncoder()
    dlon, dlat = ValueDopamineTracker(N_PN), DopamineTracker()
    return summarize([
        run_fly_episode(env, brain, encoder, dlon, dlat, seed=seed0 + i, learn=False)
        for i in range(n_episodes)
    ])


def eval_fly(kwargs, brain, n_episodes, seed0=20000):
    """Fly-only half of eval_scenario -- the numbers a promotion is judged on
    (same seeds, same procedure), without paying for the baseline."""
    return _eval_fly(DragStripEnv(**kwargs), brain, n_episodes, seed0)


def eval_scenario(name, kwargs, brain, baseline, n_episodes, seed0):
    env = DragStripEnv(**kwargs)
    fly = _eval_fly(env, brain, n_episodes, seed0)
    base_results = [
        run_baseline_episode(env, baseline, seed=seed0 + i) for i in range(n_episodes)
    ]
    return {"fly": fly, "baseline": summarize(base_results)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default="models/flypilot_brain.pkl")
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--out", default="results/scenario_results.json")
    args = ap.parse_args()

    with open(args.brain, "rb") as f:
        brain = pickle.load(f)
    baseline = BaselineController()

    all_stats = {}
    for name, kwargs in SCENARIOS.items():
        print(f"=== scenario: {name} ===")
        result = eval_scenario(name, kwargs, brain, baseline, args.episodes, seed0=20000)
        all_stats[name] = result
        print(
            f"  fly:      finished {result['fly']['finished']}/{result['fly']['n']}  "
            f"crash {result['fly']['crashes']}  "
            f"avg_time {result['fly']['avg_time']}"
        )
        print(
            f"  baseline: finished {result['baseline']['finished']}/{result['baseline']['n']}  "
            f"crash {result['baseline']['crashes']}  "
            f"avg_time {result['baseline']['avg_time']}"
        )

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
