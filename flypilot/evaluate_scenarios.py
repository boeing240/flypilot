"""Run the (frozen, already-trained) fly and the baseline controller through
every named scenario, without retraining -- this measures generalization /
robustness to conditions never seen in training, not in-scenario learning.
"""
from __future__ import annotations

import argparse
import json
import pickle
import statistics as stats

from .baseline import BaselineController
from .env import DragStripEnv
from .replay import record_baseline_episode, record_fly_episode
from .scenarios import SCENARIOS
from .sense import N_PN, SenseEncoder
from .train import run_baseline_episode, run_fly_episode
from .learning import DopamineTracker, ValueDopamineTracker


def eval_scenario(name, kwargs, brain, baseline, n_episodes, seed0):
    env = DragStripEnv(**kwargs)
    encoder = SenseEncoder()
    dlon, dlat = ValueDopamineTracker(N_PN), DopamineTracker()

    fly_results = [
        run_fly_episode(env, brain, encoder, dlon, dlat, seed=seed0 + i, learn=False)
        for i in range(n_episodes)
    ]
    base_results = [
        run_baseline_episode(env, baseline, seed=seed0 + i) for i in range(n_episodes)
    ]

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

    return {"fly": summarize(fly_results), "baseline": summarize(base_results)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default="flypilot_brain.pkl")
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--out", default="scenario_results.json")
    ap.add_argument("--replay-out", default="replays.json")
    args = ap.parse_args()

    with open(args.brain, "rb") as f:
        brain = pickle.load(f)
    baseline = BaselineController()

    all_stats = {}
    replays = {}
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

        # one representative replay per driver, fixed seed so scenarios are comparable
        env_fly = DragStripEnv(**kwargs)
        encoder = SenseEncoder()
        fly_replay = record_fly_episode(env_fly, brain, encoder, seed=99001, env_kwargs=None)
        env_base = DragStripEnv(**kwargs)
        base_replay = record_baseline_episode(env_base, baseline, seed=99001, env_kwargs=None)
        replays[name] = {"fly": fly_replay, "baseline": base_replay}

    with open(args.out, "w") as f:
        json.dump(all_stats, f, indent=2)
    with open(args.replay_out, "w") as f:
        json.dump(replays, f)
    print(f"\nwrote {args.out} and {args.replay_out}")


if __name__ == "__main__":
    main()
