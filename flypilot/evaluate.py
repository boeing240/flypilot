from __future__ import annotations

import argparse
import statistics as stats

from .baseline import BaselineController
from .brain import FlyBrain
from .env import DragStripEnv
from .learning import DopamineTracker
from .sense import N_PN, SenseEncoder
from .train import run_baseline_episode, run_fly_episode, train


def summarize(name: str, results: list[dict]):
    finishes = [r for r in results if r["phase"] == "finished"]
    fouls = sum(1 for r in results if r["phase"] == "foul")
    crashes = sum(1 for r in results if r["phase"] == "crash")
    n = len(results)
    print(f"\n{name}: {n} runs")
    print(f"  finished   {len(finishes)}/{n}")
    print(f"  fouls      {fouls}/{n}")
    print(f"  crashes    {crashes}/{n}")
    if finishes:
        times = [r["elapsed_time"] for r in finishes]
        reactions = [r["reaction_time"] for r in finishes if r["reaction_time"] is not None]
        print(f"  elapsed time   mean {stats.mean(times):.3f}s  best {min(times):.3f}s  std {stats.pstdev(times):.3f}")
        if reactions:
            print(f"  reaction time  mean {stats.mean(reactions):.3f}s  best {min(reactions):.3f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-episodes", type=int, default=3000)
    ap.add_argument("--eval-episodes", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("=== training fly ===")
    brain = train(n_episodes=args.train_episodes, seed0=args.seed)

    env = DragStripEnv()
    encoder = SenseEncoder()
    dopamine_lon = DopamineTracker()
    dopamine_lat = DopamineTracker()
    fly_results = [
        run_fly_episode(env, brain, encoder, dopamine_lon, dopamine_lat, seed=10_000 + i, learn=False)
        for i in range(args.eval_episodes)
    ]

    baseline = BaselineController()
    baseline_results = [
        run_baseline_episode(env, baseline, seed=10_000 + i)
        for i in range(args.eval_episodes)
    ]

    summarize("Fly brain (frozen weights, evaluation seeds)", fly_results)
    summarize("Rule-based baseline (same seeds)", baseline_results)


if __name__ == "__main__":
    main()
