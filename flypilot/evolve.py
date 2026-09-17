"""Population selection: train many flies with varied seed/temperament, keep the best.

Not a full genetic algorithm (no crossover) -- closer to random restarts with
selection, but the point stands: instead of hand-tuning reward coefficients
to fight each newly-discovered exploit, breed variation (KC wiring, initial
weights, exploration noise, learning rate) and let evaluation pick the
survivor. Cheap here because one individual trains in ~1-2 minutes.
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import statistics as stats

from .env import DragStripEnv
from .learning import DopamineTracker
from .sense import SenseEncoder
from .train import run_fly_episode, train


def fitness_key(summary):
    # lexicographic: finish rate first, then crash rate, then speed -- a fast
    # individual that crashes is worse than a slow one that always finishes.
    finish_rate = summary["finished"] / summary["n"]
    crash_rate = summary["crashes"] / summary["n"]
    avg_time = summary["avg_time"] if summary["avg_time"] is not None else 1e9
    return (-finish_rate, crash_rate, avg_time)


def evaluate(brain, env, encoder, n_episodes, seed0):
    dlon, dlat = DopamineTracker(), DopamineTracker()
    results = [
        run_fly_episode(env, brain, encoder, dlon, dlat, seed=seed0 + i, learn=False)
        for i in range(n_episodes)
    ]
    finished = [r for r in results if r["phase"] == "finished"]
    crashes = sum(1 for r in results if r["phase"] == "crash")
    return {
        "n": n_episodes,
        "finished": len(finished),
        "crashes": crashes,
        "avg_time": stats.mean(r["elapsed_time"] for r in finished) if finished else None,
        "best_time": min((r["elapsed_time"] for r in finished), default=None),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", type=int, default=8)
    ap.add_argument("--train-episodes", type=int, default=3000)
    ap.add_argument("--eval-episodes", type=int, default=150)
    ap.add_argument("--out", default="flypilot_brain.pkl")
    ap.add_argument("--leaderboard-out", default="evolve_leaderboard.json")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    genomes = []
    for i in range(args.population):
        genomes.append({
            "seed": rng.randint(0, 1_000_000),
            "noise_sigma": round(rng.uniform(0.3, 0.9), 3),
            "eta": round(rng.uniform(0.01, 0.04), 4),
        })

    env = DragStripEnv()
    encoder = SenseEncoder()

    leaderboard = []
    for i, genome in enumerate(genomes):
        print(f"=== individual {i+1}/{len(genomes)}  seed={genome['seed']} "
              f"noise_sigma={genome['noise_sigma']} eta={genome['eta']} ===")
        brain = train(
            n_episodes=args.train_episodes,
            seed0=genome["seed"],
            brain_kwargs={"noise_sigma": genome["noise_sigma"], "eta": genome["eta"]},
            quiet=True,
        )
        summary = evaluate(brain, env, encoder, args.eval_episodes, seed0=90_000)
        summary["genome"] = genome
        leaderboard.append((summary, brain))
        finish_rate = round(100 * summary["finished"] / summary["n"])
        t = f"{summary['avg_time']:.2f}s" if summary["avg_time"] else "-"
        print(f"    -> finished {finish_rate}%  crashes {summary['crashes']}/{summary['n']}  avg_time {t}")

    leaderboard.sort(key=lambda pair: fitness_key(pair[0]))

    print("\n=== leaderboard (best first) ===")
    table = []
    for rank, (summary, _brain) in enumerate(leaderboard):
        finish_rate = round(100 * summary["finished"] / summary["n"])
        t = f"{summary['avg_time']:.2f}s" if summary["avg_time"] else "-"
        print(f"{rank+1}. seed={summary['genome']['seed']} "
              f"noise_sigma={summary['genome']['noise_sigma']} eta={summary['genome']['eta']}  "
              f"finished {finish_rate}%  crashes {summary['crashes']}/{summary['n']}  avg_time {t}")
        table.append({k: v for k, v in summary.items()})

    best_summary, best_brain = leaderboard[0]
    with open(args.out, "wb") as f:
        pickle.dump(best_brain, f)
    with open(args.leaderboard_out, "w") as f:
        json.dump(table, f, indent=2)
    print(f"\nwrote {args.out} (best individual) and {args.leaderboard_out}")


if __name__ == "__main__":
    main()
