"""Population selection: train many flies with varied seed/temperament, keep the best.

Not a full genetic algorithm (no crossover) -- closer to random restarts with
selection, but the point stands: instead of hand-tuning reward coefficients
to fight each newly-discovered exploit, breed variation (KC wiring, initial
weights, exploration noise, learning rate) and let evaluation pick the
survivor. Cheap here because one individual trains in ~1-2 minutes.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import random
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .env import DragStripEnv
from .sense import SenseEncoder
from .train import continue_train, evaluate, fitness_key, train


def genome_str(genome):
    extra = "".join(f" {k}={v}" for k, v in genome.items() if k != "seed")
    return f"seed={genome['seed']}{extra}"


def _train_and_evaluate(genome, train_episodes, eval_episodes):
    # Runs in a worker process: builds its own env/encoder, so nothing is
    # shared with the parent or with sibling workers.
    brain = train(
        n_episodes=train_episodes,
        seed0=genome["seed"],
        brain_kwargs={"noise_sigma": genome["noise_sigma"], "eta": genome["eta"]},
        quiet=True,
    )
    env = DragStripEnv()
    encoder = SenseEncoder()
    summary = evaluate(brain, env, encoder, eval_episodes, seed0=90_000)
    summary["genome"] = genome
    return summary, brain


def _continue_and_evaluate(parent_brain, genome, train_episodes, eval_episodes,
                            finetune_scale, checkpoint_every, checkpoint_episodes):
    # Clone the parent's learned weights, but reseed the clone's own RNG --
    # otherwise every clone would draw the exact same exploration noise as
    # its siblings (same weights + same rng state = identical trajectory)
    # and training them separately would be pointless.
    brain = copy.deepcopy(parent_brain)
    brain.rng = np.random.default_rng(genome["seed"])
    # The plasticity rule never converges -- it keeps making eta-sized noisy
    # updates for as long as training runs. Continuing at full strength on an
    # already-good brain is a random walk that's as likely to wander away
    # from the optimum as toward it, so scale eta/noise_sigma down for a
    # gentler fine-tune instead of a second full training run.
    brain.eta *= finetune_scale
    brain.noise_sigma *= finetune_scale
    brain = continue_train(
        brain, train_episodes, seed0=genome["seed"], quiet=True,
        checkpoint_every=checkpoint_every, checkpoint_episodes=checkpoint_episodes,
    )
    env = DragStripEnv()
    encoder = SenseEncoder()
    summary = evaluate(brain, env, encoder, eval_episodes, seed0=90_000)
    summary["genome"] = genome
    return summary, brain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", type=int, default=8)
    ap.add_argument("--train-episodes", type=int, default=3000)
    ap.add_argument("--eval-episodes", type=int, default=150)
    ap.add_argument("--out", default="flypilot_brain.pkl")
    ap.add_argument("--leaderboard-out", default="evolve_leaderboard.json")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=os.cpu_count(),
                     help="parallel worker processes (default: all cores)")
    ap.add_argument("--continue-from", default=None,
                     help="pickle of a previously-trained brain; population becomes "
                          "clones of it, each continuing training independently, "
                          "instead of fresh random individuals")
    ap.add_argument("--finetune-scale", type=float, default=0.3,
                     help="multiply eta/noise_sigma by this when continuing training "
                          "from a parent brain, so it's a gentle fine-tune rather than "
                          "a second full-strength training run (only used with "
                          "--continue-from)")
    ap.add_argument("--checkpoint-every", type=int, default=300,
                     help="when continuing training, periodically freeze and evaluate "
                          "the brain and keep the best checkpoint seen instead of just "
                          "the final episode's weights (only used with --continue-from; "
                          "0 disables checkpointing)")
    ap.add_argument("--checkpoint-episodes", type=int, default=30,
                     help="episodes used for each checkpoint evaluation")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    parent_brain = None
    if args.continue_from:
        with open(args.continue_from, "rb") as f:
            parent_brain = pickle.load(f)
        genomes = [{"seed": rng.randint(0, 1_000_000)} for _ in range(args.population)]
    else:
        genomes = [
            {
                "seed": rng.randint(0, 1_000_000),
                "noise_sigma": round(rng.uniform(0.3, 0.9), 3),
                "eta": round(rng.uniform(0.01, 0.04), 4),
            }
            for _ in range(args.population)
        ]

    leaderboard = []
    workers = max(1, min(args.workers, len(genomes)))
    print(f"training {len(genomes)} individuals across {workers} worker process(es)"
          f"{' (continuing from ' + args.continue_from + ')' if parent_brain else ''}...")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        if parent_brain is not None:
            checkpoint_every = args.checkpoint_every or None
            futures = {
                pool.submit(_continue_and_evaluate, parent_brain, genome,
                            args.train_episodes, args.eval_episodes,
                            args.finetune_scale, checkpoint_every, args.checkpoint_episodes): i
                for i, genome in enumerate(genomes)
            }
        else:
            futures = {
                pool.submit(_train_and_evaluate, genome, args.train_episodes, args.eval_episodes): i
                for i, genome in enumerate(genomes)
            }
        for future in futures:
            i = futures[future]
            genome = genomes[i]
            summary, brain = future.result()
            leaderboard.append((summary, brain))
            finish_rate = round(100 * summary["finished"] / summary["n"])
            t = f"{summary['avg_time']:.2f}s" if summary["avg_time"] else "-"
            print(f"individual {i+1}/{len(genomes)}  {genome_str(genome)}  "
                  f"-> finished {finish_rate}%  crashes {summary['crashes']}/{summary['n']}  avg_time {t}")

    leaderboard.sort(key=lambda pair: fitness_key(pair[0]))

    print("\n=== leaderboard (best first) ===")
    table = []
    for rank, (summary, _brain) in enumerate(leaderboard):
        finish_rate = round(100 * summary["finished"] / summary["n"])
        t = f"{summary['avg_time']:.2f}s" if summary["avg_time"] else "-"
        print(f"{rank+1}. {genome_str(summary['genome'])}  "
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
