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
from .scenarios import SCENARIOS
from .sense import SenseEncoder
from .train import continue_train, evaluate, fitness_key, train


def genome_str(genome):
    extra = "".join(f" {k}={v}" for k, v in genome.items() if k != "seed")
    return f"seed={genome['seed']}{extra}"


def summary_str(summary):
    finish_rate = round(100 * summary["finished"] / summary["n"])
    t = f"{summary['avg_time']:.2f}s" if summary["avg_time"] else "-"
    out = f"finished {finish_rate}%  crashes {summary['crashes']}/{summary['n']}  avg_time {t}"
    rob = summary.get("robust")
    if rob:
        out += f"  | robust finished {rob['finished']}/{rob['n']} crashes {rob['crashes']}"
    return out


def evaluate_robustness(brain, scenario_names, n_episodes, seed0=95_000):
    """Pooled finish/crash counts across held-out scenarios (never trained on).
    seed0 differs from both the nominal selection eval (90_000) and
    evaluate_scenarios.py (20_000), so a winner isn't screened on the same
    episodes it will later be verified on."""
    encoder = SenseEncoder()
    total = {"n": 0, "finished": 0, "crashes": 0}
    for name in scenario_names:
        s = evaluate(brain, DragStripEnv(**SCENARIOS[name]), encoder, n_episodes, seed0=seed0)
        for k in total:
            total[k] += s[k]
    return total


def selection_key(summary):
    # With robustness scores present, pool them with the nominal eval so an
    # individual that only works at nominal can't outrank one that also
    # survives the held-out scenarios; nominal avg_time is still the tiebreak.
    rob = summary.get("robust")
    if rob is None:
        return fitness_key(summary)
    n = summary["n"] + rob["n"]
    finish_rate = (summary["finished"] + rob["finished"]) / n
    crash_rate = (summary["crashes"] + rob["crashes"]) / n
    avg_time = summary["avg_time"] if summary["avg_time"] is not None else 1e9
    return (-finish_rate, crash_rate, avg_time)


def _train_and_evaluate(genome, train_episodes, eval_episodes, lat_kc_fraction,
                         robust_scenarios=(), robust_episodes=50):
    # Runs in a worker process: builds its own env/encoder, so nothing is
    # shared with the parent or with sibling workers.
    brain = train(
        n_episodes=train_episodes,
        seed0=genome["seed"],
        brain_kwargs={
            "noise_sigma": genome["noise_sigma"],
            "eta": genome["eta"],
            "lat_kc_fraction": lat_kc_fraction,
        },
        quiet=True,
    )
    env = DragStripEnv()
    encoder = SenseEncoder()
    summary = evaluate(brain, env, encoder, eval_episodes, seed0=90_000)
    if robust_scenarios:
        summary["robust"] = evaluate_robustness(brain, robust_scenarios, robust_episodes)
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
    ap.add_argument("--eta-min", type=float, default=0.01,
                     help="lower bound of the sampled learning-rate range (fresh individuals only)")
    ap.add_argument("--eta-max", type=float, default=0.04,
                     help="upper bound of the sampled learning-rate range (fresh individuals only)")
    ap.add_argument("--noise-min", type=float, default=0.3,
                     help="lower bound of the sampled exploration-noise range (fresh individuals only)")
    ap.add_argument("--noise-max", type=float, default=0.9,
                     help="upper bound of the sampled exploration-noise range (fresh individuals only)")
    ap.add_argument("--robust-scenarios", nargs="*", default=[], choices=sorted(SCENARIOS),
                     help="held-out scenarios to also score each fresh individual on; "
                          "their finish/crash counts are pooled into selection so "
                          "winners must survive them, not just nominal")
    ap.add_argument("--robust-episodes", type=int, default=50,
                     help="episodes per robustness scenario")
    ap.add_argument("--lat-kc-fraction", type=float, default=0.25,
                     help="fraction of the KC population wired to the lateral "
                          "(steering) compartment (only used for fresh individuals, "
                          "not --continue-from, since KC wiring is fixed at brain init)")
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
                "noise_sigma": round(rng.uniform(args.noise_min, args.noise_max), 3),
                "eta": round(rng.uniform(args.eta_min, args.eta_max), 4),
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
                pool.submit(_train_and_evaluate, genome, args.train_episodes,
                            args.eval_episodes, args.lat_kc_fraction,
                            tuple(args.robust_scenarios), args.robust_episodes): i
                for i, genome in enumerate(genomes)
            }
        for future in futures:
            i = futures[future]
            genome = genomes[i]
            summary, brain = future.result()
            leaderboard.append((summary, brain))
            print(f"individual {i+1}/{len(genomes)}  {genome_str(genome)}  -> {summary_str(summary)}")

    leaderboard.sort(key=lambda pair: selection_key(pair[0]))

    print("\n=== leaderboard (best first) ===")
    table = []
    for rank, (summary, _brain) in enumerate(leaderboard):
        print(f"{rank+1}. {genome_str(summary['genome'])}  {summary_str(summary)}")
        table.append({k: v for k, v in summary.items()})

    best_summary, best_brain = leaderboard[0]
    with open(args.out, "wb") as f:
        pickle.dump(best_brain, f)
    with open(args.leaderboard_out, "w") as f:
        json.dump(table, f, indent=2)
    print(f"\nwrote {args.out} (best individual) and {args.leaderboard_out}")


if __name__ == "__main__":
    main()
