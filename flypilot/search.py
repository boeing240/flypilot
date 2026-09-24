"""Automated search loop: repeatedly run population selection, verify each
round's winner on the full robustness scenario suite (not just the pooled
screen used during selection -- that screen catches crashes but not slow
survival, see FINDINGS.md), and promote it to models/flypilot_brain.pkl only
if it's a genuine improvement: zero crashes on every scenario and a lower
total avg_time summed across all five.

Every round appends one line to results/search_log.jsonl and updates
results/search_status.json, and both are committed (with model/scenario
files too, on a promotion) after every round -- so progress survives an
interrupted session instead of living only in runs/, which is gitignored.

Rounds alternate between fresh random-genome individuals (exploration) and
fine-tuned clones of the current champion (exploitation), since manual
search found fresh rounds rarely beat a good champion outright but
fine-tuning rounds are far more consistent (every clone tends to finish)
without being guaranteed to improve on it either.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "flypilot_brain.pkl"
SCENARIO_RESULTS_PATH = ROOT / "results" / "scenario_results.json"
SEARCH_LOG_PATH = ROOT / "results" / "search_log.jsonl"
STATUS_PATH = ROOT / "results" / "search_status.json"
SCENARIO_ORDER = ["nominal", "hot_engine", "cold_greasy_track", "patchy_grip", "worst_case"]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def score_scenarios(results: dict) -> tuple[int, float] | None:
    """(total crashes, total avg_time) summed over every named scenario, or
    None if the brain never finished at all in some scenario (disqualified --
    there's no time to compare)."""
    crashes = 0
    total_time = 0.0
    for name in SCENARIO_ORDER:
        s = results[name]["fly"]
        crashes += s["crashes"]
        if s["avg_time"] is None:
            return None
        total_time += s["avg_time"]
    return crashes, total_time


def load_status() -> dict:
    if STATUS_PATH.exists():
        return json.loads(STATUS_PATH.read_text())
    # bootstrap from whatever's currently committed as the champion
    run(["python", "-m", "flypilot.evaluate_scenarios",
         "--brain", str(MODEL_PATH), "--out", str(SCENARIO_RESULTS_PATH)])
    results = json.loads(SCENARIO_RESULTS_PATH.read_text())
    crashes, total_time = score_scenarios(results)
    status = {
        "rounds_run": 0,
        "champion_crashes": crashes,
        "champion_total_time": total_time,
        "champion_run": "models/flypilot_brain.pkl (pre-existing)",
    }
    STATUS_PATH.write_text(json.dumps(status, indent=2))
    return status


def is_improvement(status: dict, score: tuple[int, float] | None) -> bool:
    if score is None:
        return False
    crashes, total_time = score
    if crashes > 0:
        return False
    if status["champion_crashes"] > 0:
        return True  # any crash-free challenger beats a crashing champion
    return total_time < status["champion_total_time"]


def git(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, check=check)


def commit_round(record: dict, push: bool) -> None:
    paths = ["results/search_log.jsonl", "results/search_status.json"]
    if record["promoted"]:
        paths += ["models/flypilot_brain.pkl", "results/scenario_results.json"]
    git(["add", *paths])
    if record["promoted"]:
        msg = (f"Search round {record['round']}: PROMOTED {record['run_name']} "
               f"(crashes=0, total_time={record['total_time']:.2f}s, "
               f"was {record['champion_total_time_before']})")
    else:
        msg = (f"Search round {record['round']}: rejected {record['run_name']} "
               f"(crashes={record['crashes']}, total_time={record['total_time']})")
    result = git(["commit", "-m", msg], check=False)
    if result.returncode == 0 and push:
        git(["push"], check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--seed-start", type=int, default=None,
                     help="default: continues from results/search_status.json's round count")
    ap.add_argument("--population", type=int, default=16)
    ap.add_argument("--train-episodes", type=int, default=4000)
    ap.add_argument("--eval-episodes", type=int, default=150)
    ap.add_argument("--robust-episodes", type=int, default=50)
    ap.add_argument("--finetune-scale", type=float, default=0.3)
    ap.add_argument("--mode", choices=["fresh", "finetune", "alternate"], default="alternate")
    ap.add_argument("--push", action="store_true", default=True)
    ap.add_argument("--no-push", dest="push", action="store_false")
    args = ap.parse_args()

    status = load_status()
    seed_start = args.seed_start if args.seed_start is not None else 1000 + status["rounds_run"]

    for i in range(args.rounds):
        round_seed = seed_start + i
        use_finetune = args.mode == "finetune" or (args.mode == "alternate" and i % 2 == 1)
        run_name = f"search-{status['rounds_run'] + 1:04d}-{'ft' if use_finetune else 'fresh'}-{round_seed}"
        run_dir = ROOT / "runs" / run_name

        cmd = [
            "python", "-m", "flypilot.evolve",
            "--population", str(args.population),
            "--train-episodes", str(args.train_episodes),
            "--eval-episodes", str(args.eval_episodes),
            "--seed", str(round_seed),
            "--lat-kc-fraction", "0.25",
            "--robust-scenarios", "cold_greasy_track", "worst_case",
            "--robust-episodes", str(args.robust_episodes),
            "--run-name", run_name,
        ]
        if use_finetune:
            cmd += [
                "--continue-from", str(MODEL_PATH),
                "--finetune-scale", str(args.finetune_scale),
                "--checkpoint-every", "300",
                "--checkpoint-episodes", "30",
            ]
        run(cmd)

        run(["python", "-m", "flypilot.evaluate_scenarios",
             "--brain", str(run_dir / "best.pkl"),
             "--out", str(run_dir / "scenario_results.json")])
        results = json.loads((run_dir / "scenario_results.json").read_text())
        score = score_scenarios(results)

        record = {
            "round": status["rounds_run"] + 1,
            "run_name": run_name,
            "seed": round_seed,
            "mode": "finetune" if use_finetune else "fresh",
            "crashes": score[0] if score else None,
            "total_time": score[1] if score else None,
            "champion_total_time_before": status["champion_total_time"],
            "promoted": False,
        }

        if is_improvement(status, score):
            shutil.copy(run_dir / "best.pkl", MODEL_PATH)
            SCENARIO_RESULTS_PATH.write_text(json.dumps(results, indent=2))
            status["champion_crashes"], status["champion_total_time"] = score
            status["champion_run"] = run_name
            record["promoted"] = True

        status["rounds_run"] += 1
        with open(SEARCH_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        STATUS_PATH.write_text(json.dumps(status, indent=2))
        commit_round(record, args.push)

        verdict = "PROMOTED" if record["promoted"] else "rejected"
        print(f"=== round {record['round']} ({run_name}): {verdict} "
              f"crashes={record['crashes']} total_time={record['total_time']} "
              f"champion={status['champion_total_time']:.2f}s ===", flush=True)


if __name__ == "__main__":
    main()
