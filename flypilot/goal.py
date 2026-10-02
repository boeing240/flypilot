"""Keeping the training goal alive inside the league, and never losing results.

The project's goal is a better champion: a brain with zero crashes and a lower
total avg_time over the five scenarios than the one in models/ (the same
criterion search.py uses -- see search.is_improvement). A league fly that races
close to the champion is therefore a *contender*: its exact brain (the state
that just raced -- continued training is a random walk, so peaks must be
captured when they happen) is saved to the hall of fame and verified in the
background on the full five-scenario suite, with the same seeds and procedure
that produced the champion's numbers. If it truly beats the champion it is
promoted: models/flypilot_brain.pkl, results/scenario_results.json and
results/search_status.json are updated, the previous champion is archived
first, and the promotion is appended to results/league_promotions.jsonl.

Nothing is ever deleted: retired, replaced and dethroned flies, previous
champions and reset leagues all go to stream_data/archive/.
"""
from __future__ import annotations

import json
import os
import pathlib
import pickle
import re
import shutil
import subprocess
import time

from .evaluate_scenarios import eval_fly
from .scenarios import SCENARIOS
from .search import MODEL_PATH, SCENARIO_ORDER, SCENARIO_RESULTS_PATH, STATUS_PATH, is_improvement, score_scenarios

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "stream_data"
ARCHIVE = DATA_DIR / "archive"
CANDIDATES = ARCHIVE / "candidates"
HOF_PATH = ARCHIVE / "hall_of_fame.json"
PROMOTIONS_LOG = ROOT / "results" / "league_promotions.jsonl"
TRACKED = ["models/flypilot_brain.pkl", "results/scenario_results.json", "results/search_status.json",
           "results/league_promotions.jsonl"]
KEEP_BEST_FILES = 40      # unpromoted contender brains kept on disk (metadata is kept forever)
MAX_PENDING = 2
VERIFY_EPISODES = 150


def stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", s)


def write_bytes(path: pathlib.Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def write_json(path: pathlib.Path, obj, indent=None) -> None:
    write_bytes(path, json.dumps(obj, indent=indent).encode("utf-8"))


def archive_pilot(pilot: dict, reason: str, gp: int) -> pathlib.Path:
    """Keep a fly's whole learned state (brain + value function) before it leaves the field."""
    name = f"{stamp()}_{safe(pilot['name'])}_{pilot['id']}_{reason}.pkl"
    blob = {
        "brain": pilot["brain"], "dl": pilot.get("dl"), "dt": pilot.get("dt"), "reason": reason, "gp": gp,
        "meta": {k: pilot.get(k) for k in ("id", "name", "nation", "number", "color", "legend", "seed", "joined_gp")},
        "stats": {k: pilot.get(k) for k in ("episodes", "rating", "pb", "wins", "podiums", "races", "finishes", "form")},
    }
    path = ARCHIVE / "pilots" / name
    write_bytes(path, pickle.dumps(blob))
    return path


def archive_file(src: pathlib.Path, kind: str, tag: str) -> pathlib.Path | None:
    if not src.exists():
        return None
    dst = ARCHIVE / kind / f"{stamp()}_{tag}{src.suffix}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def read_champion() -> dict | None:
    """The reigning champion as search.py sees it, plus its per-scenario times."""
    try:
        status = json.loads(STATUS_PATH.read_text())
        results = json.loads(SCENARIO_RESULTS_PATH.read_text())
        per = {n: results[n]["fly"]["avg_time"] for n in SCENARIO_ORDER}
    except (OSError, ValueError, KeyError):
        return None
    return {"status": status, "per": per, "total": status["champion_total_time"],
            "crashes": status["champion_crashes"], "run": status.get("champion_run", "")}


def _verify_task(blob: bytes, scenario: str, n_eps: int = VERIFY_EPISODES) -> dict:
    """One scenario of the full-suite check, in a worker process."""
    return eval_fly(SCENARIOS[scenario], pickle.loads(blob), n_eps)


class Goal:
    def __init__(self, get_cfg, log):
        self.cfg = get_cfg
        self.log = log
        self.pending: dict[str, dict] = {}     # contender id -> {"futs": {scenario: future}}
        CANDIDATES.mkdir(parents=True, exist_ok=True)
        try:
            self.hof: list[dict] = json.loads(HOF_PATH.read_text())
        except (OSError, ValueError):
            self.hof = []

    def _save(self):
        write_json(HOF_PATH, self.hof)

    # -- choosing and queueing contenders
    def consider(self, pool, gp: int, scenario: str, order: list[str], pilots: dict, raced: dict) -> list[str]:
        """After a race: queue the best contender, if there is one. Returns the pids queued."""
        c = self.cfg()
        champ = read_champion()
        if not c["verify"] or champ is None or len(self.pending) >= MAX_PENDING:
            return []
        ref = champ["per"][scenario]
        found = []
        for pid in order:
            p, r = pilots[pid], raced[pid]["result"]
            if p["legend"] or r["status"] != "finished" or r["time"] > ref + c["verify_margin"]:
                continue
            if p["episodes"] < c["verify_every"] or p["episodes"] - p.get("last_verify_eps", 0) < c["verify_every"]:
                continue
            if any(h["pid"] == pid and h["status"] == "pending" for h in self.hof):
                continue
            found.append((r["time"] - ref, pid))
        queued = []
        for delta, pid in sorted(found)[:1]:
            p = pilots[pid]
            blob = pickle.dumps(p["brain"])
            hid = f"gp{gp:04d}_{pid}"
            write_bytes(CANDIDATES / f"{hid}.pkl", blob)
            self.hof.append({"id": hid, "pid": pid, "name": p["name"], "nation": p["nation"], "gp": gp,
                             "episodes": p["episodes"], "scenario": scenario, "race_delta": round(delta, 3),
                             "status": "pending", "ts": time.time()})
            self._save()
            p["last_verify_eps"] = p["episodes"]
            self._submit(pool, hid, blob)
            self.log(f"contender: {p['name']} ({p['episodes']} eps, {delta:+.2f}s vs champion in this race) "
                     f"-> verifying on all {len(SCENARIO_ORDER)} scenarios")
            queued.append(pid)
        return queued

    def _submit(self, pool, hid: str, blob: bytes):
        self.pending[hid] = {"futs": {sc: pool.submit(_verify_task, blob, sc) for sc in SCENARIO_ORDER}}

    def resubmit(self, pool):
        """After a restart: contenders that were mid-verification get their checks re-queued."""
        for h in self.hof:
            f = CANDIDATES / f"{h['id']}.pkl"
            if h["status"] == "pending" and h["id"] not in self.pending:
                if f.exists():
                    self._submit(pool, h["id"], f.read_bytes())
                    self.log(f"re-queued verification of {h['name']} ({h['id']})")
                else:
                    h["status"] = "lost"
        self._save()

    # -- collecting results
    def poll(self) -> list[dict]:
        """Finished verifications (each already recorded, and promoted if it earned it)."""
        out = []
        for hid in list(self.pending):
            futs = self.pending[hid]["futs"]
            if not all(f.done() for f in futs.values()):
                continue
            del self.pending[hid]
            h = next(x for x in self.hof if x["id"] == hid)
            try:
                summaries = {sc: f.result() for sc, f in futs.items()}
            except Exception as e:  # a killed worker etc.; the brain file is kept, so it can be re-queued
                self.log(f"verification of {h['name']} failed: {e!r}")
                h["status"] = "pending"
                self._save()
                continue
            out.append(self._finish(h, summaries))
        if out:
            self._prune()
        return out

    def _finish(self, h: dict, summaries: dict) -> dict:
        results = {sc: {"fly": summaries[sc]} for sc in SCENARIO_ORDER}
        score = score_scenarios(results)                      # (crashes, total_time) or None if some scenario never finished
        champ = read_champion()
        h.update(status="done", ts_done=time.time(), per={sc: summaries[sc]["avg_time"] for sc in SCENARIO_ORDER},
                 crashes=score[0] if score else sum(s["crashes"] for s in summaries.values()),
                 total=score[1] if score else None, champion_before=champ["total"] if champ else None)
        h["delta"] = round(h["total"] - champ["total"], 3) if score and champ else None
        h["promoted"] = False
        if self.cfg()["auto_promote"] and champ is not None and is_improvement(champ["status"], score):
            self._promote(h, results, champ)
            h["promoted"] = True
        self._save()
        verdict = "PROMOTED" if h["promoted"] else "rejected"
        tot = f"{h['total']:.2f}s" if h["total"] is not None else "n/a"
        self.log(f"verified {h['name']}: total {tot}, crashes {h['crashes']} "
                 f"(champion {champ['total']:.2f}s) -> {verdict}" if champ else f"verified {h['name']}: total {tot}")
        return dict(h)

    # -- promotion: the goal being met
    def _promote(self, h: dict, results: dict, champ: dict):
        archive_file(MODEL_PATH, "champions", f"before_{h['id']}")
        archive_file(SCENARIO_RESULTS_PATH, "champions", f"before_{h['id']}_results")
        write_bytes(MODEL_PATH, (CANDIDATES / f"{h['id']}.pkl").read_bytes())
        try:
            old = json.loads(SCENARIO_RESULTS_PATH.read_text())
        except (OSError, ValueError):
            old = {}
        merged = {sc: {"fly": results[sc]["fly"], "baseline": old.get(sc, {}).get("baseline")} for sc in SCENARIO_ORDER}
        SCENARIO_RESULTS_PATH.write_text(json.dumps(merged, indent=2))
        status = dict(champ["status"])
        status.update(champion_crashes=h["crashes"], champion_total_time=h["total"],
                      champion_run=f"league: {h['name']} (GP {h['gp']}, {h['episodes']} episodes)")
        STATUS_PATH.write_text(json.dumps(status, indent=2))
        PROMOTIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(PROMOTIONS_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), "id": h["id"], "name": h["name"], "gp": h["gp"], "episodes": h["episodes"],
                                "total_time": h["total"], "crashes": h["crashes"], "champion_total_time_before": champ["total"],
                                "per_scenario": h["per"]}) + "\n")
        self.log(f"NEW CHAMPION {h['name']}: {h['total']:.2f}s (was {champ['total']:.2f}s); previous champion archived")
        self._git(h, champ)

    def _git(self, h: dict, champ: dict):
        c = self.cfg()
        if not c["git_commit"]:
            return
        def run(args):
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
        run(["add", *TRACKED])
        msg = (f"League promotion: {h['name']} (GP {h['gp']}, {h['episodes']} episodes) - "
               f"total_time={h['total']:.2f}s, 0 crashes (was {champ['total']:.2f}s)")
        res = run(["commit", "-m", msg, "--", *TRACKED])
        self.log("committed the promotion" if res.returncode == 0 else f"git commit failed: {res.stderr.strip()[:200]}")
        if res.returncode == 0 and c["git_push"]:
            res = run(["push"])
            self.log("pushed" if res.returncode == 0 else f"git push failed: {res.stderr.strip()[:200]}")

    def _prune(self):
        """Keep every promoted brain and the best KEEP_BEST_FILES others; the metadata stays regardless."""
        done = [h for h in self.hof if h["status"] == "done" and not h.get("promoted")]
        keep = {h["id"] for h in sorted((h for h in done if h["total"] is not None), key=lambda h: h["total"])[:KEEP_BEST_FILES]}
        for h in done:
            if h["id"] not in keep:
                (CANDIDATES / f"{h['id']}.pkl").unlink(missing_ok=True)
                h["file_pruned"] = True
        self._save()

    # -- reporting
    def info(self) -> dict:
        champ = read_champion()
        done = [h for h in self.hof if h["status"] == "done" and h["total"] is not None and h["crashes"] == 0]
        best = min(done, key=lambda h: h["total"], default=None)
        return {
            "champion_total": champ["total"] if champ else None,
            "champion_crashes": champ["crashes"] if champ else None,
            "champion_run": champ["run"] if champ else None,
            "verify": self.cfg()["verify"], "pending": len(self.pending),
            "verified": sum(1 for h in self.hof if h["status"] == "done"),
            "promotions": sum(1 for h in self.hof if h.get("promoted")),
            "best": {"name": best["name"], "total": best["total"], "delta": best["delta"]} if best else None,
            "recent": [{k: h.get(k) for k in ("id", "name", "gp", "episodes", "status", "total", "crashes", "delta", "promoted")}
                       for h in self.hof[-12:]][::-1],
            "archive": {"pilots": len(list((ARCHIVE / "pilots").glob("*.pkl"))) if (ARCHIVE / "pilots").exists() else 0,
                        "champions": len(list((ARCHIVE / "champions").glob("*.pkl"))) if (ARCHIVE / "champions").exists() else 0,
                        "contenders": len(list(CANDIDATES.glob("*.pkl")))},
        }
