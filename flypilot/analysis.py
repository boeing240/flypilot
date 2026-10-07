"""A permanent, append-only record of every fly the league ever ran -- for analysis afterwards.

Files in stream_data/analysis/ (JSON lines; never rewritten, never wiped by a league reset):
  leagues.jsonl  one line when a league is created and whenever its settings change (the full training setup)
  pilots.jsonl   one line when a fly is created (origin, parent, generation, genome, seed) and one when it leaves
  rounds.jsonl   one line per fly per Grand Prix: placing, time, rating before/after, what it trained on, eta/noise
  events.jsonl   every league event (retirements, rookies, reverts, record laps, champions, verifications)

    python -m flypilot.analysis                 # summary: how well do offspring do against random flies?
    python -m flypilot.analysis --csv out_dir   # flat CSV tables for pandas / a spreadsheet
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import threading
import time

from .goal import DATA_DIR

DIR = DATA_DIR / "analysis"
_lock = threading.Lock()


def append(name: str, rec: dict) -> None:
    """Never raises: losing a log line must not stop the league."""
    try:
        with _lock:
            DIR.mkdir(parents=True, exist_ok=True)
            with open(DIR / f"{name}.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": round(time.time(), 2), **rec}, separators=(",", ":")) + "\n")
    except Exception:
        pass


def load(name: str) -> list[dict]:
    path = DIR / f"{name}.jsonl"
    out = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                pass                                   # a half-written last line
    return out


def _flat(rec: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in rec.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{prefix}{k}_"))
        elif isinstance(v, (list, tuple)):
            out[f"{prefix}{k}"] = json.dumps(v)
        else:
            out[f"{prefix}{k}"] = v
    return out


def export_csv(out_dir: pathlib.Path) -> list[pathlib.Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name in ("leagues", "pilots", "rounds", "events"):
        rows = [_flat(r) for r in load(name)]
        if not rows:
            continue
        cols = list(dict.fromkeys(k for r in rows for k in r))
        path = out_dir / f"{name}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        written.append(path)
    return written


def summary() -> str:
    pilots = [p for p in load("pilots") if p.get("event") == "join" and not p.get("legend")]
    rounds = load("rounds")
    if not pilots:
        return "no data yet -- run the league first"
    by_pilot: dict[tuple, list[dict]] = {}
    for r in rounds:
        if not r.get("legend"):
            by_pilot.setdefault((r["league"], r["pid"]), []).append(r)
    lines = [f"{len(set(p['league'] for p in pilots))} league(s), {len(pilots)} flies, {len(rounds)} race entries", ""]
    lines.append(f"{'origin':<10} {'flies':>5} {'mean peak':>10} {'first-10 Elo':>13} {'finish %':>9} {'best time':>10}")
    for origin in ("initial", "immigrant", "offspring"):
        group = [p for p in pilots if p.get("origin") == origin]
        peaks, early, fin, best = [], [], [], []
        for p in group:
            rs = sorted(by_pilot.get((p["league"], p["pid"]), []), key=lambda r: r["gp"])
            if not rs:
                continue
            peaks.append(max(r["rating_after"] for r in rs))
            early.append(sum(r["rating_after"] for r in rs[:10]) / len(rs[:10]))
            fin.append(100 * sum(r["status"] == "finished" for r in rs) / len(rs))
            times = [r["time"] for r in rs if r["status"] == "finished" and r.get("time")]
            if times:
                best.append(min(times))
        if group:
            avg = lambda xs: f"{sum(xs) / len(xs):.1f}" if xs else "-"
            lines.append(f"{origin:<10} {len(group):>5} {avg(peaks):>10} {avg(early):>13} {avg(fin):>9} "
                         f"{(f'{min(best):.2f}') if best else '-':>10}")
    gens = sorted({p.get("gen", 0) for p in pilots})
    if len(gens) > 1:
        lines += ["", f"{'generation':<10} {'flies':>5} {'mean first-10 Elo':>18}"]
        for g in gens:
            vals = []
            for p in (q for q in pilots if q.get("gen", 0) == g):
                rs = sorted(by_pilot.get((p["league"], p["pid"]), []), key=lambda r: r["gp"])[:10]
                if rs:
                    vals.append(sum(r["rating_after"] for r in rs) / len(rs))
            if vals:
                lines.append(f"{g:<10} {len(vals):>5} {sum(vals) / len(vals):>18.1f}")
    lines += size_table(pilots, rounds)
    return "\n".join(lines)


def size_table(pilots, rounds) -> list[str]:
    """Do bigger mushroom bodies learn faster or go lower?  Per brain size: flies, how many reached loss < 12 and
    after how many training blocks, the best and the recent check loss, and the loss on unseen conditions."""
    size_of = {(p["league"], p["pid"]): (p.get("genome") or {}).get("n_kc", 600) for p in pilots}
    series: dict = {}
    for r in sorted(rounds, key=lambda r: (r["league"], r["gp"])):
        ev = (r.get("train") or {}).get("eval")
        if ev and not r.get("legend"):
            series.setdefault((r["league"], r["pid"]), []).append(ev)
    rows = {}
    for key, evs in series.items():
        n = size_of.get(key, 600)
        losses = [e["loss"] for e in evs]
        reach = next((i + 1 for i, v in enumerate(losses) if v < 12), None)
        gs = [e["gauntlet"] for e in evs[-10:] if e.get("gauntlet") is not None]
        rows.setdefault(n, []).append({"best": min(losses), "recent": sum(losses[-10:]) / len(losses[-10:]), "reach": reach,
                                       "gaunt": sum(gs) / len(gs) if gs else None, "blocks": len(losses)})
    if len(rows) < 1:
        return []
    out = ["", f"{'cells':>6} {'flies':>5} {'reached<12':>11} {'median blocks':>14} {'best loss':>10} {'recent loss':>12} {'unseen loss':>12}"]
    avg = lambda xs: f"{sum(xs) / len(xs):.2f}" if xs else "-"
    for n in sorted(rows):
        rs = rows[n]
        reached = sorted(r["reach"] for r in rs if r["reach"])
        med = reached[len(reached) // 2] if reached else "-"
        out.append(f"{n:>6} {len(rs):>5} {len(reached):>5}/{len(rs):<5} {str(med):>14} {avg([r['best'] for r in rs]):>10} "
                   f"{avg([r['recent'] for r in rs]):>12} {avg([r['gaunt'] for r in rs if r['gaunt'] is not None]):>12}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", metavar="DIR", help="write flat CSV tables into this folder")
    args = ap.parse_args()
    if args.csv:
        for p in export_csv(pathlib.Path(args.csv)):
            print("wrote", p)
    print(summary())


if __name__ == "__main__":
    main()
