"""Live league: a field of flies keeps training while racing each other, and
everything a viewer needs (race replays, standings, rating history, events) is
written as JSON for the page in flypilot/web/ to play back -- i.e. a training
run you can stream, with an admin panel to configure and run it.

Each "Grand Prix" is one training block for every pilot (stage_episodes
episodes of the usual node-perturbation learning), followed by a frozen-policy
race on shared conditions and a shared seed, so results are comparable. Finishing
order updates an Elo rating; that rating is the pilot ranking. The reigning
champion from models/ joins as the "Legend" (frozen, never trained) so there is
a visible benchmark to chase. Every few Grands Prix the lowest-rated veteran
retires and a rookie fly with a fresh random genome takes the seat.

    python -m flypilot.league
    viewer  http://127.0.0.1:8765/        (OBS Browser Source, 1920x1080)
    admin   http://127.0.0.1:8765/admin   (start / pause / reset, all settings)

Rounds run back to back: the next round trains while the viewer is still
playing the current one (the page reports which round it is showing, and
training only runs `lead_rounds` ahead of it), so there are no dead gaps.

League state (brains included) is checkpointed after every Grand Prix, so a
restart resumes where it stopped; the admin panel (or --fresh) starts over.
The admin API only answers on loopback addresses.
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import os
import pathlib
import pickle
import random
import shutil
import threading
import time
import traceback
import urllib.parse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from . import analysis
from .brain import FlyBrain
from .env import DragStripEnv
from .evolution import choose_parent, genome, mutate_brain
from .goal import CANDIDATES, DATA_DIR, ROOT, Goal, archive_file, archive_pilot
from .learning import DopamineTracker, ValueDopamineTracker
from .pilots import make_identity
from .scenarios import SCENARIOS
from .sense import N_PN, SenseEncoder
from .train import CONDITION_RANGES, NOMINAL_CONDITIONS, run_fly_episode, set_conditions

RACES_DIR = DATA_DIR / "races"
STATE_PATH = DATA_DIR / "state.json"
LIVE_PATH = DATA_DIR / "live.json"
CHECKPOINT_PATH = DATA_DIR / "league.pkl"
CHECKPOINT_BAK = DATA_DIR / "league.pkl.bak"
SETTINGS_PATH = DATA_DIR / "settings.json"
WEB_DIR = pathlib.Path(__file__).resolve().parent / "web"
LEGEND_PATH = ROOT / "models" / "flypilot_brain.pkl"

START_RATING = 1500.0
KEEP_RACES = 80          # race files kept on disk
LISTED_RACES = 40        # races offered to the viewer for replay
HISTORY_POINTS = 120
VIEWER_TIMEOUT = 20.0    # seconds without a heartbeat before the viewer counts as gone
CPUS = os.cpu_count() or 2


def lower_priority() -> None:
    """Training must never starve the browser / stream encoder: run below normal priority."""
    try:
        if os.name == "nt":
            import ctypes
            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)   # BELOW_NORMAL
        else:
            os.nice(10)
    except Exception:
        pass


CONDITION_TITLES = {
    "nominal": ["Dry Strip Sprint", "Sunday Showdown", "Blue Hour Dash", "Eighth-Mile Open", "Prime Time Cup"],
    "hot_engine": ["Heatwave Cup", "Overheat Trophy"],
    "cold_greasy_track": ["Cold Rubber Classic", "Slick Strip Trophy"],
    "patchy_grip": ["Oil-Down Challenge", "Bleach-Box Brawl"],
    "worst_case": ["Mayhem Grand Final", "Survival Series"],
}

# ---------------------------------------------------------------- settings
# One table drives validation, defaults, the CLI flags and the admin form.
# apply: "live" = takes effect immediately, "round" = at the start of the next
# round, "reset" = only when a new league is created.
SETTINGS_SPEC = [
    dict(key="name", group="League", label="League title", type="text", default="FLY PILOT LEAGUE", apply="live",
         help="Shown in the stream header."),
    dict(key="pilots", group="League", label="Learner flies", type="int", min=2, max=15, default=11, apply="round",
         help="Trained pilots in the field. The Legend comes on top."),
    dict(key="legend", group="League", label="Include the Legend", type="bool", default=True, apply="round",
         help="The reigning champion from models/ races as a frozen benchmark."),
    dict(key="seed", group="League", label="Seed", type="int", min=0, max=10**9, default=0, apply="reset",
         help="0 = a fresh random seed for every new league (different pilots each time); a fixed number reproduces a league."),
    dict(key="k", group="League", label="Elo K-factor", type="float", min=4, max=128, step=1, default=32.0, apply="round",
         help="How far one race moves a rating."),

    dict(key="stage_episodes", group="Training", label="Episodes per round", type="int", min=5, max=2000, default=60,
         apply="round", help="Training episodes per pilot between races. Lower = a slower storyline."),
    dict(key="workers", group="Training", label="Worker processes", type="int", min=1, max=CPUS,
         default=max(1, CPUS // 2), apply="round", help=f"Of {CPUS} cores. Leave plenty free for the browser and the stream encoder."),
    dict(key="randomize_prob", group="Training", label="Domain randomization", type="float", min=0, max=1, step=0.05,
         default=0.25, apply="round", help="Share of training episodes on random traction/engine conditions "
                                                      "(the champion goal is scored over five scenarios, so train for them)."),

    dict(key="nominal_share", group="Season", label="Dry-strip share", type="float", min=0, max=1, step=0.05,
         default=0.55, apply="round", help="Fraction of races on perfect conditions; the rest rotate through the harder ones."),
    dict(key="retire_every", group="Season", label="Rounds per season", type="int", min=1, max=200, default=6,
         apply="round", help="At the end of a season the lowest-rated veteran may retire."),
    dict(key="retire_count", group="Season", label="Retirements per season", type="int", min=0, max=5, default=1,
         apply="round", help="0 disables retirements and rookies."),
    dict(key="season_carry", group="Season", label="Rating carried into next season", type="float", min=0, max=1, step=0.05,
         default=0.5, apply="round",
         help="At the end of a season every rating is pulled back toward 1500: 1 = keep everything, 0 = full reset, 0.5 = halfway."),
    dict(key="evolve", group="Evolution", label="Breed new flies from the best", type="bool", default=True, apply="round",
         help="A rookie is a mutated child of a top fly (or the Legend) instead of a random newcomer."),
    dict(key="mutation_rate", group="Evolution", label="Mutation strength", type="float", min=0, max=1, step=0.01, default=0.15,
         apply="round", help="Weight noise as a share of the weights' own spread. 0 = exact clones."),
    dict(key="immigrant_share", group="Evolution", label="Random newcomers", type="float", min=0, max=1, step=0.05, default=0.25,
         apply="round", help="Share of rookies that are fully random, to keep the gene pool varied."),
    dict(key="legend_parent_share", group="Evolution", label="Legend as parent", type="float", min=0, max=1, step=0.05, default=0.25,
         apply="round", help="Chance that a bred rookie is a child of the reigning champion."),
    dict(key="revert_gap", group="Evolution", label="Revert to best form (Elo)", type="int", min=0, max=500, default=60,
         apply="round", help="A fly that falls this far below its peak rating returns to its best brain. 0 = never."),
    dict(key="min_age", group="Season", label="Veteran after (rounds)", type="int", min=1, max=1000, default=12,
         apply="round", help="A pilot must have raced this many rounds before it can retire."),

    dict(key="lead_rounds", group="Pacing", label="Training lead (rounds)", type="int", min=0, max=6, default=2,
         apply="round", help="How many finished rounds may wait ahead of what the page is showing. 0 = unlimited."),
    dict(key="min_round_seconds", group="Pacing", label="Min seconds per round", type="float", min=0, max=600,
         default=30.0, apply="round", help="Floor on the round rate when no viewer is connected."),

    dict(key="verify", group="Goal", label="Verify contenders on the full suite", type="bool", default=True, apply="round",
         help="A fly racing close to the champion is checked in the background on all 5 scenarios, exactly as search.py does."),
    dict(key="verify_every", group="Goal", label="Re-verify a pilot every (episodes)", type="int", min=50, max=1000000,
         default=300, apply="round", help="Minimum training between two verifications of the same fly (and before its first)."),
    dict(key="verify_margin", group="Goal", label="Contender margin (s)", type="float", min=0, max=10, step=0.1,
         default=0.6, apply="round", help="Verified when its race time is within this of the champion's on the same conditions."),
    dict(key="auto_promote", group="Goal", label="Auto-promote a better champion", type="bool", default=True,
         apply="round", help="Updates models/ and results/ (the old champion is archived first). Off = verify and record only."),
    dict(key="git_commit", group="Goal", label="Commit promotions to git", type="bool", default=False, apply="round",
         help="Local commit of the champion files only."),
    dict(key="git_push", group="Goal", label="Push promotion commits", type="bool", default=False, apply="round",
         help="Needs the commit option; pushes to the current remote."),

    dict(key="intro_s", group="Broadcast", label="Round intro (s)", type="float", min=1, max=12, step=0.5, default=4.5,
         apply="live", help="Title card and the cars rolling onto the grid."),
    dict(key="staging_s", group="Broadcast", label="Staging lights (s)", type="float", min=0.5, max=8, step=0.5,
         default=2.0, apply="live", help="Amber lights before the green."),
    dict(key="podium_s", group="Broadcast", label="Podium (s)", type="float", min=3, max=30, step=0.5, default=8.0,
         apply="live", help="End-of-round podium and result table."),
    dict(key="intermission_s", group="Broadcast", label="Pause between rounds (s)", type="float", min=0, max=180,
         step=1, default=8.0, apply="live", help="Countdown shown after the podium, before the next round."),
    dict(key="fast_forward", group="Broadcast", label="Tail fast-forward (x)", type="int", min=1, max=8, default=4,
         apply="live", help="Speed-up once the winner is home and only stragglers remain."),
    dict(key="animations", group="Broadcast", label="Round start/end animations", type="bool", default=True,
         apply="live", help="Flag wipe, podium and confetti."),
    dict(key="idle_replays", group="Broadcast", label="Replay while waiting", type="bool", default=False,
         apply="live", help="If the next round isn't ready, replay an old one instead of showing the training screen."),
]
SPEC_BY_KEY = {s["key"]: s for s in SETTINGS_SPEC}
DEFAULTS = {s["key"]: s["default"] for s in SETTINGS_SPEC}
BROADCAST_KEYS = ("intro_s", "staging_s", "podium_s", "intermission_s", "fast_forward", "animations", "idle_replays")


def clean_settings(raw: dict, base: dict) -> dict:
    """Merge `raw` over `base`, coercing and clamping; unknown keys are ignored."""
    out = dict(base)
    for key, val in (raw or {}).items():
        spec = SPEC_BY_KEY.get(key)
        if spec is None or val is None:
            continue
        try:
            if spec["type"] == "bool":
                val = val if isinstance(val, bool) else str(val).lower() in ("1", "true", "yes", "on")
            elif spec["type"] == "text":
                val = str(val).strip()[:60] or spec["default"]
            else:
                val = float(val) if spec["type"] == "float" else int(float(val))
                val = max(spec["min"], min(spec["max"], val))
        except (TypeError, ValueError):
            continue
        out[key] = val
    return out


# ---------------------------------------------------------------- worker side

def record_race(brain: FlyBrain, scenario: str, seed: int) -> dict:
    """One frozen-policy run under `scenario`, recorded tick by tick (every 2nd
    tick kept) so the page can replay it."""
    env = DragStripEnv(**SCENARIOS[scenario])
    rows = []

    def on_tick(e, k):
        rows.append((k, e.x, e.y, e.v, e.rpm, e.gear, e.wheel_slip, e.phase))

    res = run_fly_episode(env, brain, SenseEncoder(), ValueDopamineTracker(N_PN), DopamineTracker(),
                          seed=seed, learn=False, on_tick=on_tick)
    k_green = next((r[0] for r in rows if r[7] == "green"), 0)
    kept = [r for r in rows if r[0] % 2 == 0]
    if rows[-1] is not kept[-1]:
        kept.append(rows[-1])
    last = rows[-1]
    status = {"finished": "finished", "crash": "crash", "foul": "crash"}.get(last[7], "timeout")
    return {
        "k_green": k_green,
        "k_end": last[0],
        "trace": {
            "k": [r[0] for r in kept],
            "x": [round(r[1], 2) for r in kept],
            "y": [round(r[2], 3) for r in kept],
            "v": [round(r[3], 2) for r in kept],
            "r": [int(r[4]) for r in kept],
            "g": [r[5] for r in kept],
            "s": [round(r[6], 2) for r in kept],
        },
        "result": {
            "status": status,
            "time": round(res["elapsed_time"], 3) if status == "finished" else None,
            "x": round(last[1], 2),
            "reaction": res["reaction_time"],
            "top_kmh": round(3.6 * max(r[3] for r in rows), 1),
        },
    }


def _stage_task(brain, dl, dt, episodes, seed, n_eps, randomize_prob, scenario, race_seed):
    """Train one pilot for a block of episodes, then race it. Runs in a worker
    process; trackers travel with the brain so value learning carries over."""
    rng = random.Random(seed * 1_000_003 + episodes)
    env, enc = DragStripEnv(), SenseEncoder()
    stats = {"finished": 0, "crash": 0, "timeout": 0, "time_sum": 0.0, "reward": 0.0}
    for i in range(n_eps):
        if randomize_prob > 0.0 and rng.random() < randomize_prob:
            set_conditions(env, {n: rng.uniform(lo, hi) for n, (lo, hi) in CONDITION_RANGES.items()})
        else:
            set_conditions(env, NOMINAL_CONDITIONS)
        r = run_fly_episode(env, brain, enc, dl, dt, seed=(seed * 7919 + episodes + i) % 2_147_483_647, learn=True)
        stats["reward"] += r["total_reward"]
        if r["phase"] == "finished":
            stats["finished"] += 1
            stats["time_sum"] += r["elapsed_time"]
        elif r["phase"] == "crash":
            stats["crash"] += 1
        else:
            stats["timeout"] += 1
    stats["n"] = n_eps
    return brain, dl, dt, episodes + n_eps, stats, record_race(brain, scenario, race_seed)


# ---------------------------------------------------------------- file helpers

_write_lock = threading.Lock()  # the run loop and the admin API both publish live.json


def atomic_write_json(path: pathlib.Path, obj) -> None:
    with _write_lock:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(obj, separators=(",", ":")), encoding="utf-8")
        for attempt in range(20):  # a reader may hold the file open for a moment (Windows)
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                time.sleep(0.05)
        os.replace(tmp, path)


def atomic_write_bytes(path: pathlib.Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


# ---------------------------------------------------------------- league

class Aborted(Exception):
    """Raised inside a round when the admin pauses/resets: the round is dropped, not finished."""


def kill_pool(pool) -> None:
    """Stop a process pool right now, including workers that are in the middle of a task."""
    procs = list((getattr(pool, "_processes", None) or {}).values())
    try:
        pool.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass
    for pr in procs:
        try:
            pr.kill()
        except Exception:
            pass


class League:
    def __init__(self, cfg: dict, legend_path: str = str(LEGEND_PATH)):
        self.cfg = cfg
        self.legend_path = legend_path
        self.lock = threading.RLock()
        self.logbuf: collections.deque = collections.deque(maxlen=300)
        self.abort_req = False
        self.round_gp = 0
        self.goal = Goal(lambda: self.cfg, self.log)
        self.status = "stopped"            # stopped | running | pausing
        self.phase = "stopped"
        self.thread: threading.Thread | None = None
        self.pause_req = False
        self.started_at: float | None = None
        self.viewer_gp = 0
        self.viewer_ts = 0.0
        self.eps_window: list[tuple[float, int]] = []
        self.reset_state()

    # -- logging
    def log(self, msg: str):
        line = f"{time.strftime('%H:%M:%S')}  {msg}"
        self.logbuf.append(line)
        try:
            print(line, flush=True)
        except OSError:
            pass

    # -- state
    def reset_state(self):
        self.league_seed = self.cfg["seed"] or random.SystemRandom().randrange(1, 10**9)
        self.rng = random.Random(self.league_seed)
        self.gp = 0
        self.season = 1
        self.season_gp = 0
        self.pilots: dict[str, dict] = {}
        self.history: dict[str, list] = {}
        self.records: dict = {}
        self.events: list[dict] = []
        self.races: list[dict] = []
        self.used_names: set[str] = self.load_known_names()   # names of every pilot of every past league
        self.next_pid = 1
        self.pending_events: list[dict] = []

    def init_field(self):
        self.log(f"new league, seed {self.league_seed}")
        analysis.append("leagues", {"event": "start", "league": self.league_seed, "gp": self.gp, "cfg": dict(self.cfg)})
        if self.cfg["legend"]:
            self.add_legend()
        while len(self.trained()) < self.cfg["pilots"]:
            self.add_rookie(breed=False)

    def save(self):
        blob = {k: getattr(self, k) for k in ("rng", "gp", "season", "season_gp", "pilots", "history", "records",
                                              "events", "races", "used_names", "next_pid", "league_seed")}
        data = pickle.dumps(blob)
        if CHECKPOINT_PATH.exists():
            try:                                   # keep the last checkpoint that still loads as the backup
                pickle.loads(CHECKPOINT_PATH.read_bytes())
                shutil.copy2(CHECKPOINT_PATH, CHECKPOINT_BAK)
            except Exception:
                pass
        atomic_write_bytes(CHECKPOINT_PATH, data)

    def load(self) -> bool:
        for path in (CHECKPOINT_PATH, CHECKPOINT_BAK):
            if not path.exists():
                continue
            try:
                blob = pickle.loads(path.read_bytes())
            except Exception as e:
                self.log(f"checkpoint {path.name} is unreadable ({e!r}); trying the backup")
                continue
            for k, v in blob.items():
                setattr(self, k, v)
            if path == CHECKPOINT_BAK:
                self.log("restored the league from the backup checkpoint")
            return True
        return False

    def archive_league(self, tag: str):
        """Reset / --fresh never delete trained brains: the whole checkpoint is kept."""
        dst = archive_file(CHECKPOINT_PATH, "leagues", tag)
        if dst:
            self.log(f"previous league kept in {dst.relative_to(ROOT)}")

    def wipe_files(self):
        for f in (CHECKPOINT_PATH, CHECKPOINT_BAK, STATE_PATH, LIVE_PATH):
            f.unlink(missing_ok=True)
        shutil.rmtree(RACES_DIR, ignore_errors=True)
        RACES_DIR.mkdir(parents=True, exist_ok=True)

    def save_settings(self):
        atomic_write_json(SETTINGS_PATH, self.cfg)

    # -- pilots
    def _taken(self):
        ps = self.pilots.values()
        return ({p["name"] for p in ps} | self.used_names, {p["number"] for p in ps}, {p["color"] for p in ps})

    def add_legend(self):
        path = pathlib.Path(self.legend_path)
        if not path.exists():
            self.log(f"(no legend brain at {path}; racing without one)")
            return
        names, numbers, colors = self._taken()
        ident = make_identity(0, names, numbers, colors, legend=True)
        pilot = self._new_pilot("legend", ident, pickle.loads(path.read_bytes()), seed=0)
        pilot["sig"] = self.legend_sig()
        pilot["origin"] = "legend"
        self.pilots["legend"] = pilot
        self.history["legend"] = [[self.gp, START_RATING]]
        self.log_join(pilot)

    def legend_sig(self):
        try:
            st = pathlib.Path(self.legend_path).stat()
            return [st.st_mtime_ns, st.st_size]
        except OSError:
            return None

    def sync_legend(self):
        """The Legend always mirrors the champion on disk (e.g. after search.py promotes one)."""
        lg = self.pilots.get("legend")
        sig = self.legend_sig()
        if lg is None or sig is None:
            return
        if lg.get("sig") is None:
            lg["sig"] = sig
        elif lg["sig"] != sig:
            try:
                lg["brain"] = pickle.loads(pathlib.Path(self.legend_path).read_bytes())
            except Exception as e:
                self.log(f"could not reload the champion: {e!r}")
                return
            lg["sig"] = sig
            self.pending_events.append({"kind": "legend_upgrade",
                                        "text": "The Legend has been upgraded: a new champion was deployed"})
            self.log("legend reloaded from the champion on disk")

    def coronate(self, out: dict):
        """A contender became the champion: it takes over the Legend seat and its own seat goes to a rookie."""
        lg = self.pilots.get("legend")
        if lg is None:
            return
        self.archive(lg, "dethroned", self.gp)
        lg["brain"] = pickle.loads((CANDIDATES / f"{out['id']}.pkl").read_bytes())
        lg["name"], lg["nation"] = out["name"], out["nation"]
        lg["sig"] = self.legend_sig()
        cp = self.pilots.pop(out["pid"], None)
        if cp is not None:
            self.archive(cp, "crowned", self.gp)
            self.history.pop(out["pid"], None)

    def add_rookie(self, rating: float | None = None, breed: bool = True) -> dict:
        """A new fly takes over a vacated seat *with that seat's rating*, so points are neither created nor lost
        (a rookie entering at a flat 1500 while the retiree sat lower would inflate everybody's rating).
        With no seat to inherit it starts at the field's lowest rating.

        With `evolve` on, most rookies are mutated children of a top fly (see evolution.py); the rest are
        random immigrants. The very first field (breed=False) is all random: nobody has learned anything yet."""
        if rating is None:
            rs = [q["rating"] for q in self.trained()]
            rating = min(rs) if rs else START_RATING
        pid = f"p{self.next_pid}"
        self.next_pid += 1
        seed = self.rng.randrange(1, 10**9)
        names, numbers, colors = self._taken()
        ident = make_identity(seed, names, numbers, colors)
        parent = None
        if breed and self.cfg["evolve"] and self.rng.random() >= self.cfg["immigrant_share"]:
            parent = choose_parent(self.trained(), self.pilots.get("legend"), self.rng, self.cfg["legend_parent_share"])
        mutation = None
        if parent is not None:
            brain, mutation = mutate_brain(parent["brain"], seed, self.cfg["mutation_rate"])
        else:
            rr = random.Random(seed)
            brain = FlyBrain(n_pn=N_PN, seed=seed, noise_sigma=round(rr.uniform(0.3, 0.9), 3),
                             eta=round(rr.uniform(0.01, 0.04), 4), lat_kc_fraction=0.25)
        pilot = self._new_pilot(pid, ident, brain, seed)
        if parent is not None:
            pilot.update(gen=parent.get("gen", 0) + 1, parent=parent["name"], parent_pid=parent["id"], origin="offspring",
                         dl=copy.deepcopy(parent["dl"]), dt=copy.deepcopy(parent["dt"]))
        elif breed:
            pilot["origin"] = "immigrant"
        self.pilots[pid] = pilot
        pilot["rating"] = round(rating, 1)
        self.history[pid] = [[self.gp, pilot["rating"]]]
        self.log_join(pilot, mutation)
        return pilot

    def rookie_text(self, r: dict, prefix: str = "Rookie ") -> str:
        kin = f", child of {r['parent']}" if r.get("parent") else ""
        return f"{prefix}{r['name']} ({r['nation']}{kin}) joins as #{r['number']}"

    def log_join(self, p: dict, mutation: dict | None = None) -> None:
        analysis.append("pilots", {
            "event": "join", "league": self.league_seed, "pid": p["id"], "name": p["name"], "nation": p["nation"],
            "number": p["number"], "legend": bool(p["legend"]), "gp": self.gp, "origin": p.get("origin"),
            "gen": p.get("gen", 0), "parent": p.get("parent"), "parent_pid": p.get("parent_pid"), "seed": p["seed"],
            "rating_start": p["rating"], "genome": genome(p["brain"]), "mutation": mutation,
            "training": {k: self.cfg[k] for k in ("stage_episodes", "randomize_prob", "nominal_share", "mutation_rate",
                                                   "immigrant_share", "legend_parent_share", "revert_gap", "evolve")}})

    # -- revert to the best brain a fly ever had (the plasticity rule random-walks away from good solutions)
    def track_form(self, gp: int) -> list[dict]:
        gap = self.cfg["revert_gap"]
        events = []
        if gap <= 0:
            return events
        for p in self.trained():
            snap = p.get("snap")
            if snap is None or p["rating"] > snap["rating"]:
                p["snap"] = {"rating": p["rating"], "gp": gp, "brain": copy.deepcopy(p["brain"]),
                             "dl": copy.deepcopy(p["dl"]), "dt": copy.deepcopy(p["dt"])}
            elif p["rating"] < snap["rating"] - gap and gp - snap["gp"] >= 3 and gp - p.get("revert_gp", -99) >= 3:
                p["brain"] = copy.deepcopy(snap["brain"])
                p["brain"].rng = np.random.default_rng(self.rng.randrange(1, 10**9))
                p["brain"].eligibility[:] = 0.0
                p["dl"], p["dt"] = copy.deepcopy(snap["dl"]), copy.deepcopy(snap["dt"])
                p["revert_gp"], p["reverts"] = gp, p.get("reverts", 0) + 1
                events.append({"kind": "revert", "pid": p["id"],
                               "text": f"{p['name']} returns to their best form (peak {snap['rating']:.0f} Elo)"})
                snap["rating"], snap["gp"] = p["rating"], gp
        return events

    def log_round(self, gp, season, scenario, order, raced, deltas, rating_before, train_stats) -> None:
        for pos, pid in enumerate(order, 1):
            p, r = self.pilots[pid], raced[pid]["result"]
            ts = train_stats.get(pid)
            analysis.append("rounds", {
                "league": self.league_seed, "gp": gp, "season": season, "scenario": scenario, "pid": pid, "name": p["name"],
                "legend": bool(p["legend"]), "gen": p.get("gen", 0), "origin": p.get("origin"), "pos": pos, "of": len(order),
                "status": r["status"], "time": round(r["time"], 3) if r["status"] == "finished" else None,
                "x": round(r["x"], 1), "rating_before": round(rating_before[pid], 1), "rating_after": round(p["rating"], 1),
                "delta": round(deltas[pid], 2), "episodes": p["episodes"],
                "eta": round(float(p["brain"].eta), 5), "noise_sigma": round(float(p["brain"].noise_sigma), 4),
                "train": ts and {"n": ts["n"], "finished": ts["finished"], "crash": ts["crash"], "timeout": ts["timeout"],
                                 "reward": round(ts["reward"], 2),
                                 "avg_time": round(ts["time_sum"] / ts["finished"], 3) if ts["finished"] else None},
                "stage_episodes": self.cfg["stage_episodes"], "randomize_prob": self.cfg["randomize_prob"]})

    @staticmethod
    def names_path():
        return CANDIDATES.parent / "names.json"

    def load_known_names(self) -> set:
        try:
            return set(json.loads(self.names_path().read_text(encoding="utf-8")))
        except Exception:
            return set()

    def _new_pilot(self, pid, ident, brain, seed) -> dict:
        self.used_names.add(ident["name"])
        try:
            self.names_path().parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(self.names_path(), sorted(self.used_names))
        except Exception:
            pass
        return {
            "id": pid, **ident, "seed": seed, "brain": brain,
            "dl": ValueDopamineTracker(N_PN), "dt": DopamineTracker(),
            "episodes": 0, "rating": START_RATING, "last_delta": 0.0, "pb": None,
            "wins": 0, "podiums": 0, "races": 0, "finishes": 0, "form": [],
            "joined_gp": self.gp, "beat_legend": False,
            "gen": 0, "parent": None, "parent_pid": None, "origin": "initial",
        }

    def archive(self, p: dict, reason: str, gp: int) -> None:
        """Keep the fly's brain (archive/pilots) and write its final line to the analysis log."""
        archive_pilot(p, reason, gp)
        analysis.append("pilots", {
            "event": "exit", "league": self.league_seed, "pid": p["id"], "name": p["name"], "legend": bool(p["legend"]),
            "reason": reason, "gp": gp, "gen": p.get("gen", 0), "origin": p.get("origin"), "titles": p.get("titles", 0),
            "reverts": p.get("reverts", 0),
            "final": {k: (round(p[k], 2) if isinstance(p.get(k), float) else p.get(k))
                      for k in ("rating", "peak", "episodes", "races", "wins", "podiums", "finishes", "pb")}})

    def trained(self) -> list[dict]:
        return [p for p in list(self.pilots.values()) if not p["legend"]]

    def meta(self, p: dict) -> dict:
        return {**{k: p[k] for k in ("name", "nation", "number", "color", "legend", "joined_gp")},
                "titles": p.get("titles", 0), "peak": round(p.get("peak", p["rating"]), 1),
                "gen": p.get("gen", 0), "parent": p.get("parent"), "origin": p.get("origin")}

    def apply_field_settings(self):
        """Bring the roster in line with the current settings (round boundary only)."""
        if self.cfg["legend"] and "legend" not in self.pilots:
            self.add_legend()
        elif not self.cfg["legend"] and "legend" in self.pilots:
            self.archive(self.pilots["legend"], "removed", self.gp)
            del self.pilots["legend"]
            self.history.pop("legend", None)
        while len(self.trained()) < self.cfg["pilots"]:
            r = self.add_rookie()
            self.pending_events.append({"kind": "rookie", "pid": r["id"],
                                       "text": self.rookie_text(r, "")})
        while len(self.trained()) > self.cfg["pilots"]:
            worst = min(self.trained(), key=lambda p: p["rating"])
            self.archive(worst, "removed", self.gp)
            del self.pilots[worst["id"]]
            self.history.pop(worst["id"], None)
            self.pending_events.append({"kind": "retire", "text": f"{worst['name']} leaves the league"})

    def snapshot(self, prev: list[dict] | None = None) -> list[dict]:
        prev_rank = {s["pid"]: s["rank"] for s in prev} if prev else {}
        out = []
        for rank, p in enumerate(sorted(self.pilots.values(), key=lambda q: -q["rating"]), 1):
            out.append({
                "pid": p["id"], "rank": rank, "rank_prev": prev_rank.get(p["id"], rank),
                "rating": round(p["rating"], 1), "delta": round(p["last_delta"], 1), "pb": p["pb"],
                "wins": p["wins"], "podiums": p["podiums"], "races": p["races"], "finishes": p["finishes"],
                "form": p["form"][-5:], "episodes": p["episodes"], "train": p.get("last_train"),
            })
        return out

    # -- the goal: verification outcomes
    def process_goal(self):
        outs = self.goal.poll()
        for o in outs:
            if o["promoted"]:
                self.pending_events.append({
                    "kind": "champion", "pid": o["pid"], "name": o["name"], "total": o["total"],
                    "text": f"NEW CHAMPION: {o['name']} {o['total']:.2f} s total, {abs(o['delta']):.2f} s better than the Legend"})
                self.coronate(o)
            else:
                if o["total"] is not None:
                    text = f"{o['name']} verified on all scenarios: {o['total']:.2f} s total ({o['delta']:+.2f} vs champion)"
                else:
                    text = f"{o['name']} did not finish every scenario in verification"
                if o["crashes"]:
                    text += f", {o['crashes']} crashes"
                self.pending_events.append({"kind": "verify", "pid": o["pid"], "text": text})
        if outs:
            self.write_state()

    def drain_goal(self, timeout: float = 900.0):
        """On pause/stop, wait for verifications already running -- they are part of the goal."""
        if not self.goal.pending:
            return
        self.write_live("verifying")
        self.log(f"waiting for {len(self.goal.pending)} verification(s) to finish")
        t0 = time.time()
        while self.goal.pending and time.time() - t0 < timeout:
            self.process_goal()
            time.sleep(1)

    def pick_scenario(self) -> str:
        if self.rng.random() < self.cfg["nominal_share"]:
            return "nominal"
        return self.rng.choice([k for k in SCENARIOS if k != "nominal"])

    # -- output
    def broadcast(self) -> dict:
        return {**{k: self.cfg[k] for k in BROADCAST_KEYS}, "lead_rounds": self.cfg["lead_rounds"]}

    def write_live(self, phase: str | None = None, done: int = 0, total: int = 0, done_pids=(), gp: int | None = None):
        if phase is not None:
            self.phase = phase
        now = time.time()
        eps = sum(p["episodes"] for p in self.trained())
        self.eps_window.append((now, eps))
        self.eps_window = [(t, e) for t, e in self.eps_window if now - t <= 120]
        t0, e0 = self.eps_window[0]
        rate = (eps - e0) / (now - t0) if now - t0 > 1 and self.status != "stopped" else 0.0
        atomic_write_json(LIVE_PATH, {
            "phase": self.phase, "status": self.status, "gp": gp if gp is not None else self.gp + 1,
            "done": done, "total": total, "done_pids": list(done_pids), "episodes": eps,
            "eps_per_s": round(max(rate, 0.0), 1), "ts": now, "broadcast": self.broadcast(),
        })

    def write_state(self):
        atomic_write_json(STATE_PATH, {
            "league": self.cfg["name"], "ts": time.time(), "gp": self.gp,
            "season": self.season, "season_gp": self.season_gp + 1, "season_len": self.cfg["retire_every"],
            "standings": self.snapshot(), "pilots": {pid: self.meta(p) for pid, p in self.pilots.items()},
            "history": {pid: self.history[pid][-HISTORY_POINTS:] for pid in self.pilots},
            "races": self.races[-LISTED_RACES:], "records": self.records, "events": self.events[-40:],
            "totals": {"episodes": sum(p["episodes"] for p in self.trained()), "races": self.gp,
                       "stage_episodes": self.cfg["stage_episodes"]},
            "goal": self.goal.info(),
        })

    # -- control (called from the admin API)
    def start(self) -> str:
        with self.lock:
            if self.thread and self.thread.is_alive():
                self.pause_req = False
                if self.status == "pausing":
                    self.status = "running"
                    self.write_live()
                return "already running"
            self.pause_req = self.abort_req = False
            self.status = "running"
            self.started_at = time.time()
            self.thread = threading.Thread(target=self._run, name="league", daemon=True)
            self.thread.start()
        self.log("started")
        return "started"

    def pause(self) -> str:
        with self.lock:
            if not (self.thread and self.thread.is_alive()):
                return "not running"
            self.pause_req = self.abort_req = True
            self.status = "pausing"
        self.write_live()
        self.log("pause requested -- stopping right now")
        return "paused (an unfinished round is dropped)"

    def reset(self) -> str:
        if not self.stop_now():
            return "could not stop the running league"
        with self.lock:
            self.archive_league(f"reset_gp{self.gp}")
            self.reset_state()
            self.wipe_files()
            self.init_field()
            self.viewer_gp = 0
        self.write_state()
        self.write_live("stopped")
        self.log(f"league reset: {len(self.pilots)} pilots")
        return "reset"

    def hard_restart(self) -> str:
        """Stop at once (even mid-round), archive the whole league, create a brand-new one and start it right away."""
        with self.lock:
            if getattr(self, "restarting", False):
                return "already restarting"
            self.restarting = True

        def work():
            try:
                self.log("hard restart: stopping the current run")
                if self.reset() == "reset":
                    self.start()
            finally:
                self.restarting = False

        threading.Thread(target=work, name="hard-restart", daemon=True).start()
        return "hard restart: new league starts in a moment (the old one is archived)"

    def set_settings(self, raw: dict) -> dict:
        with self.lock:
            before = dict(self.cfg)
            self.cfg = clean_settings(raw, self.cfg)
            changed = [k for k in self.cfg if self.cfg[k] != before[k]]
            self.save_settings()
        if changed:
            self.log("settings changed: " + ", ".join(f"{k}={self.cfg[k]}" for k in changed))
            analysis.append("leagues", {"event": "settings", "league": self.league_seed, "gp": self.gp,
                                        "changed": {k: self.cfg[k] for k in changed}})
        self.write_live()
        return {"changed": changed, "reset_needed": [k for k in changed if SPEC_BY_KEY[k]["apply"] == "reset"]}

    def heartbeat(self, gp):
        try:
            self.viewer_gp = int(gp)
        except (TypeError, ValueError):
            pass
        self.viewer_ts = time.time()

    def viewer_alive(self) -> bool:
        return time.time() - self.viewer_ts < VIEWER_TIMEOUT

    def admin_status(self) -> dict:
        pilots = sorted(({"pid": p["id"], "name": p["name"], "nation": p["nation"], "number": p["number"],
                          "rating": round(p["rating"], 1), "episodes": p["episodes"], "races": p["races"],
                          "pb": p["pb"], "legend": p["legend"]} for p in list(self.pilots.values())),
                        key=lambda p: -p["rating"])
        try:
            live = json.loads(LIVE_PATH.read_text())
        except (OSError, ValueError):
            live = {}
        return {
            "status": self.status, "phase": self.phase, "gp": self.gp - (1 if self.phase == "training" else 0),
            "season": self.season,
            "season_gp": self.season_gp, "settings": self.cfg, "spec": SETTINGS_SPEC, "live": live,
            "viewer": {"connected": self.viewer_alive(), "gp": self.viewer_gp},
            "uptime": time.time() - self.started_at if self.started_at and self.status != "stopped" else 0,
            "pilots": pilots, "log": list(self.logbuf)[-120:], "goal": self.goal.info(),
        }

    # -- the run loop
    def _run(self):
        pool, workers = None, 0
        try:
            while not self.pause_req:
                self.process_goal()
                self.sync_legend()
                self.apply_field_settings()
                if not self.wait_for_viewer():
                    break
                if pool is None or workers != self.cfg["workers"]:
                    if pool:
                        pool.shutdown(wait=True)
                    workers = self.cfg["workers"]
                    pool = ProcessPoolExecutor(max_workers=workers, initializer=lower_priority)
                    self.goal.resubmit(pool)
                t0 = time.time()
                self.play_gp(pool)
                self.pace(t0)
        except Aborted:
            self.rollback()
        except Exception:
            self.log("league crashed:\n" + traceback.format_exc())
        finally:
            if pool:
                try:
                    if not self.abort_req:
                        self.drain_goal()
                finally:
                    if self.abort_req:
                        kill_pool(pool)
                    else:
                        pool.shutdown(wait=False, cancel_futures=True)
            self.status = "stopped"
            self.write_live("stopped")
            self.log("stopped")

    def rollback(self):
        """A round was dropped half-way: go back to the last finished (checkpointed) round."""
        gp = self.round_gp
        if not self.load():
            self.gp = gp - 1
        self.pending_events = []
        self.write_state()
        self.log(f"stopped immediately -- the unfinished Grand Prix {gp} was dropped")

    def stop_now(self, timeout: float = 90.0) -> bool:
        """Stop a running league at once (a round in progress is dropped) and wait until it is down."""
        t = self.thread
        if t and t.is_alive():
            self.abort_req = self.pause_req = True
            self.status = "pausing"
            self.write_live()
            t.join(timeout=timeout)
            if t.is_alive():
                return False
        self.abort_req = False
        return True

    def wait_for_viewer(self) -> bool:
        """Training only runs `lead_rounds` ahead of the round the page is showing,
        so rounds stay continuous without the state racing far ahead of the screen."""
        lead = self.cfg["lead_rounds"]
        last_live = 0.0
        while not self.pause_req:
            if lead <= 0 or not self.viewer_alive() or self.gp - self.viewer_gp < lead:
                return True
            self.process_goal()
            if time.time() - last_live > 2.0:
                self.write_live("waiting", gp=self.gp + 1)
                last_live = time.time()
            time.sleep(0.4)
        return False

    def pace(self, t0: float):
        if self.cfg["lead_rounds"] > 0 and self.viewer_alive():
            return
        while not self.pause_req and time.time() - t0 < self.cfg["min_round_seconds"]:
            time.sleep(0.25)

    # -- one Grand Prix
    def play_gp(self, pool) -> None:
        t_start = time.time()
        self.gp += 1
        gp = self.gp
        self.round_gp = gp
        season, season_gp = self.season, self.season_gp + 1
        scenario = self.pick_scenario()
        title = self.rng.choice(CONDITION_TITLES[scenario])
        race_seed = self.rng.randrange(2_000_000_000, 2_100_000_000)
        before = self.snapshot()
        grid = [s["pid"] for s in before]  # best-rated pilot gets lane 1
        rating_before = {pid: self.pilots[pid]["rating"] for pid in grid}

        trained = self.trained()
        self.write_live("training", 0, len(trained), (), gp)
        futures = {pool.submit(_stage_task, p["brain"], p["dl"], p["dt"], p["episodes"], p["seed"],
                               self.cfg["stage_episodes"], self.cfg["randomize_prob"], scenario, race_seed): p["id"]
                   for p in trained}
        raced = {}
        if "legend" in self.pilots:
            raced["legend"] = record_race(self.pilots["legend"]["brain"], scenario, race_seed)
        train_stats = {}
        done_pids = []
        waiting = set(futures)
        while waiting:
            if self.abort_req:
                raise Aborted()
            done, waiting = wait(waiting, timeout=0.25, return_when=FIRST_COMPLETED)
            for fut in done:
                pid = futures[fut]
                p = self.pilots[pid]
                p["brain"], p["dl"], p["dt"], p["episodes"], train_stats[pid], raced[pid] = fut.result()
                s = train_stats[pid]
                p["last_train"] = [s["finished"], s["crash"], s["timeout"], s["n"]]
                done_pids.append(pid)
                self.write_live("training", len(done_pids), len(trained), done_pids, gp)

        # classification: finishers by time, everyone else by distance covered
        def key(pid):
            r = raced[pid]["result"]
            return (0, r["time"], 0.0) if r["status"] == "finished" else (1, 0.0, -r["x"])
        order = sorted(grid, key=key)
        ranks = [key(pid) for pid in order]

        deltas = self.update_ratings(order, ranks)
        winner_time = raced[order[0]]["result"]["time"]
        events = self.pending_events
        self.pending_events = []
        events += self.update_stats_and_events(order, raced, scenario, winner_time)
        self.goal.consider(pool, gp, scenario, order, self.pilots, raced)
        events += self.track_form(gp)
        self.log_round(gp, season, scenario, order, raced, deltas, rating_before, train_stats)
        after = self.snapshot(before)
        for pid in grid:
            self.history[pid].append([gp, round(self.pilots[pid]["rating"], 1)])

        classification = []
        for pos, pid in enumerate(order, 1):
            r = raced[pid]["result"]
            classification.append({
                "pos": pos, "pid": pid, **r,
                "gap": round(r["time"] - winner_time, 3) if r["status"] == "finished" and winner_time else None,
                "rating_delta": round(deltas[pid], 1),
            })
        # read everything about this race's field before the season rollover can retire someone
        meta_pilots = {pid: self.meta(self.pilots[pid]) for pid in grid}
        names = {pid: self.pilots[pid]["name"] for pid in grid}
        history = {pid: self.history[pid][-60:] for pid in grid}
        records = json.loads(json.dumps(self.records))
        self.season_gp += 1
        if self.season_gp >= self.cfg["retire_every"]:
            events += self.season_end()
            self.season += 1
            self.season_gp = 0

        race_id = f"gp_{gp:04d}"
        race = {
            "id": race_id, "gp": gp, "season": season, "season_gp": season_gp, "season_len": self.cfg["retire_every"],
            "title": title, "seed": race_seed,
            "conditions": condition_info(scenario), "dt": 0.02, "track_length": DragStripEnv().track_length,
            "k_green": next(iter(raced.values()))["k_green"], "pilots": meta_pilots,
            "entries": [{"pid": pid, "lane": lane, "trace": raced[pid]["trace"], "result": raced[pid]["result"]}
                        for lane, pid in enumerate(grid, 1)],
            "classification": classification, "before": before, "after": after, "events": events,
            "ratings_before": {pid: round(v, 1) for pid, v in rating_before.items()},
            "history": history, "records": records,
        }
        atomic_write_json(RACES_DIR / f"{race_id}.json", race)
        self.races.append({"id": race_id, "gp": gp, "season": season, "title": title, "conditions": scenario,
                           "winner": order[0], "winner_name": names[order[0]], "time": winner_time, "ts": time.time()})
        self.events.extend({"gp": gp, "ts": time.time(), **e} for e in events)
        for e in events:
            analysis.append("events", {"league": self.league_seed, "gp": gp, **e})
        self.prune_races()
        self.save()
        self.write_state()
        self.write_live("waiting", len(trained), len(trained), done_pids, gp + 1)

        top = ", ".join(f"{i}. {names[pid]} "
                        + (f"{raced[pid]['result']['time']:.2f}s" if raced[pid]['result']['status'] == 'finished'
                           else raced[pid]['result']['status'].upper())
                        for i, pid in enumerate(order[:3], 1))
        self.log(f"GP {gp:>3} [{scenario}] {title}: {top}  ({time.time() - t_start:.0f}s)")
        self.process_goal()

    def update_ratings(self, order, ranks) -> dict:
        k = self.cfg["k"]
        n = len(order)
        deltas = {pid: 0.0 for pid in order}
        if n < 2:
            return deltas
        for i, a in enumerate(order):
            for j, b in enumerate(order):
                if i == j:
                    continue
                ra, rb = self.pilots[a]["rating"], self.pilots[b]["rating"]
                expected = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
                score = 0.5 if ranks[i] == ranks[j] else (1.0 if i < j else 0.0)
                deltas[a] += k / (n - 1) * (score - expected)
        for pid, d in deltas.items():
            p = self.pilots[pid]
            p["rating"] += d
            p["last_delta"] = d
            p["peak"] = max(p.get("peak", p["rating"]), p["rating"])
        return deltas

    def update_stats_and_events(self, order, raced, scenario, winner_time) -> list[dict]:
        events = []
        legend_res = raced.get("legend", {}).get("result")
        for pos, pid in enumerate(order, 1):
            p, r = self.pilots[pid], raced[pid]["result"]
            finished = r["status"] == "finished"
            p["races"] += 1
            if finished:
                if p["finishes"] == 0 and not p["legend"]:
                    events.append({"kind": "first_finish", "pid": pid,
                                   "text": f"{p['name']} finishes a race for the first time ({r['time']:.2f} s)"})
                p["finishes"] += 1
                if p["pb"] is None or r["time"] < p["pb"]:      # best finish in any conditions (hot/cold/patchy are only slower)
                    p["pb"] = round(r["time"], 3)
                if scenario == "nominal":
                    rec = self.records.get("nominal")
                    if rec is None:
                        self.records["nominal"] = {"pid": pid, "name": p["name"], "time": round(r["time"], 3), "gp": self.gp}
                    elif r["time"] < rec["time"] - 0.05:
                        self.records["nominal"] = {"pid": pid, "name": p["name"], "time": round(r["time"], 3), "gp": self.gp}
                        events.append({"kind": "record", "pid": pid,
                                       "text": f"NEW TRACK RECORD: {p['name']} {r['time']:.2f} s"})
                if (not p["legend"] and not p["beat_legend"] and legend_res and legend_res["status"] == "finished"
                        and r["time"] < legend_res["time"]):
                    p["beat_legend"] = True
                    events.append({"kind": "beat_legend", "pid": pid,
                                   "text": f"{p['name']} BEATS THE LEGEND on {r['time']:.2f} s!"})
            if finished and pos == 1:
                p["wins"] += 1
            if finished and pos <= 3:
                p["podiums"] += 1
            p["form"].append("W" if finished and pos == 1 else "F" if finished else
                             "T" if r["status"] == "timeout" else "C")
            p["form"] = p["form"][-10:]
        return events

    def crown_season(self, events: list) -> None:
        """The season's champion is the best-rated trained fly; the title stays with the fly for good."""
        table = sorted(self.trained(), key=lambda p: -p["rating"])
        if not table:
            return
        champ = table[0]
        champ["titles"] = champ.get("titles", 0) + 1
        top = [{"pid": p["id"], "name": p["name"], "rating": round(p["rating"], 1)} for p in table[:3]]
        events.append({"kind": "season_champion", "pid": champ["id"], "name": champ["name"], "season": self.season,
                       "rating": round(champ["rating"], 1), "titles": champ["titles"], "top": top,
                       "text": f"SEASON {self.season} CHAMPION: {champ['name']} ({champ['rating']:.0f} Elo)"})
        path = CANDIDATES.parent / "seasons.json"
        try:
            log = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        except Exception:
            log = []
        log.append({"season": self.season, "gp": self.gp, "ts": time.time(), "champion": top[0], "podium": top,
                    "peak": round(champ.get("peak", champ["rating"]), 1)})
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(path, log)
        except Exception:
            pass

    def soft_reset_ratings(self, events: list) -> None:
        keep = self.cfg["season_carry"]
        if keep >= 1:
            return
        for p in self.pilots.values():
            p["rating"] = START_RATING + (p["rating"] - START_RATING) * keep
            if p.get("snap"):
                p["snap"]["rating"] = START_RATING + (p["snap"]["rating"] - START_RATING) * keep
        events.append({"kind": "season_reset", "text": "New season: ratings pulled "
                       + ("back to 1500" if keep <= 0 else f"{round((1 - keep) * 100)}% of the way back to 1500")})

    def season_end(self) -> list[dict]:
        events = [{"kind": "season", "text": f"Season {self.season} complete"}]
        gp = self.gp
        self.crown_season(events)
        for _ in range(self.cfg["retire_count"]):
            eligible = [p for p in self.trained() if gp - p["joined_gp"] + 1 >= self.cfg["min_age"]]
            if not eligible or len(self.trained()) < 2:
                break
            worst = min(eligible, key=lambda p: (p["rating"], p["finishes"]))
            self.archive(worst, "retired", gp)
            del self.pilots[worst["id"]]
            self.history.pop(worst["id"], None)
            best = f", best {worst['pb']:.2f} s" if worst["pb"] else ""
            events.append({"kind": "retire", "text": f"{worst['name']} ({worst['nation']}) retires after "
                           f"{worst['races']} races{best}"})
            rookie = self.add_rookie(worst["rating"])
            events.append({"kind": "rookie", "pid": rookie["id"],
                           "text": self.rookie_text(rookie)})
        self.soft_reset_ratings(events)
        return events

    def prune_races(self):
        for f in sorted(RACES_DIR.glob("gp_*.json"))[:-KEEP_RACES]:
            f.unlink(missing_ok=True)


def condition_info(key: str) -> dict:
    s = SCENARIOS[key]
    return {
        "key": key,
        "engine": round(100 * s["engine_power_scale"]),
        "grip": round(100 * s["traction_scale"]),
        "patchy": s["traction_noise_sigma"] > 0,
    }


# ---------------------------------------------------------------- web server

LOOPBACK = ("127.0.0.1", "::1", "::ffff:127.0.0.1")


class Handler(SimpleHTTPRequestHandler):
    league: League = None

    def translate_path(self, path):
        path = urllib.parse.unquote(path.split("?", 1)[0].split("#", 1)[0])
        if path.startswith("/data/"):
            base, rel = DATA_DIR, path[len("/data/"):]
        else:
            base, rel = WEB_DIR, path.lstrip("/") or "index.html"
        full = (base / rel).resolve()
        if base.resolve() not in full.parents and full != base.resolve():
            return str(base / "__forbidden__")
        return str(full)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *a):
        pass

    def _local(self) -> bool:
        return self.client_address[0] in LOOPBACK

    def _send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/admin/status":
            if not self._local():
                return self._send_json({"error": "admin is local-only"}, 403)
            return self._send_json(self.league.admin_status())
        if path in ("/admin", "/admin/"):
            if not self._local():
                return self._send_json({"error": "admin is local-only"}, 403)
            self.path = "/admin.html"
        super().do_GET()

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        try:
            n = min(int(self.headers.get("Content-Length") or 0), 65536)
            body = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, OSError):
            return self._send_json({"error": "bad request"}, 400)
        if path == "/api/viewer":
            self.league.heartbeat(body.get("gp"))
            return self._send_json({"ok": True})
        if path.startswith("/api/admin/"):
            # custom header: a cross-site page can't send it without a CORS preflight we never allow
            if not self._local() or self.headers.get("X-Admin") != "1":
                return self._send_json({"error": "forbidden"}, 403)
            if path == "/api/admin/settings":
                return self._send_json(self.league.set_settings(body))
            if path == "/api/admin/action":
                act = body.get("action")
                fn = {"start": self.league.start, "pause": self.league.pause, "reset": self.league.reset,
                                                  "hard_restart": self.league.hard_restart}.get(act)
                if fn is None:
                    return self._send_json({"error": "unknown action"}, 400)
                return self._send_json({"result": fn()})
        self._send_json({"error": "not found"}, 404)


def serve(league: League, host: str, port: int):
    Handler.league = league
    srv = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=srv.serve_forever, name="http", daemon=True).start()
    return srv


# ---------------------------------------------------------------- main

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for s in SETTINGS_SPEC:
        flag = "--" + s["key"].replace("_", "-")
        if s["type"] == "bool":
            ap.add_argument(flag, action=argparse.BooleanOptionalAction, default=None, help=s["help"])
        else:
            ap.add_argument(flag, type={"int": int, "float": float, "text": str}[s["type"]], default=None,
                            help=f"{s['help']} (default {s['default']})")
    ap.add_argument("--legend-path", default=str(LEGEND_PATH))
    ap.add_argument("--fresh", action="store_true", help="discard saved league state and start over")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-serve", action="store_true", help="only write stream_data/, don't start the web server")
    return ap


def main():
    lower_priority()
    args = build_parser().parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RACES_DIR.mkdir(parents=True, exist_ok=True)

    # defaults < settings.json (edited in the admin panel) < explicit command-line flags
    cfg = dict(DEFAULTS)
    if SETTINGS_PATH.exists():
        try:
            cfg = clean_settings(json.loads(SETTINGS_PATH.read_text(encoding="utf-8-sig")), cfg)
        except Exception as e:                       # a damaged settings file must never stop the program
            print(f"settings.json is unreadable ({e!r}); using the defaults", flush=True)
    cfg = clean_settings({s["key"]: getattr(args, s["key"]) for s in SETTINGS_SPEC}, cfg)

    league = League(cfg, args.legend_path)
    league.save_settings()
    if args.fresh:
        league.archive_league("fresh_start")
        league.wipe_files()
    if league.load():
        league.log(f"resumed: GP {league.gp}, {len(league.pilots)} pilots")
    else:
        league.init_field()
        league.log(f"new league: {len(league.pilots)} pilots")
    for p in sorted(league.pilots.values(), key=lambda q: q["number"]):
        league.log(f"  #{p['number']:>2} {p['name']:<22} {p['nation']}{'  (Legend)' if p['legend'] else ''}")
    league.write_state()
    league.write_live("stopped")

    if not args.no_serve:
        serve(league, args.host, args.port)
        league.log(f"viewer http://{args.host}:{args.port}/   admin http://{args.host}:{args.port}/admin")
    league.log("stopped -- press Start in the admin panel")        # a run is always started from the admin

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        league.log("shutting down; state is saved after every Grand Prix")
        league.pause_req = True
        try:
            if league.thread:
                league.thread.join(timeout=180)   # let the current round finish; Ctrl+C again to force
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
