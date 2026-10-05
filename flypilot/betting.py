"""Viewer betting with virtual points ("FlyCoins") -- no money, no prizes, nothing to cash out.

Viewers bet that a fly finishes in the top three (a *place* pool):

  * every stake goes into one pool; stakes on flies that finish in the top three win,
  * the losing stakes are shared equally between the placed flies that had bets,
  * inside a fly's share the winners split in proportion to their stakes, and everybody gets their stake back,
  * if nobody backed a placed fly, all stakes are returned.

The league (not the page) holds the result, so the page can't be used to peek. Window life cycle, driven by the
broadcast page that knows what is on screen:  open  ->  close (green light)  ->  reveal (podium: payouts applied).
A window that is never revealed is settled by itself after `AUTO_REVEAL_S`.
"""
from __future__ import annotations

import json
import pathlib
import random
import re
import threading
import time

from .goal import DATA_DIR

DIR = DATA_DIR / "bets"
AUTO_REVEAL_S = 90.0
RESCUE_COINS, RESCUE_EVERY_S = 200, 30 * 60
USER_RE = re.compile(r"^[a-z0-9_]{1,25}$")
DEMO_NAMES = ["anna", "bork", "cleo", "dmitri", "elsa", "farid", "gus", "hana", "ivo", "jun", "kira", "lars", "mina", "nox"]


def payouts(stakes: dict, winners: list) -> dict:
    """stakes: {user: {pid: coins}}; winners: placed pids (top three finishers).
    Returns {user: coins returned or won} for every user (0 if they lost everything)."""
    per_fly: dict = {}
    for user, bets in stakes.items():
        for pid, amt in bets.items():
            per_fly[pid] = per_fly.get(pid, 0) + amt
    backed = [pid for pid in winners if per_fly.get(pid)]
    if not backed:                                                   # nobody backed a placed fly: everything comes back
        return {u: sum(b.values()) for u, b in stakes.items()}
    lost = sum(a for pid, a in per_fly.items() if pid not in backed)
    share = lost / len(backed)
    out = {u: 0 for u in stakes}
    parts = []                                                       # (fraction lost to rounding, user), one per bet
    for user, bets in stakes.items():
        for pid, amt in bets.items():
            if pid in backed:
                exact = amt + share * amt / per_fly[pid]
                out[user] += int(exact)
                parts.append((exact - int(exact), user))
    leftover = sum(per_fly.values()) - sum(out.values())             # whole coins lost to rounding go to the largest remainders
    for _, user in sorted(parts, key=lambda x: -x[0])[:max(leftover, 0)]:
        out[user] += 1
    return out


def parse_command(text: str):
    """'!bet 5 100' / '!bet #5 all' / '!bal' / '!top' / '!bets'  ->  (command, args) or None."""
    m = re.match(r"^\s*!(\w+)\s*(.*)$", text or "")
    if not m:
        return None
    cmd, rest = m.group(1).lower(), m.group(2).split()
    cmd = {"ставка": "bet", "баланс": "bal", "топ": "top"}.get(cmd, cmd)
    if cmd == "bet":
        if len(rest) < 2:
            return ("bet", None)
        num = rest[0].lstrip("#")
        amt = rest[1].lower()
        if not num.isdigit() or not (amt.isdigit() or amt == "all"):
            return ("bet", None)
        return ("bet", (int(num), amt))
    return (cmd, rest) if cmd in ("bal", "top", "bets", "help") else None


class Betting:
    def __init__(self, get_cfg, log, send=None):
        self.cfg, self.log, self.send = get_cfg, log, send or (lambda msg: None)
        self.lock = threading.RLock()
        self.wallets: dict = {}
        self.window: dict | None = None
        self.last: dict | None = None
        self.demo_users: dict = {}
        self._load()

    # ---------------------------------------------------------------- persistence
    def _load(self):
        try:
            self.wallets = json.loads((DIR / "wallets.json").read_text(encoding="utf-8"))
        except Exception:
            self.wallets = {}

    def _save(self):
        try:
            DIR.mkdir(parents=True, exist_ok=True)
            tmp = DIR / "wallets.json.tmp"
            tmp.write_text(json.dumps(self.wallets, separators=(",", ":")), encoding="utf-8")
            tmp.replace(DIR / "wallets.json")
        except Exception:
            pass

    def _journal(self, rec: dict):
        try:
            DIR.mkdir(parents=True, exist_ok=True)
            with open(DIR / "bets.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": round(time.time(), 2), **rec}, separators=(",", ":")) + "\n")
        except Exception:
            pass

    # ---------------------------------------------------------------- wallets
    def _wallet(self, user: str) -> dict:
        w = self.wallets.get(user)
        if w is None:
            w = self.wallets[user] = {"coins": self.cfg()["start_coins"], "bets": 0, "wins": 0, "won": 0, "staked": 0, "rescued": 0.0}
        return w

    def _rescue(self, user: str, w: dict) -> str:
        """A bankrupt viewer gets a small top-up now and then, so nobody is locked out of the fun."""
        if w["coins"] < self.cfg()["min_bet"] and time.time() - w.get("rescued", 0) > RESCUE_EVERY_S:
            w["coins"], w["rescued"] = RESCUE_COINS, time.time()
            return f" (you were broke: here are {RESCUE_COINS} coins)"
        return ""

    def top(self, n: int = 8) -> list:
        with self.lock:
            rows = sorted(self.wallets.items(), key=lambda kv: -kv[1]["coins"])[:n]
            return [{"user": u, "coins": w["coins"], "wins": w["wins"], "bets": w["bets"]} for u, w in rows]

    def reset_wallets(self):
        with self.lock:
            self.wallets = {}
            self._save()
        self._journal({"type": "reset"})

    # ---------------------------------------------------------------- the window
    def open_window(self, gp: int, field: list, result: list) -> dict:
        """field: [{pid, number, name, color, rating}], result: pids that will finish in the top three (known to the
        league only)."""
        with self.lock:
            self._auto_reveal()
            self.window = {"gp": gp, "state": "open", "opened": time.time(), "field": field, "winners": result,
                           "stakes": {}, "by_number": {f["number"]: f["pid"] for f in field}}
        self._journal({"type": "open", "gp": gp, "field": [f["pid"] for f in field]})
        self.log(f"betting open for GP {gp}")
        self.send(f"Betting is OPEN for Grand Prix {gp}! Bet that a fly finishes top 3: !bet <number> <coins>  (virtual points only)")
        return self.state()

    def close_window(self, gp: int | None = None) -> dict:
        with self.lock:
            w = self.window
            if w and w["state"] == "open" and (gp is None or w["gp"] == gp):
                w["state"], w["closed"] = "closed", time.time()
                n = len(w["stakes"])
                self._journal({"type": "close", "gp": w["gp"], "bettors": n})
                self.send(f"Betting is CLOSED: {n} viewer(s) bet {sum(sum(b.values()) for b in w['stakes'].values())} coins. Good luck!")
        return self.state()

    def _auto_reveal(self):
        w = self.window
        if w and w["state"] != "settled" and time.time() - w.get("closed", w["opened"]) > AUTO_REVEAL_S:
            self.reveal(w["gp"])

    def reveal(self, gp: int | None = None) -> dict | None:
        with self.lock:
            w = self.window
            if not w or w["state"] == "settled" or (gp is not None and w["gp"] != gp):
                return self.last
            w["state"] = "settled"
            pay = payouts(w["stakes"], w["winners"])
            rows, pool = [], 0
            for user, bets in w["stakes"].items():
                staked, got = sum(bets.values()), pay.get(user, 0)
                pool += staked
                if not user.startswith("demo_"):                          # demo viewers never touch a wallet
                    wl = self._wallet(user)
                    wl["coins"] += got
                    if got > staked:
                        wl["wins"] += 1
                    wl["won"] += max(got - staked, 0)
                rows.append({"user": user, "staked": staked, "got": got, "net": got - staked, "bets": dict(bets)})
            self._save()
            rows.sort(key=lambda r: -r["net"])
            names = {f["pid"]: f for f in w["field"]}
            self.last = {"gp": w["gp"], "winners": [{"pid": p, "number": names[p]["number"], "name": names[p]["name"]} for p in w["winners"] if p in names],
                         "pool": pool, "bettors": len(rows), "top": [r for r in rows if r["net"] > 0][:5], "ts": time.time()}
            self._journal({"type": "settle", "gp": w["gp"], "winners": w["winners"], "pool": pool, "rows": rows})
            podium = ", ".join(f"#{x['number']} {x['name']}" for x in self.last["winners"]) or "no finishers"
            msg = f"GP {w['gp']} top 3: {podium}."
            if rows:
                msg += " Best bet: " + ", ".join(f"{r['user']} {r['net']:+d}" for r in self.last["top"][:3]) if self.last["top"] else " Nobody came out ahead."
            self.send(msg)
            return self.last

    # ---------------------------------------------------------------- placing bets
    def place(self, user: str, number: int, amount) -> tuple[bool, str]:
        user = (user or "").lower().lstrip("@")
        if not USER_RE.match(user):
            return False, "bad user name"
        cfg = self.cfg()
        with self.lock:
            w = self.window
            if not w or w["state"] != "open":
                return False, "betting is closed right now"
            pid = w["by_number"].get(number)
            if pid is None:
                return False, f"no fly #{number} in this race"
            wl = self._wallet(user)
            note = self._rescue(user, wl)
            amt = wl["coins"] if amount == "all" else int(amount)
            amt = min(amt, cfg["max_bet"])
            if amt < cfg["min_bet"]:
                return False, f"minimum bet is {cfg['min_bet']} coins"
            if amt > wl["coins"]:
                return False, f"you only have {wl['coins']} coins"
            wl["coins"] -= amt
            wl["bets"] += 1
            wl["staked"] += amt
            bets = w["stakes"].setdefault(user, {})
            bets[pid] = bets.get(pid, 0) + amt
            self._save()
            f = next(x for x in w["field"] if x["pid"] == pid)
            self._journal({"type": "bet", "gp": w["gp"], "user": user, "pid": pid, "amount": amt})
            return True, f"{amt} coins on #{number} {f['name']} to finish top 3. Balance {wl['coins']}.{note}"

    def handle_chat(self, user: str, text: str) -> str | None:
        """A chat line -> the reply to post (or None)."""
        if not self.cfg()["betting"]:
            return None
        cmd = parse_command(text)
        if cmd is None:
            return None
        name, args = cmd
        user = (user or "").lower()
        if name == "bet":
            if args is None:
                return f"@{user} usage: !bet <fly number> <coins|all>"
            ok, msg = self.place(user, args[0], args[1])
            return f"@{user} {msg}"
        if name == "bal":
            with self.lock:
                w = self._wallet(user)
                note = self._rescue(user, w)
                self._save()
                return f"@{user} you have {w['coins']} coins.{note}"
        if name == "top":
            rows = self.top(5)
            return "Top bettors: " + ", ".join(f"{r['user']} {r['coins']}" for r in rows) if rows else "Nobody has bet yet."
        if name == "bets":
            s = self.state()["window"]
            if not s or s["state"] != "open":
                return "Betting is closed right now."
            pools = ", ".join(f"#{f['number']} {f['pool']}" for f in s["field"] if f["pool"])
            return "Pools: " + pools if pools else "No bets yet."
        if name == "help":
            return "!bet <number> <coins|all> = bet a fly finishes top 3 | !bal | !top | !bets  (virtual points only)"
        return None

    # ---------------------------------------------------------------- what the page shows
    def state(self) -> dict:
        with self.lock:
            self._auto_reveal()
            w = self.window
            win = None
            if w:
                per_fly: dict = {}
                bettors: dict = {}
                for user, bets in w["stakes"].items():
                    for pid, amt in bets.items():
                        per_fly[pid] = per_fly.get(pid, 0) + amt
                        bettors[pid] = bettors.get(pid, 0) + 1
                total = sum(per_fly.values())
                field = []
                for f in w["field"]:
                    pool = per_fly.get(f["pid"], 0)
                    # what one coin on this fly would return if it places, given the stakes so far (3 places share the losing stakes)
                    lost_if = total - pool
                    mult = round(1 + (lost_if / 3) / pool, 2) if pool and lost_if > 0 else (1.0 if pool else None)
                    field.append({**f, "pool": pool, "bettors": bettors.get(f["pid"], 0), "x": mult})
                win = {"gp": w["gp"], "state": w["state"], "total": total, "bettors": len(w["stakes"]), "field": field,
                       "opened": w["opened"]}
            return {"enabled": bool(self.cfg()["betting"]), "window": win, "top": self.top(8), "last": self.last,
                    "wallets": len(self.wallets), "now": time.time()}

    # ---------------------------------------------------------------- demo viewers (clearly fake, never saved)
    def run_demo(self, duration: float):
        """Spread a few fake bets over the window so the screen can be shown without a chat."""
        w = self.window
        if not w or w["state"] != "open":
            return
        gp, field = w["gp"], w["field"]
        rng = random.Random(gp)
        weights = [max(f.get("rating", 1500) - 1300, 50) for f in field]

        def go():
            for name in rng.sample(DEMO_NAMES, rng.randint(6, len(DEMO_NAMES))):
                time.sleep(rng.uniform(0.3, max(duration / 10, 0.6)))
                with self.lock:
                    cur = self.window
                    if not cur or cur["gp"] != gp or cur["state"] != "open":
                        return
                    pid = rng.choices([f["pid"] for f in field], weights=weights)[0]
                    num = next(f["number"] for f in field if f["pid"] == pid)
                    # demo accounts live in the same window but are flagged and refunded: no wallet is touched
                    cur["stakes"].setdefault("demo_" + name, {})[pid] = rng.choice([50, 100, 150, 250, 400])
                    self.demo_users["demo_" + name] = True
        threading.Thread(target=go, name="bet-demo", daemon=True).start()
