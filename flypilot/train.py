from __future__ import annotations

import statistics as stats

from .brain import FlyBrain
from .env import Action, DragStripEnv
from .learning import DopamineTracker
from .sense import N_PN, SenseEncoder


def run_fly_episode(env: DragStripEnv, brain: FlyBrain, encoder: SenseEncoder,
                     dopamine_lon: DopamineTracker, dopamine_lat: DopamineTracker,
                     seed: int, learn: bool = True) -> dict:
    obs = env.reset(seed=seed)
    encoder.reset()
    brain.reset()
    total_reward = 0.0
    ticks = 0
    while True:
        pn = encoder.encode(obs)
        out = brain.forward(pn, encoder._green_trace, explore=learn)
        # not-before-green is a hard safety gate, not something worth spending
        # learning capacity (and risking foul-start crashes) on -- the reflex
        # pathway still gets to prove itself the instant the light turns green.
        staged = obs.phase == "staged"
        throttle = 0.0 if staged else out["throttle"]
        action = Action(throttle=throttle, steer=out["steer"], shift=out["shift"])
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if learn and not staged:
            d_lon = dopamine_lon.step(info["reward_lon"])
            d_lat = dopamine_lat.step(info["reward_lat"])
            brain.apply_dopamine(d_lon, d_lat)
        ticks += 1
        if done or ticks > 1500:  # 30s of race clock -- generous, real runs finish well under 15s
            break
    return {
        "phase": obs.phase,
        "elapsed_time": info.get("elapsed_time"),
        "reaction_time": info.get("reaction_time"),
        "total_reward": total_reward,
    }


def run_baseline_episode(env: DragStripEnv, controller, seed: int) -> dict:
    obs = env.reset(seed=seed)
    total_reward = 0.0
    ticks = 0
    while True:
        action = controller.act(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        ticks += 1
        if done or ticks > 1500:
            break
    return {
        "phase": obs.phase,
        "elapsed_time": info.get("elapsed_time"),
        "reaction_time": info.get("reaction_time"),
        "total_reward": total_reward,
    }


def train(n_episodes: int = 3000, report_every: int = 200, seed0: int = 0,
          brain_kwargs: dict | None = None, quiet: bool = False) -> FlyBrain:
    env = DragStripEnv()
    encoder = SenseEncoder()
    brain = FlyBrain(n_pn=N_PN, seed=seed0, **(brain_kwargs or {}))
    dopamine_lon = DopamineTracker()
    dopamine_lat = DopamineTracker()

    window = []
    for ep in range(n_episodes):
        result = run_fly_episode(
            env, brain, encoder, dopamine_lon, dopamine_lat, seed=seed0 + ep, learn=True
        )
        window.append(result)
        if not quiet and (ep + 1) % report_every == 0:
            recent = window[-report_every:]
            finishes = [r for r in recent if r["phase"] == "finished"]
            fouls = sum(1 for r in recent if r["phase"] == "foul")
            crashes = sum(1 for r in recent if r["phase"] == "crash")
            timeouts = sum(1 for r in recent if r["phase"] == "racing")
            avg_time = stats.mean(r["elapsed_time"] for r in finishes) if finishes else float("nan")
            avg_reward = stats.mean(r["total_reward"] for r in recent)
            print(
                f"ep {ep + 1:5d}  finished {len(finishes):3d}/{report_every}  "
                f"foul {fouls:3d}  crash {crashes:3d}  timeout {timeouts:3d}  "
                f"avg_finish_time {avg_time:6.2f}s  avg_reward {avg_reward:7.2f}"
            )
    return brain


if __name__ == "__main__":
    train()
