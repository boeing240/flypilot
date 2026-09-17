"""Record a full per-tick trace of one episode, for the HTML visualizer."""
from __future__ import annotations

from .env import Action, DragStripEnv
from .sense import SenseEncoder


def _frame(env: DragStripEnv, obs, action: Action) -> dict:
    return {
        "t": round(env.t if obs.phase == "staged" else (env.t_since_green or 0.0), 3),
        "phase": obs.phase,
        "x": round(env.x, 3),
        "y": round(env.y, 3),
        "v": round(obs.speed, 3),
        "rpm": round(obs.rpm, 1),
        "gear": obs.gear,
        "throttle": round(action.throttle, 3),
        "steer": round(action.steer, 3),
        "shift": bool(action.shift),
        "slip": round(env.wheel_slip, 3),
        "dist_left": round(obs.dist_left, 3),
        "dist_right": round(obs.dist_right, 3),
    }


def record_fly_episode(env: DragStripEnv, brain, encoder: SenseEncoder, seed: int, env_kwargs=None) -> dict:
    if env_kwargs:
        for k, v in env_kwargs.items():
            setattr(env, k, v)
    obs = env.reset(seed=seed)
    encoder.reset()
    brain.reset()
    frames = []
    ticks = 0
    while True:
        pn = encoder.encode(obs)
        out = brain.forward(pn, encoder._green_trace, explore=False)
        staged = obs.phase == "staged"
        action = Action(
            throttle=0.0 if staged else out["throttle"],
            steer=out["steer"],
            shift=out["shift"],
        )
        frames.append(_frame(env, obs, action))
        obs, reward, done, info = env.step(action)
        ticks += 1
        if done or ticks > 1500:
            break
    frames.append(_frame(env, obs, action))
    return {
        "driver": "fly",
        "phase": obs.phase,
        "elapsed_time": info.get("elapsed_time"),
        "reaction_time": info.get("reaction_time"),
        "lane_half_width": env.lane_half_width,
        "track_length": env.track_length,
        "redline": env.redline_rpm,
        "top_gear": len(env.gear_ratios),
        "frames": frames,
    }


def record_baseline_episode(env: DragStripEnv, controller, seed: int, env_kwargs=None) -> dict:
    if env_kwargs:
        for k, v in env_kwargs.items():
            setattr(env, k, v)
    obs = env.reset(seed=seed)
    frames = []
    ticks = 0
    while True:
        action = controller.act(obs)
        frames.append(_frame(env, obs, action))
        obs, reward, done, info = env.step(action)
        ticks += 1
        if done or ticks > 1500:
            break
    frames.append(_frame(env, obs, action))
    return {
        "driver": "baseline",
        "phase": obs.phase,
        "elapsed_time": info.get("elapsed_time"),
        "reaction_time": info.get("reaction_time"),
        "lane_half_width": env.lane_half_width,
        "track_length": env.track_length,
        "redline": env.redline_rpm,
        "top_gear": len(env.gear_ratios),
        "frames": frames,
    }
