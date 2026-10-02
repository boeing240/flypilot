from __future__ import annotations

import copy
import random
import statistics as stats

from .brain import FlyBrain
from .env import Action, DragStripEnv
from .learning import DopamineTracker, ValueDopamineTracker
from .sense import N_PN, SenseEncoder


def fitness_key(summary):
    # lexicographic: finish rate first, then crash rate, then speed -- a fast
    # individual that crashes is worse than a slow one that always finishes.
    finish_rate = summary["finished"] / summary["n"]
    crash_rate = summary["crashes"] / summary["n"]
    avg_time = summary["avg_time"] if summary["avg_time"] is not None else 1e9
    return (-finish_rate, crash_rate, avg_time)


def evaluate(brain: FlyBrain, env: DragStripEnv, encoder: SenseEncoder,
             n_episodes: int, seed0: int) -> dict:
    dlon, dlat = ValueDopamineTracker(N_PN), DopamineTracker()
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


def run_fly_episode(env: DragStripEnv, brain: FlyBrain, encoder: SenseEncoder,
                     dopamine_lon: ValueDopamineTracker, dopamine_lat: DopamineTracker,
                     seed: int, learn: bool = True, on_tick=None) -> dict:
    # on_tick(env, ticks), if given, runs after every env.step -- used to record
    # a trajectory for replay without duplicating this loop.
    obs = env.reset(seed=seed)
    encoder.reset()
    brain.reset()
    total_reward = 0.0
    ticks = 0
    # TD(0) needs the *next* state to bootstrap from, which isn't available
    # until the following iteration -- so each transition (pn, reward) is
    # held in `pending` and closed out one tick later, using this tick's pn
    # as the next-state for it. That close-out must happen *before* this
    # tick's forward() call, since forward() overwrites brain.eligibility
    # (the trace the dopamine signal needs to gate) with a fresh one for the
    # new action.
    pending = None
    while True:
        pn = encoder.encode(obs)
        if learn and pending is not None:
            prev_pn, prev_r_lon, prev_r_lat = pending
            d_lon = dopamine_lon.step(prev_r_lon, prev_pn, pn)
            d_lat = dopamine_lat.step(prev_r_lat, prev_pn, pn)
            brain.apply_dopamine(d_lon, d_lat)
        out = brain.forward(pn, encoder._green_trace, explore=learn)
        # not-before-green is a hard safety gate, not something worth spending
        # learning capacity (and risking foul-start crashes) on -- the reflex
        # pathway still gets to prove itself the instant the light turns green.
        staged = obs.phase == "staged"
        throttle = 0.0 if staged else out["throttle"]
        action = Action(throttle=throttle, steer=out["steer"], shift=out["shift"])
        obs, reward, done, info = env.step(action)
        total_reward += reward
        pending = (pn, info["reward_lon"], info["reward_lat"]) if not staged else None
        ticks += 1
        if on_tick is not None:
            on_tick(env, ticks)
        if done or ticks > 1500:  # 30s of race clock -- generous, real runs finish well under 15s
            if learn and pending is not None:
                prev_pn, prev_r_lon, prev_r_lat = pending
                d_lon = dopamine_lon.step(prev_r_lon, prev_pn, None)  # terminal: no bootstrap
                d_lat = dopamine_lat.step(prev_r_lat, prev_pn, None)
                brain.apply_dopamine(d_lon, d_lat)
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


# Ranges sampled per training episode when domain randomization is on. They
# deliberately span the named scenarios in scenarios.py (traction 0.65,
# engine 0.75, jitter 0.35), so those scenarios stop being out-of-distribution
# once randomization is used.
CONDITION_RANGES = {
    "traction_scale": (0.6, 1.0),
    "engine_power_scale": (0.7, 1.0),
    "traction_noise_sigma": (0.0, 0.4),
}
NOMINAL_CONDITIONS = {"traction_scale": 1.0, "engine_power_scale": 1.0, "traction_noise_sigma": 0.0}


def set_conditions(env: DragStripEnv, conditions: dict) -> None:
    for name, value in conditions.items():
        setattr(env, name, value)


def train(n_episodes: int = 3000, report_every: int = 200, seed0: int = 0,
          brain_kwargs: dict | None = None, quiet: bool = False,
          randomize_prob: float = 0.0) -> FlyBrain:
    brain = FlyBrain(n_pn=N_PN, seed=seed0, **(brain_kwargs or {}))
    return continue_train(brain, n_episodes, report_every=report_every, seed0=seed0,
                          quiet=quiet, randomize_prob=randomize_prob)


def continue_train(brain: FlyBrain, n_episodes: int, report_every: int = 200,
                    seed0: int = 0, quiet: bool = False,
                    checkpoint_every: int | None = None,
                    checkpoint_episodes: int = 30,
                    checkpoint_seed0: int = 500_000,
                    randomize_prob: float = 0.0) -> FlyBrain:
    """Keeps training an already-initialized brain (e.g. a clone of a winner
    from a previous selection round) instead of starting from scratch.

    The node-perturbation plasticity rule never converges -- it keeps making
    noisy updates for as long as training runs, so continuing to train an
    already-good brain is as likely to wander away from that optimum as to
    improve it. If `checkpoint_every` is set, the brain is periodically
    frozen and evaluated on a small held-out set, and the *best* checkpoint
    seen is returned instead of just whatever the final episode left behind.

    With `randomize_prob` > 0, that fraction of training episodes runs under
    randomly drawn traction/engine conditions (CONDITION_RANGES) instead of
    nominal, so robustness to them is trained for rather than found by luck.
    """
    env = DragStripEnv()
    encoder = SenseEncoder()
    dopamine_lon = ValueDopamineTracker(N_PN)
    dopamine_lat = DopamineTracker()
    condition_rng = random.Random(seed0)

    best_brain = None
    best_fitness = None

    def maybe_checkpoint():
        nonlocal best_brain, best_fitness
        if checkpoint_every is None:
            return
        # checkpoints are scored at nominal, not at whatever the last training
        # episode happened to randomize the shared env to
        set_conditions(env, NOMINAL_CONDITIONS)
        summary = evaluate(brain, env, encoder, checkpoint_episodes, seed0=checkpoint_seed0)
        fit = fitness_key(summary)
        if best_fitness is None or fit < best_fitness:
            best_fitness = fit
            best_brain = copy.deepcopy(brain)

    window = []
    for ep in range(n_episodes):
        if randomize_prob > 0.0 and condition_rng.random() < randomize_prob:
            set_conditions(env, {n: condition_rng.uniform(lo, hi)
                                 for n, (lo, hi) in CONDITION_RANGES.items()})
        else:
            set_conditions(env, NOMINAL_CONDITIONS)
        result = run_fly_episode(
            env, brain, encoder, dopamine_lon, dopamine_lat, seed=seed0 + ep, learn=True
        )
        window.append(result)
        if checkpoint_every and (ep + 1) % checkpoint_every == 0:
            maybe_checkpoint()
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

    if checkpoint_every is None:
        return brain
    maybe_checkpoint()  # always check the final state too
    return best_brain


if __name__ == "__main__":
    train()
