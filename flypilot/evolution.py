"""Inheritance for the league: new flies are bred from good ones instead of being random restarts.

The node-perturbation plasticity rule never converges (see train.py), so a fly's learned weights are the
only thing worth passing on. An offspring gets a copy of its parent's brain -- weights, KC wiring and the
value function -- with a small mutation on the weights and on the two temperament numbers (eta, noise).
Nothing here touches the environment or the league bookkeeping, so it can be tested on its own.
"""
from __future__ import annotations

import copy
import random

import numpy as np

from .brain import FlyBrain

ETA_RANGE = (0.004, 0.08)
NOISE_RANGE = (0.15, 1.2)


def genome(brain: FlyBrain) -> dict:
    """The numbers that describe how a fly learns -- logged for every pilot so runs can be compared later."""
    return {"eta": round(float(brain.eta), 5), "noise_sigma": round(float(brain.noise_sigma), 4),
            "n_kc": int(brain.n_kc), "sparsity": float(brain.sparsity), "w_max": float(brain.w_max),
            "w_std": round(float(np.std(brain.W[brain.connection_mask > 0])), 5)}


def mutate_brain(parent: FlyBrain, seed: int, strength: float) -> tuple[FlyBrain, dict]:
    """A child of `parent`: same wiring, weights nudged by `strength` x the weights' own spread."""
    rng = np.random.default_rng(seed)
    child = copy.deepcopy(parent)
    live = child.connection_mask > 0
    spread = max(float(np.std(child.W[live])), 0.02)
    child.W = np.clip(child.W + rng.normal(0.0, strength * spread, child.W.shape) * child.connection_mask,
                      -child.w_max, child.w_max)
    child.eta = float(np.clip(child.eta * np.exp(rng.normal(0.0, 0.25)), *ETA_RANGE))
    child.noise_sigma = float(np.clip(child.noise_sigma * np.exp(rng.normal(0.0, 0.2)), *NOISE_RANGE))
    child.rng = np.random.default_rng(seed + 1)      # siblings must not share exploration noise
    child.eligibility[:] = 0.0
    return child, {"w_sigma": round(strength * spread, 5), "parent_eta": round(float(parent.eta), 5),
                   "parent_noise_sigma": round(float(parent.noise_sigma), 4)}


def choose_parent(trained: list[dict], legend: dict | None, rng: random.Random, legend_share: float,
                  min_races: int = 3) -> dict | None:
    """Rank-weighted (3:2:1) among the flies with the lowest held-out loss (see league.eval_loss), falling back
    to Elo for flies that have not been evaluated yet; sometimes the reigning champion itself."""
    if legend is not None and legend_share > 0 and rng.random() < legend_share:
        return legend
    vets = sorted((p for p in trained if p["races"] >= min_races),
                  key=lambda p: (p.get("fit") if p.get("fit") is not None else 99.0, -p["rating"]))[:3]
    if not vets:
        return None
    return rng.choices(vets, weights=[3, 2, 1][:len(vets)])[0]
