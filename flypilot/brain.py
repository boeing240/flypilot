"""Mushroom-body-inspired controller.

Two pathways, mirroring the fly's own split between a fast non-KC reflex arc
(giant fiber -> escape) and the slow, learned mushroom-body circuit:

- reflex: green-onset transient drives throttle directly, bypassing the KC
  layer entirely, so the first "floor it" impulse is not gated by learning.
- learned: PN -> sparse random KC projection -> APL-style top-k inhibition
  -> plastic KC->MBON pools. This is what learns to back off throttle when
  the wheel-slip channel is hot, hold a steering line, and time shifts.

Plasticity is node perturbation: each pool's raw output is jittered by a
small trial-to-trial noise term, and the eligibility trace correlates that
jitter with presynaptic KC activity. Dopamine then reinforces whichever
jitter direction preceded a better-than-expected outcome. This needs the
noise -- a deterministic forward pass gives a reward-times-Hebbian rule
nothing to correlate with (push/pull pool pairs like go/lift just oscillate,
since raw co-activation is symmetric in both). Trial-to-trial MB response
variability exploited by dopamine gating is a real proposed mechanism, not
just a training trick; it is the same idea as weight/node perturbation in
birdsong learning. [B]/[C] simplification of KC->MBON plasticity, not a
claim about measured synapses.
"""
from __future__ import annotations

import numpy as np

POOLS = ("go", "lift", "steer_pos", "steer_neg", "shift")
# which dopamine stream (see learning.py) gates each pool -- mirrors the fly MB's
# compartmentalized DAN input, where different compartments get reward info about
# different things rather than one global broadcast. Without this split, steering's
# reward variance drowns the throttle signal and vice versa (see brain module docstring).
POOL_GROUP = np.array([0, 0, 1, 1, 0])  # 0 = longitudinal (go/lift/shift), 1 = lateral (steer)


class FlyBrain:
    def __init__(
        self,
        n_pn: int,
        n_kc: int = 600,
        n_claws: int = 6,
        sparsity: float = 0.10,
        eta: float = 0.02,
        noise_sigma: float = 0.5,
        seed: int = 0,
    ):
        self.rng = np.random.default_rng(seed)
        self.n_pn = n_pn
        self.n_kc = n_kc
        self.sparsity = sparsity
        self.claw_idx = self.rng.integers(0, n_pn, size=(n_kc, n_claws))
        self.claw_w = self.rng.uniform(0.5, 1.5, size=(n_kc, n_claws))
        self.W = self.rng.normal(0.0, 0.05, size=(len(POOLS), n_kc))
        self.eligibility = np.zeros_like(self.W)
        self.eta = eta
        self.noise_sigma = noise_sigma
        self.w_max = 3.0
        self.reflex_gain = 3.0
        self.shift_threshold = 0.5
        self._last_kc = np.zeros(n_kc)

    def reset(self):
        self.eligibility[:] = 0.0
        self._last_kc[:] = 0.0

    def forward(self, pn: np.ndarray, green_trace: float, explore: bool = True) -> dict:
        gathered = pn[self.claw_idx]  # (n_kc, n_claws)
        raw = np.maximum(np.sum(gathered * self.claw_w, axis=1), 0.0)

        k = max(1, int(self.n_kc * self.sparsity))
        if np.count_nonzero(raw) > k:
            thresh = np.partition(raw, -k)[-k]
            kc = np.where(raw >= thresh, raw, 0.0)
        else:
            kc = raw

        pool_raw = self.W @ kc
        noise = self.rng.normal(0.0, self.noise_sigma, size=pool_raw.shape) if explore else 0.0
        pool_used = pool_raw + noise
        go, lift, steer_pos, steer_neg, shift = pool_used

        # tanh (not sigmoid) so throttle defaults to 0 with no drive, instead of 0.5 --
        # a sigmoid baseline would floor-it before the light even goes green.
        throttle = max(0.0, float(np.tanh(self.reflex_gain * green_trace + go - lift)))
        steer = float(np.tanh(steer_pos - steer_neg))
        shift_flag = bool(shift > self.shift_threshold)

        if explore:
            # single-tick trace only: correlating this tick's dopamine with a
            # *previous* tick's noise sample is pure variance, not signal, since
            # each forward() call draws an independent perturbation.
            self.eligibility = np.outer(noise, kc)
        self._last_kc = kc

        return {"throttle": float(throttle), "steer": steer, "shift": shift_flag}

    def apply_dopamine(self, dopamine_lon: float, dopamine_lat: float):
        dopamine = np.where(POOL_GROUP == 0, dopamine_lon, dopamine_lat)
        self.W += self.eta * dopamine[:, None] * self.eligibility
        np.clip(self.W, -self.w_max, self.w_max, out=self.W)
        self.W *= 0.999  # slow decay keeps weights bounded without hard rescaling
