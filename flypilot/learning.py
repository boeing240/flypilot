"""Two dopamine baselines, one per compartment (see brain.py's POOL_GROUP
split), because the two streams don't have the same problem.

The lateral (steering) stream already has a dense, easy-to-learn per-tick
centering reward and no delayed-credit issue -- FINDINGS.md's own numbers
show it gets 0 crashes with nothing more than reward-minus-a-slow-scalar-
average. The longitudinal (throttle/shift) stream is the one FINDINGS.md
diagnosed as myopic: a per-tick slip penalty is immediate, but the payoff of
finishing sooner only appears in the terminal bonus, so a same-tick-only
baseline can't credit an action that trades a small immediate cost for a
larger future gain.

Empirically, giving *both* streams the more sophisticated TD(0) baseline
made steering dramatically worse (100% crash rates across every genome in
testing) -- a linear value function that's still unreliable early in
training injects noise on every tick, and the lateral stream had nothing
to gain from fixing a delayed-credit problem it never had. So: keep the
simple scalar baseline for lateral, and use TD(0) only where the actual
diagnosed problem is.
"""
from __future__ import annotations

import numpy as np


class DopamineTracker:
    """reward minus a slow running average -- same baseline for every state.
    Used for the lateral (steering) stream; see module docstring for why."""

    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
        self.baseline = 0.0

    def reset_baseline(self):
        self.baseline = 0.0

    def step(self, reward: float, features=None, next_features=None) -> float:
        # features/next_features accepted (and ignored) so callers can treat
        # both tracker types uniformly.
        dopamine = reward - self.baseline
        self.baseline += self.alpha * (reward - self.baseline)
        return dopamine


class ValueDopamineTracker:
    """TD(0): reward plus the discounted value of the next state, minus the
    value of the current one, with V a linear function of the sensory (PN)
    state learned online. Used for the longitudinal stream: bootstrapping is
    what lets the terminal finish bonus propagate backward into earlier
    states over the course of training, instead of only ever being visible
    on the one tick it's paid out.
    """

    def __init__(self, n_features: int, alpha_v: float = 0.005, gamma: float = 0.98,
                 delta_clip: float = 10.0):
        self.w = np.zeros(n_features, dtype=np.float64)
        self.alpha_v = alpha_v
        self.gamma = gamma
        # Rewards include rare large terminal bonuses/penalties (crash -50,
        # finish up to +20ish) alongside tiny per-tick shaping terms. Early
        # on, V is zero everywhere and hasn't learned to tell those apart, so
        # an unclipped TD error can spike to the terminal reward's scale on
        # *any* tick while V is still catching up. Clipping bounds the
        # damage that does to both the brain's plasticity update and V's own
        # bootstrapped update.
        self.delta_clip = delta_clip

    def reset_baseline(self):
        self.w[:] = 0.0

    def value(self, features: np.ndarray) -> float:
        return float(self.w @ features)

    def step(self, reward: float, features: np.ndarray, next_features: np.ndarray | None) -> float:
        """Closes out the transition that started at `features`: computes the
        TD error against `next_features` (None for a terminal transition,
        i.e. no bootstrap), updates V(features) towards that target, and
        returns the TD error to use as the dopamine signal.
        """
        v_next = self.value(next_features) if next_features is not None else 0.0
        target = reward + self.gamma * v_next
        delta = np.clip(target - self.value(features), -self.delta_clip, self.delta_clip)
        self.w += self.alpha_v * delta * features
        return delta
