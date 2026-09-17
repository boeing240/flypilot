"""Dopamine as reward-prediction error: reward minus a slow running baseline.

Mirrors PAM/PPL1 opponent dopamine neurons signaling better/worse than
expected, rather than raw reward magnitude -- keeps the plasticity signal
small and roughly zero-mean once the controller is doing its average thing,
so learning does not just track absolute reward scale.
"""
from __future__ import annotations


class DopamineTracker:
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
        self.baseline = 0.0

    def reset_baseline(self):
        self.baseline = 0.0

    def step(self, reward: float) -> float:
        dopamine = reward - self.baseline
        self.baseline += self.alpha * (reward - self.baseline)
        return dopamine
