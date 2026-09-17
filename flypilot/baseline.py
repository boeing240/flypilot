"""Simple rule-based driver: fixed launch ramp, RPM-threshold shifts, proportional steer.

Not learned -- a normal engineering baseline to judge whether the fly circuit
is doing anything the obvious hand-tuned controller doesn't already do.
"""
from __future__ import annotations

from .env import Action, Obs


class BaselineController:
    def __init__(self, shift_rpm: float = 6300.0, steer_kp: float = 0.6):
        self.shift_rpm = shift_rpm
        self.steer_kp = steer_kp

    def act(self, obs: Obs) -> Action:
        if obs.phase == "staged":
            return Action(throttle=0.0, steer=0.0, shift=False)

        throttle = 0.55 if obs.wheel_slip > 0.05 else 1.0
        shift = obs.rpm >= self.shift_rpm and obs.gear < obs.top_gear
        error = obs.dist_right - obs.dist_left  # >0 means drifting toward right curb
        steer = max(-1.0, min(1.0, self.steer_kp * error / obs.lane_half_width))
        return Action(throttle=throttle, steer=steer, shift=shift)
