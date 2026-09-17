"""Sensor -> projection-neuron encoding. ON/OFF split per scalar channel, as in olfactory PNs."""
from __future__ import annotations

import numpy as np

from .env import Obs

FEATURE_NAMES = [
    "rpm_ratio",
    "rpm_rate",
    "throttle_fb",
    "speed",
    "wheel_slip",
    "gear",
    "progress",
    "curb_error",
    "curb_min",
    "staged_idle",
]
N_FEATURES = len(FEATURE_NAMES)
N_PN = N_FEATURES * 2 + 1  # ON/OFF per feature, plus raw green-onset transient


class SenseEncoder:
    def __init__(self, speed_ref: float = 45.0):
        self.speed_ref = speed_ref
        self._prev_rpm_ratio = 0.0
        self._green_trace = 0.0

    def reset(self):
        self._prev_rpm_ratio = 0.0
        self._green_trace = 0.0

    def encode(self, obs: Obs) -> np.ndarray:
        rpm_ratio = obs.rpm / obs.redline - 0.5
        rpm_rate = np.clip((rpm_ratio - self._prev_rpm_ratio) * 20.0, -1.0, 1.0)
        self._prev_rpm_ratio = rpm_ratio

        curb_error = np.clip((obs.dist_right - obs.dist_left) / obs.lane_half_width, -1.0, 1.0)
        curb_min = np.clip(min(obs.dist_left, obs.dist_right) / obs.lane_half_width, 0.0, 1.0) - 0.5

        values = np.array(
            [
                np.clip(rpm_ratio, -1.0, 1.0),
                rpm_rate,
                obs.throttle_fb - 0.5,
                np.clip(obs.speed / self.speed_ref, 0.0, 1.0) - 0.5,
                np.clip(obs.wheel_slip, 0.0, 1.0) - 0.1,
                obs.gear / obs.top_gear - 0.5,
                np.clip(1.0 - obs.dist_to_finish / obs.track_length, 0.0, 1.0) - 0.5,
                curb_error,
                curb_min,
                1.0 if obs.phase == "staged" else -0.5,
            ],
            dtype=np.float32,
        )

        on = np.clip(values, 0.0, None)
        off = np.clip(-values, 0.0, None)

        self._green_trace = self._green_trace * 0.6
        if obs.green_onset:
            self._green_trace = 1.0
        green_channel = np.array([self._green_trace], dtype=np.float32)

        return np.concatenate([on, off, green_channel])
