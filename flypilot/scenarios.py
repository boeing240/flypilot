"""Named uncertainty scenarios: engine-power derate, traction loss, and combinations.

None of these were seen during training (trained only at nominal 1.0/1.0) --
the point is to see how the learned controller degrades, not to retrain per
scenario.
"""
from __future__ import annotations

SCENARIOS = {
    "nominal": dict(
        engine_power_scale=1.0,
        traction_scale=1.0,
        traction_noise_sigma=0.0,
    ),
    "hot_engine": dict(
        # sustained high coolant/intake temp pulling timing and derating torque
        engine_power_scale=0.75,
        traction_scale=1.0,
        traction_noise_sigma=0.0,
    ),
    "cold_greasy_track": dict(
        # low track temp / rubber-poor surface: less rear grip everywhere
        engine_power_scale=1.0,
        traction_scale=0.65,
        traction_noise_sigma=0.0,
    ),
    "patchy_grip": dict(
        # oil-down / bleach-box inconsistency: grip varies tick to tick
        engine_power_scale=1.0,
        traction_scale=1.0,
        traction_noise_sigma=0.35,
    ),
    "worst_case": dict(
        # hot engine AND cold patchy track at once
        engine_power_scale=0.75,
        traction_scale=0.65,
        traction_noise_sigma=0.35,
    ),
}
