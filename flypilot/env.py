"""Straight-line drag strip physics: tree light, manual gearbox, wheel slip, curb sensors.

Everything here is [C] engineering approximation, not a certified vehicle model.
Time step is fixed (50 Hz) to mirror a typical ECU/sensor polling loop.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass
class Action:
    throttle: float  # commanded 0..1, agent's request
    steer: float  # -1 (push left) .. +1 (push right)
    shift: bool  # request an upshift this tick


@dataclass
class Obs:
    phase: str  # 'staged' | 'green' | 'racing' | 'finished' | 'foul' | 'crash'
    green_onset: bool
    rpm: float
    redline: float
    gear: int
    top_gear: int
    throttle_fb: float
    speed: float
    wheel_slip: float
    dist_left: float
    dist_right: float
    lane_half_width: float
    dist_to_finish: float
    track_length: float


@dataclass
class DragStripEnv:
    dt: float = 0.02
    track_length: float = 201.0  # eighth mile, keeps episodes short for training
    lane_half_width: float = 1.2
    gear_ratios: tuple = (3.827, 2.360, 1.685, 1.312, 1.000, 0.793)
    final_drive: float = 3.90
    wheel_radius: float = 0.32
    idle_rpm: float = 900.0
    redline_rpm: float = 7000.0
    peak_torque: float = 450.0
    mass: float = 1400.0
    drag_area: float = 0.7  # Cd*A
    rolling_crr: float = 0.015
    mu: float = 1.1
    rear_weight_frac: float = 0.55
    shift_duration: float = 0.25
    shift_cooldown_time: float = 0.15
    foul_throttle_threshold: float = 0.08
    safety_margin: float = 0.35  # m from a curb counted as "near miss"
    ou_theta: float = 1.2
    ou_sigma: float = 0.9
    disturbance_gain: float = 0.35
    steer_gain: float = 2.2
    lateral_damping: float = 1.5
    slip_pull_gain: float = 1.0  # how hard wheelspin yanks the car sideways

    # --- uncertainty / scenario knobs, all 1.0 = nominal ---
    engine_power_scale: float = 1.0  # e.g. 0.8 = hot engine pulling timing, down on torque
    traction_scale: float = 1.0  # e.g. 0.7 = colder/greasier track, less rear grip
    traction_noise_sigma: float = 0.0  # per-tick multiplicative traction jitter (patchy grip)

    def reset(self, seed: int | None = None) -> Obs:
        self.rng = random.Random(seed)
        self.t = 0.0
        self.t_since_green = None
        self.green_delay = self.rng.uniform(0.6, 1.5)
        self.phase = "staged"
        self.x = 0.0
        self.v = 0.0
        self.gear = 1
        self.rpm = self.idle_rpm
        self.throttle_fb = 0.0
        self.wheel_slip = 0.0
        self.shift_state = "engaged"
        self.shift_timer = 0.0
        self.pending_gear = None
        self.shift_cooldown = 0.0
        self.y = 0.0
        self.lateral_vel = 0.0
        self.disturbance = 0.0
        # which side wheelspin pulls the car toward, and how hard -- real cars
        # don't spin up perfectly symmetrically (uneven weight transfer, one
        # tire with marginally more grip, engine torque reaction through the
        # driveline). Fixed per run, like a real asymmetry would be.
        self.slip_pull_bias = self.rng.uniform(-1.0, 1.0)
        self.reaction_time = None
        self.green_onset_flag = False
        return self._obs()

    def _obs(self) -> Obs:
        dist_right = self.lane_half_width - self.y
        dist_left = self.lane_half_width + self.y
        return Obs(
            phase=self.phase,
            green_onset=self.green_onset_flag,
            rpm=self.rpm,
            redline=self.redline_rpm,
            gear=self.gear,
            top_gear=len(self.gear_ratios),
            throttle_fb=self.throttle_fb,
            speed=self.v,
            wheel_slip=self.wheel_slip,
            dist_left=dist_left,
            dist_right=dist_right,
            lane_half_width=self.lane_half_width,
            dist_to_finish=max(0.0, self.track_length - self.x),
            track_length=self.track_length,
        )

    def _engine_torque(self, rpm: float) -> float:
        # torque at idle is not zero (engines can pull from a stop); it rises to a
        # peak mid-range and tapers, but the hard rev limiter is what actually
        # kills torque above redline -- not the shape of this curve.
        if rpm >= self.redline_rpm:
            return 0.0
        span = self.redline_rpm - self.idle_rpm
        frac = min(max((rpm - self.idle_rpm) / span, 0.0), 1.0)
        return self.peak_torque * (0.35 + 0.65 * math.sin(math.pi * frac))

    def step(self, action: Action):
        self.green_onset_flag = False
        reward = 0.0
        reward_lon = 0.0  # throttle/shift-relevant: progress, slip, finish
        reward_lat = 0.0  # steer-relevant: centering, near-miss, crash
        done = False
        info = {}

        if self.phase == "staged":
            if action.throttle > self.foul_throttle_threshold:
                self.phase = "foul"
                reward -= 50.0
                reward_lon -= 50.0
                done = True
                info["reward_lon"] = reward_lon
                info["reward_lat"] = reward_lat
                return self._obs(), reward, done, info
            self.t += self.dt
            if self.t >= self.green_delay:
                self.phase = "green"
                self.t_since_green = 0.0
                self.green_onset_flag = True
            info["reward_lon"] = reward_lon
            info["reward_lat"] = reward_lat
            return self._obs(), reward, done, info

        # phases: green / racing -- clock and distance run from here
        self.t_since_green += self.dt
        self.throttle_fb = max(0.0, min(1.0, action.throttle))
        if self.reaction_time is None and self.throttle_fb > 0.15:
            self.reaction_time = self.t_since_green

        # --- longitudinal dynamics ---
        if self.shift_state == "shifting":
            self.shift_timer -= self.dt
            accel = -(self._aero_drag() + self._rolling_resistance()) / self.mass
            self.wheel_slip = 0.0
            if self.shift_timer <= 0.0:
                self.gear = self.pending_gear
                self.shift_state = "engaged"
                self.shift_cooldown = self.shift_cooldown_time
                self.rpm = self._rpm_from_speed(self.v, self.gear)
        else:
            torque = self._engine_torque(self.rpm) * self.engine_power_scale
            ratio = self.gear_ratios[self.gear - 1] * self.final_drive
            wheel_force = torque * self.throttle_fb * ratio * 0.9 / self.wheel_radius
            traction_jitter = 1.0
            if self.traction_noise_sigma > 0.0:
                traction_jitter = max(0.3, 1.0 + self.rng.gauss(0, self.traction_noise_sigma))
            max_traction = self.mu * self.traction_scale * traction_jitter * self.mass * 9.81 * self.rear_weight_frac
            if wheel_force <= max_traction:
                self.wheel_slip = 0.0
                accel = (wheel_force - self._aero_drag() - self._rolling_resistance()) / self.mass
                self.v = max(0.0, self.v + accel * self.dt)
                self.rpm = self._rpm_from_speed(self.v, self.gear)
            else:
                self.wheel_slip = min(1.0, (wheel_force - max_traction) / max_traction)
                eff_force = max_traction * 0.7  # kinetic grip < static while spinning
                accel = (eff_force - self._aero_drag() - self._rolling_resistance()) / self.mass
                self.v = max(0.0, self.v + accel * self.dt)
                self.rpm = min(self.redline_rpm, self.rpm + self.wheel_slip * 8000.0 * self.dt)

            self.shift_cooldown = max(0.0, self.shift_cooldown - self.dt)
            if (
                action.shift
                and self.gear < len(self.gear_ratios)
                and self.shift_cooldown <= 0.0
            ):
                self.shift_state = "shifting"
                self.shift_timer = self.shift_duration
                self.pending_gear = self.gear + 1
                # dense, immediate credit for the *act* of shifting -- without this,
                # "redline forever in 1st gear, ride the wheelspin plateau" is a
                # locally-competitive equilibrium the over-rev penalty alone doesn't
                # beat, since that penalty is small per tick and shifting carries a
                # short power-interruption cost of its own.
                # only reward a well-timed shift; no penalty for a premature one --
                # that penalty, stacked with a shift's zero-slip grace period, made
                # "shift immediately through every gear at a standstill" a way to
                # dodge the slip penalty entirely. Let a bad shift's own consequence
                # (a torque-starved car in too tall a gear) be the only cost.
                rpm_at_shift = self.rpm / self.redline_rpm
                if 0.60 <= rpm_at_shift <= 0.97:
                    # 4.0 was tried and destabilized training across an entire
                    # population (every individual collapsed to 0% finishes) --
                    # the reward-scale spike was too large for this learning rule.
                    # Back to a milder, empirically-stable value.
                    reward_lon += 1.2

        self.x += self.v * self.dt

        # --- lateral dynamics ---
        self.disturbance += (-self.ou_theta * self.disturbance + self.ou_sigma * self.rng.gauss(0, 1)) * self.dt
        # steer > 0 must push y up (toward the right curb) to match the baseline's and
        # the brain's convention (steer command has the same sign as "increase y").
        # wheelspin risk isn't just lost time -- a spinning wheel breaks traction
        # unevenly and pulls the car off line, harder the more it's slipping.
        slip_pull = self.wheel_slip * self.slip_pull_bias * self.slip_pull_gain
        lateral_accel = self.disturbance * self.disturbance_gain + slip_pull + action.steer * self.steer_gain
        self.lateral_vel += lateral_accel * self.dt
        self.lateral_vel *= max(0.0, 1.0 - self.lateral_damping * self.dt)
        self.y += self.lateral_vel * self.dt

        dist_right = self.lane_half_width - self.y
        dist_left = self.lane_half_width + self.y
        curb_min = min(dist_left, dist_right)
        # dense every-tick centering signal -- without it, steering only ever gets
        # blamed at the crash tick itself, which is too sparse for single-tick
        # credit assignment to connect to the steering choices that caused it.
        # This is also the fly's own strategy: hold the corridor center via
        # continuous optic-flow balance, not react only when a wall is imminent.
        reward_lat -= 1.5 * (self.y / self.lane_half_width) ** 2 * self.dt
        # small steering-effort cost -- without it, yanking the wheel back and forth
        # is free as long as y recovers, so nothing discourages needless swerving.
        reward_lat -= 0.05 * action.steer ** 2 * self.dt
        if curb_min < self.safety_margin:
            reward_lat -= 2.0 * (self.safety_margin - curb_min) / self.safety_margin * self.dt
        if curb_min <= 0.0:
            self.phase = "crash"
            reward_lat -= 50.0
            reward = reward_lon + reward_lat
            done = True
            info["reward_lon"] = reward_lon
            info["reward_lat"] = reward_lat
            return self._obs(), reward, done, info

        reward_lon += 1.0 * self.v * self.dt
        reward_lon -= 0.8 * self.wheel_slip * self.dt
        # dense "stay in the productive rev band" signal -- without it, the only
        # throttle/shift feedback is slip penalty (immediate) vs finish bonus
        # (delayed, single-tick eligibility can't reach it), and the network
        # settles on "never risk revving" instead of "get there fast".
        band_lo, band_hi = 0.45, 0.90  # fraction of redline
        rpm_frac = self.rpm / self.redline_rpm
        if band_lo <= rpm_frac <= band_hi:
            reward_lon += 0.6 * self.dt
        elif rpm_frac > band_hi:
            # a mild nudge, not the main deterrent -- the slip_pull physics above
            # is what actually makes "redline in 1st gear forever" dangerous now
            # (real crash risk), rather than trying to out-tune progress reward
            # with an ever-larger penalty coefficient.
            reward_lon -= 1.0 * (rpm_frac - band_hi) * self.dt

        if self.x >= self.track_length:
            self.phase = "finished"
            reward_lon += 20.0 - self.t_since_green
            info["elapsed_time"] = self.t_since_green
            info["reaction_time"] = self.reaction_time
            done = True
        else:
            self.phase = "racing"

        reward = reward_lon + reward_lat
        info["reward_lon"] = reward_lon
        info["reward_lat"] = reward_lat
        return self._obs(), reward, done, info

    def _aero_drag(self) -> float:
        rho = 1.2
        return 0.5 * rho * self.drag_area * self.v * self.v

    def _rolling_resistance(self) -> float:
        return self.rolling_crr * self.mass * 9.81

    def _rpm_from_speed(self, v: float, gear: int) -> float:
        ratio = self.gear_ratios[gear - 1] * self.final_drive
        rpm = v / self.wheel_radius * ratio * 60.0 / (2 * math.pi)
        return min(max(rpm, self.idle_rpm), self.redline_rpm)
