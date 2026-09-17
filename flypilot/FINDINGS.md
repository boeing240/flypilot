# Findings log

## 2026-09-17 — MVP built, three bugs found and fixed, one architectural limit found

**Setup.** Straight-line drag strip simulator (eighth mile, 50 Hz), Christmas-tree
light, manual gearbox with wheel slip, curb distance sensors with OU lateral
disturbance. Mushroom-body-inspired controller: PN -> sparse random KC projection
(600 KC, 6 claws, ~10% active) -> plastic KC->MBON pools (go/lift for throttle,
steer_pos/steer_neg for steering, shift). Reflex pathway bypasses the KC layer
for the green-light launch impulse (giant-fiber analog). Plasticity is node
perturbation (trial noise on each pool's output, correlated with a dopamine
signal = reward minus a running baseline).

**Bug 1 — sigmoid throttle floor.** `sigmoid(0) = 0.5`, so with no drive at all
the untrained network already commanded ~50% throttle before the light went
green: 100% false-start rate. Fixed by switching to `max(0, tanh(...))`, which
defaults to 0.

**Bug 2 — torque curve zero at idle.** The engine model made zero torque at idle
RPM by construction, so the car could never move from a stop regardless of
throttle. Fixed the curve to keep a torque floor (idle engines do pull).

**Bug 3 (the interesting one) — no exploration noise, so the dopamine-gated
Hebbian rule had nothing to correlate with.** A deterministic forward pass
gives reward-times-correlation nothing to learn from; push/pull pools like
go/lift are symmetric in raw co-activation, so naive correlation-based updates
just oscillate. Fixed with node perturbation: jitter each pool's output by
trial noise, correlate *that* noise with the dopamine signal. This is a real
proposed mechanism (MB response variability exploited by dopamine gating), not
just a training trick.

**Bug 3b — multi-tick eligibility decay added variance, not signal.** Smoothing
the eligibility trace across ticks correlated *this* tick's dopamine with
*previous* ticks' independent noise draws, which is unrelated by construction.
Fixed by using a single-tick trace for immediate per-tick reward.

**Physics bug — inverted steering sign.** `action.steer` was subtracted where it
needed to be added, so a correctly-computed corrective steering command
*accelerated* the drift instead of countering it. This alone caused a 98.5%
crash rate for a hand-tuned rule-based baseline controller. Fixed; baseline now
finishes 300/300 at ~10.1s.

**Compartmentalized dopamine.** A single scalar dopamine signal gating all five
pools let steering's reward variance swamp the throttle-relevant signal (and
vice versa), since the two are only weakly related but share the same
broadcast. Split into two streams -- longitudinal (go/lift/shift, gated by
progress/slip/finish reward) and lateral (steer, gated by centering/near-miss/
crash reward) -- mirroring the fly MB's real compartmentalization (different
DAN clusters carry different reinforcement types into different compartments).
This is not just an engineering fix; it is closer to the actual biology than
the single-dopamine version was.

**Current result.** After the fixes: steering is solid (0 crashes over 300
frozen-policy evaluation episodes, holds the lane center). Throttle/shift is
not: the frozen (noise-off) policy settles on ~5-10% throttle for the whole
run and never upshifts out of first gear, finishing 0/300 within the 30s cap.

**Why.** The plasticity rule is a single-tick, reward-times-perturbation update
-- it has no value function or multi-step return, so it is effectively solving
a myopic (per-tick) reward-maximization problem, not the episode's actual
objective. Wheel-slip penalty is immediate and large; the payoff for getting to
the finish line faster is delayed and only visible in the (much larger)
terminal bonus. Per-tick credit assignment systematically favors "never risk
slip" over "get there fast," even though the latter wins over a whole episode.
This is the throttle/shift analog of what steering needed a dense per-tick
signal for -- except here the myopia is about the *time horizon* of the
signal, not its presence.

## 2026-09-17 — RPM-band shaping fixes throttle/shift; robustness test finds a real failure mode

**Fix.** Added a small dense reward for keeping RPM in a productive band
(0.45-0.90 of redline) and a penalty for over-revving past it, on top of
reducing the slip penalty weight. This gives the longitudinal dopamine stream
the same kind of immediate, informative signal the lateral stream already had
from the centering reward.

**Result.** Frozen (noise-off) policy: 300/300 finishes, 0 crashes, avg 10.02s
-- matching the hand-tuned baseline's 10.14s. Confirms the diagnosis: it was a
credit-assignment gap, not a capacity problem.

**Robustness test (no retraining, five scenarios, 150 episodes each,
`flypilot/scenarios.py` / `evaluate_scenarios.py`):**

| scenario | fly finish | fly crash | baseline finish | baseline crash |
|---|---|---|---|---|
| nominal | 150/150 | 0 | 150/150 | 0 |
| hot_engine (0.75x torque) | 150/150 | 0 | 150/150 | 0 |
| cold_greasy_track (0.65x traction) | **0/150** | **150** | 150/150 | 0 |
| patchy_grip (traction jitter) | 44/150 | 106 | 150/150 | 0 |
| worst_case (both) | **0/150** | **150** | 150/150 | 0 |

**The fly generalizes to engine derate for free, and fails completely on
traction loss it never trained on -- the baseline (a simple rule) handles
every scenario fine, just slower.** This is the interesting result, not a
disappointing one: engine power only ever entered training through the
longitudinal reward stream (progress/RPM-band/slip), and reduced power just
means "everything scales down a bit," well inside what the network already
generalizes over throttle levels. Traction loss changes the *relationship*
between throttle and wheel-slip in a way the network never experienced, *and*
the crashes are lateral (steering), not longitudinal -- despite traction
having no direct physical effect on steering.

**Suspected cause.** KC wiring is a random projection over the *whole* PN
vector -- nothing constrains "slip-relevant" PN channels to only reach the
longitudinal MBON pools. An out-of-distribution slip signal can activate KCs
that happen to also feed steer_pos/steer_neg, producing steering commands
that have nothing to do with the actual curb distance. This would be a
genuine, checkable hypothesis (compare KC overlap between high-slip and
normal-slip activity patterns) rather than a guess -- worth testing before
trusting any fix for it.

**Next candidates (not yet tried):**
- A TD-style value baseline per state (not just a global running-reward
  baseline) so dopamine reflects "better than expected from here," which
  naturally credits actions that trade a small immediate cost for a larger
  future gain.
- A genuine multi-tick eligibility trace, done correctly this time: decay the
  trace but keep dopamine and noise properly time-aligned (e.g. accumulate
  trace but only apply it once per meaningful event, not every tick against
  a moving target).
- Simplest fix worth trying first: shape a small dense reward directly for
  RPM staying in a "useful" band and for making a shift when the powerband is
  exceeded, so throttle/shift gets the same kind of immediate, informative
  signal steering already has -- sidestepping the credit-assignment problem
  rather than solving it in general.
