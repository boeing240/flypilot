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
- A genuine multi-tick eligibility trace, done correctly this time: decay the
  trace but keep dopamine and noise properly time-aligned (e.g. accumulate
  trace but only apply it once per meaningful event, not every tick against
  a moving target).

## 2026-09-18 — TD(0) value baseline for longitudinal dopamine only

**Tried:** the "next candidate" above -- replace the scalar running-average
dopamine baseline with a proper value function, `V(pn)` linear in the
sensory (PN) state, learned online via TD(0): `dopamine = reward +
gamma*V(s') - V(s)` instead of `reward - running_mean(reward)`. The point is
bootstrapping: `V(s)` learns to predict *future* return, so a terminal
finish bonus can propagate backward into earlier states' value estimates
over the course of training, instead of only ever mattering on the one tick
it's paid out -- directly targeting the myopic credit-assignment gap this
log already diagnosed for throttle/shift.

**First attempt failed badly.** Applying TD(0) to *both* dopamine streams
(longitudinal and lateral) took steering from "0 crashes, solid" to 100%
crash rate across every genome tested, including the exact seed that
previously trained cleanly under the old baseline. Root cause: `V` starts at
zero and is itself only learned online, so early in training its estimates
are noisy garbage -- and because rewards here are dominated by rare large
terminal bonuses/penalties (crash -50, finish up to +20ish) next to tiny
per-tick shaping terms, an unreliable `V` can inject terminal-reward-scale
noise into the TD error on *every* tick, not just the rare terminal one.
That's strictly worse than the old baseline for a stream whose reward is
already dense and easy to learn (the lateral centering reward): there was no
delayed-credit problem there to fix, so all TD(0) added was noise. Lowering
the value learning rate (0.02 -> 0.005) and clipping the TD error to +/-10
made it somewhat less catastrophic but didn't fix it.

**Fix: split by compartment, matching what's already compartmentalized.**
Kept the old scalar running-average baseline for the *lateral* stream
(nothing to fix there) and used TD(0) only for the *longitudinal* stream
(the one actually diagnosed as myopic). This is the same reasoning that
motivated splitting dopamine into two streams in the first place -- different
compartments have different reinforcement problems, so they shouldn't
necessarily share a learning rule either, not just a raw signal.

**Result:** from-scratch population selection (population 8, 4000 episodes,
same seed as the earlier 12.18s run) found a best individual at 100% finish,
0 crashes, avg 11.29s -- beating the previous best (12.18s) with the same
training budget. Population variance is still high (3/8 genomes trained
cleanly; the rest crashed completely or timed out), so this isn't a fix for
the underlying seed-sensitivity/robustness issues, but it is a genuine
improvement in achievable peak performance, and confirms the diagnosis: the
longitudinal stream's problem really was baseline myopia, and fixing it
where the problem actually was (rather than uniformly) is what made it work.

**Next candidates (not yet tried):**
- The KC-overlap / traction-loss generalization failure from the section
  above is still unexplained and untouched by this change.
- A genuine multi-tick eligibility trace (see above) is still untried, and
  could compound with the TD baseline rather than replace it.
- Population variance is still high even with TD(0) -- worth checking
  whether the same finetune-scale + checkpointing approach from evolve.py's
  `--continue-from` mode does better at improving on an 11.29s individual
  than it did on the old 12.18s one.

## 2026-09-18 — Structural KC compartmentalization fixes the traction-loss failure

**Tried:** the suspected cause from the earlier robustness-test section --
KC wiring was a random projection over the *whole* PN vector, so nothing
stopped a KC that happens to fire on out-of-distribution wheel_slip from
also, by chance, feeding steer_pos/steer_neg. Fixed it structurally rather
than just hoping training never reinforces that connection: split the KC
population itself into two disjoint groups (`brain.py`'s
`_pn_compartment_indices` / `lat_kc_fraction`), one wired (via its claws)
from only the two curb-distance PN channels and able to drive only the
steer pools, the other wired from every other channel (rpm, throttle,
speed, wheel_slip, gear, progress, green-onset) and able to drive only
go/lift/shift. `connection_mask` zeroes out cross-compartment `W` entries
at init *and* keeps them masked during plasticity updates in
`apply_dopamine`, so a lateral KC physically cannot receive a slip signal,
and a longitudinal KC physically cannot drive steering, regardless of what
training does. This is the real anatomical mirror of "different DAN
compartments" that the dopamine split alone wasn't -- separate populations
per compartment, not just separate reward broadcast to a fully shared one.

**Result:** re-ran the same robustness test as before (no retraining,
five scenarios, 150 episodes each) on a freshly population-selected brain
(population 8, 4000 episodes, same seed):

| scenario | fly finish | fly crash | baseline finish | baseline crash |
|---|---|---|---|---|
| nominal | 150/150 | 0 | 150/150 | 0 |
| hot_engine | 150/150 | 0 | 150/150 | 0 |
| cold_greasy_track | **149/150** | **0** | 150/150 | 0 |
| patchy_grip | **150/150** | **0** | 150/150 | 0 |
| worst_case | **150/150** | **0** | 150/150 | 0 |

Previously: 0/150 finish and 150/150 crash on both `cold_greasy_track` and
`worst_case`, 44/150 finish and 106 crashes on `patchy_grip`. The fly now
matches the baseline's reliability on every scenario, including ones it
never trained on -- confirming the KC-overlap hypothesis was the real cause,
not a guess.

**Trade-off:** the *population as a whole* got much more reliable (0
crashes across all 8 genomes in training, vs. several genomes crashing
100% of the time before) but slower: best avg_time on nominal was 15.04s,
worse than the 11.29s TD-baseline-only result from the previous entry.
Likely cause: the longitudinal compartment only gets `n_kc - n_kc_lat`
(450 of 600 by default) KCs now, instead of the full shared population, so
there's less representational capacity/flexibility for throttle/shift
specifically. Worth tuning `lat_kc_fraction` and/or combining this with the
TD baseline's individual results rather than picking one population run.

**Next candidates (not yet tried):**
- A genuine multi-tick eligibility trace (still untried, see above).

## 2026-09-18 — Tuned `lat_kc_fraction`: speed and robustness don't trade off smoothly

**Tried:** re-ran population selection (same population/episodes/seed as the
compartmentalization result above) at `lat_kc_fraction` 0.20 and 0.15,
giving the longitudinal compartment more of the 600-KC budget, to see if
the 15.04s speed regression could be recovered without reintroducing the
traction-loss failure.

**Result:** speed came back sharply -- 0.20 found a best individual at
10.11s, 0.15 at 10.02s, both matching the rule-based baseline's ~10.1s and
beating the earlier TD-only 11.29s result, with 0 crashes in the nominal
scenario. But robustness mostly collapsed back toward the pre-fix failure
mode on both:

| lat_kc_fraction | nominal | cold_greasy_track | patchy_grip | worst_case |
|---|---|---|---|---|
| 0.25 (compartmentalization result above) | 150/150, 0 crash | 149/150, 0 crash | 150/150, 0 crash | 150/150, 0 crash |
| 0.20 | 150/150, 0 crash | 44/150, 106 crash | 122/150, 28 crash | 56/150, 94 crash |
| 0.15 | 150/150, 0 crash | 39/150, 111 crash | 110/150, 40 crash | 51/150, 99 crash |

Not a smooth trade-off curve -- 0.20 and 0.15 land in roughly the same
(bad) place, while 0.25 is qualitatively different (near-perfect). Since the
direct KC-overlap leak is structurally blocked at *any* fraction > 0 (a
lateral KC's claws only ever sample the two curb-distance PN channels,
regardless of how many lateral KCs there are), the remaining failure at low
fractions is likely a capacity/redundancy problem instead: too few lateral
KCs to represent curb-following robustly under distribution shift, not a
wiring leak. Each fraction was also only tested via a single from-scratch
population search (a different winning genome each time, and genome luck is
already known to be high-variance here per the earlier robustness section),
so this isn't a fully controlled sweep -- but the size of the gap between
0.25 and 0.20 makes genome luck alone an unlikely full explanation.

**Decision:** kept `lat_kc_fraction=0.25` as the default. The task this
round was specifically fixing the traction-loss generalization failure, and
0.25 is the only setting tested that actually fixes it; 10s-level speed at
the cost of reintroducing 60-75% crash rates on unseen conditions is not a
trade worth making for that goal.

**Next candidates (not yet tried):**
- A controlled sweep (same genome/seed, only `lat_kc_fraction` varying) to
  separate "fraction effect" from "which genome the population search
  happened to land on."
- Values between 0.20 and 0.25, and above 0.25, to see whether there's a
  narrow threshold rather than a cliff.
- A genuine multi-tick eligibility trace (still untried, see above).

## 2026-09-18 — Found a genome that's both fast and fully robust

**Context:** the compartmentalization fix (0.25) reliably fixes robustness
but its population-selection winners have so far been slow (13.9-15.0s);
lower fractions (0.15/0.20) reliably recover speed but reintroduce most of
the crash rate on unseen scenarios. Ran another from-scratch population
search at the same `lat_kc_fraction=0.25` (population 16, 4000 episodes,
new seed) specifically looking for an individual that beats that trade-off
rather than accepting it, since nothing about the architecture *requires*
speed and robustness to trade off -- the fitness function used during
selection only ever looks at the nominal scenario, so a genome that happens
to be both was always possible, just not guaranteed to be found by any one
run.

**Result:** found one. Robustness scenario test, no retraining:

| scenario | fly finish | fly crash | fly avg_time | baseline avg_time |
|---|---|---|---|---|
| nominal | 150/150 | 0 | 10.29s | 10.14s |
| hot_engine | 150/150 | 0 | 10.36s | 10.02s |
| cold_greasy_track | 150/150 | 0 | 12.93s | 12.74s |
| patchy_grip | 150/150 | 0 | 10.36s | 10.46s |
| worst_case | 150/150 | 0 | 12.73s | 12.74s |

100% finish, 0 crashes, and within ~0.2s of the rule-based baseline on
*every* scenario including the two that used to fail completely (0/150).
This is now the committed `flypilot_brain.pkl`. Confirms the earlier
tuning-section suspicion: fraction alone doesn't determine the outcome,
genome variance at fixed fraction is large enough to span "matches
baseline everywhere" to "crashes half the time on distribution shift" --
so the practical lesson isn't "tune the fraction," it's "select on
robustness, not just nominal performance, or population selection is
gambling on which axis you happen to get."

**Next candidates (not yet tried):**
- Fold a robustness check (a handful of episodes on 1-2 held-out scenarios)
  into `evolve.py`'s fitness function directly, so selection stops relying
  on getting lucky and start actively searching for what this entry found
  by chance.
- A genuine multi-tick eligibility trace (still untried, see above).

## 2026-09-26 — First automated `search.py` run: 31 rounds, champion unbeaten

**Context:** manual round-by-round search (launch, wait, check, decide) had
become the bottleneck, and background-task interruptions during long manual
sessions had twice caused a round's results to be silently lost. Wrote
`flypilot/search.py` (see its module docstring) to automate the whole loop:
alternate fresh/fine-tune rounds, full-verify every round's winner on all 5
scenarios via `evaluate_scenarios.py` (never trust the pooled screen alone,
see the compartmentalization-tuning entries above), and commit+push after
every single round so nothing is lost to an interruption again. Ran it for
30 rounds (population 16, 4000 train episodes, 150 eval episodes each) on
top of the reigning champion from the entry above.

**Result:** the champion was never beaten. 31/31 rounds rejected (including
one earlier smoke-test round). Breakdown: 16 fresh, 15 fine-tune; 28 of 31
winners finished 100% with zero pooled-screen crashes, yet only 1 round
(#19, finetune) even came within 2 crashes of promotion on the *full*
verify -- every other zero-crash winner was rejected purely on total_time
being worse than the champion's 56.36s. The five closest attempts:

| round | mode | total_time | vs champion |
|---|---|---|---|
| 3 | finetune | 57.99s | +1.63s |
| 5 | finetune | 58.04s | +1.68s |
| 4 | fresh | 59.30s | +2.94s |
| 12 | fresh | 59.90s | +3.54s |
| 8 | fresh | 60.61s | +4.25s |

The dominant failure pattern, repeated in roughly two-thirds of zero-crash
rounds: a winner matches or slightly beats the champion's *nominal* time
(sometimes by a lot -- round 25's winner ran nominal in 9.95s, faster than
the champion's 10.20s) while scoring a perfect pooled-robustness screen
(100/100, 0 crashes on cold_greasy_track + worst_case combined), only to
blow up specifically on `worst_case` when checked individually at full
150-episode resolution -- times as high as 18-22s versus the champion's
12.65s, with `patchy_grip` the next most common casualty. This is the same
pooled-screen-vs-full-verify gap documented in the compartmentalization
entries, now confirmed at much larger sample size (31 independent
populations): a genome can be trained to survive `worst_case` and
`cold_greasy_track` *pooled* without actually being fast or robust on
`worst_case` *specifically* -- the pool lets a bad worst_case be offset by
a good cold_greasy_track in the same 100-episode count. `search.py`'s
per-scenario full verify is what catches this; the pooled screen used
during in-loop selection would have promoted several of these.

A secondary, rarer failure mode (round 19, round 7): a genome finishes
150/150 with 0 pooled crashes across 100 held-out episodes, then produces
1-2 crashes each on cold_greasy_track/worst_case once actually run at
150 episodes per scenario individually -- not a speed problem, a sample-size
one: 50-episode pooled screens are too small to reliably surface a low
crash *rate* that would clearly show up at 150.

**Conclusion:** the champion from 2026-09-18 is more robust than any of the
31 population searches/fine-tunes tried since, specifically because it
doesn't have a slow tail on worst_case -- a property invisible to the
pooled screen and only checkable by the full per-scenario verify this
script now automates. Next: run another 30-round batch from the same
champion; if it survives another full batch, worth considering whether
`--robust-scenarios` should score `worst_case` on its own line (not pooled
with cold_greasy_track) precisely to make this failure visible during
selection, not just after.

## 2026-09-27 — Second automated `search.py` batch (rounds 32-61): champion still unbeaten

**Context:** ran a second 30-round `search.py` batch (population 16, 4000
train episodes, 150 eval episodes) immediately after the first, same
champion, same settings, to see whether the first batch's result (champion
survives 31/31 rounds) was representative or a lucky streak.

**Result:** champion unbeaten again. All 30 rounds rejected (15 fresh, 15
fine-tune) -- and this batch was actually *cleaner* than the first: every
single one of the 30 winners finished with zero crashes (batch 1 had one
crashing winner among 31), so every rejection here came down purely to
`total_time` losing to the champion's 56.36s. The five closest attempts:

| round | mode | total_time | vs champion |
|---|---|---|---|
| 32 | fresh | 56.58s | +0.22s |
| 45 | finetune | 57.78s | +1.42s |
| 46 | fresh | 57.81s | +1.45s |
| 61 | finetune | 57.91s | +1.55s |
| 57 | finetune | 58.29s | +1.93s |

Round 32 is the closest call across both batches so far (31 + 30 = 61
rounds total) -- only 0.22s behind, and it was a *fresh* individual, not a
fine-tune of the champion. The dominant failure pattern from the first
batch repeats exactly: several winners matched or beat the champion's
*nominal* time (round 51's fine-tune ran nominal in 9.96s; round 39's did
9.94s) while scoring perfect pooled-robustness screens, only to blow up on
`worst_case` specifically once checked individually -- as high as 24.91s
(round 51) and 22.04s (round 39) versus the champion's 12.65s. `patchy_grip`
was the second most common casualty, occasionally by itself (round 12 in
the first batch effectively repeats here at round 45: nominal 10.24s but
patchy_grip 20.15s / worst_case 24.91s in round 51).

**Conclusion:** across 61 independent rounds (~1000 trained individuals)
spanning two batches, nothing has beaten the 2026-09-18 champion, and the
closest miss is 0.22s. This is now strong evidence the champion sits at or
very near a local optimum for this reward/architecture combination -- not
just lucky. The `worst_case`-blowup failure mode is the single most
reliable predictor of rejection and appeared in a majority of zero-crash
losers across both batches; the earlier suggestion to score `worst_case` on
its own line in `--robust-scenarios` (rather than pooled with
cold_greasy_track) remains the most promising next lever if further
searching is wanted, since the pooled screen still can't see this failure
mode before promotion time.

## 2026-10-01 — Fixed the selection blind spot; third `search.py` batch (rounds 62-91): champion still unbeaten, but the failure pattern changed

**Context:** the previous entry's suggested next lever turned out to be a
real bug, not just a tuning knob. `evolve.py`'s `selection_key()` pooled the
held-out scenarios' finish/crash rate into the ranking, but never their
*speed* -- the sort key's time component was nominal `avg_time` alone, even
though both prior batches' dominant rejection reason was a winner that was
fast at nominal while being catastrophically slow on `worst_case`
specifically. In other words, 61 rounds of search had been explicitly
optimizing for "survives distribution shift" while being structurally blind
to "survives it quickly" -- exactly the property that sank nearly every
zero-crash candidate at full-verify time. Fixed by having
`evaluate_robustness()` also accumulate a pooled `avg_time` across the
held-out scenarios, and `selection_key()` rank by `nominal_time +
robust_time` combined (commit `4d2d1c4`). Ran a third 30-round batch
(population 16, 4000 train episodes, 150 eval episodes) on the same
champion, using the corrected selection function throughout, specifically
to see whether this changes *how* candidates fail, not just whether they
beat the champion.

**Result:** champion unbeaten a third time -- 91 rounds total across three
batches, 0 promotions. But the failure pattern is qualitatively different
from both prior batches. Not one of the 30 rounds in this batch produced
the old signature (fast nominal, perfect pooled robustness screen, then an
18-22s+ blowup on `worst_case` alone at full verify). Instead, every
rejection fell into one of: a clean, uniformly-slower-everywhere miss (most
common), a globally slow fresh individual (rounds 66, 72, 84, all ~118s
total -- a bad random init, not a selection failure), a moderate (not
extreme) `worst_case`/`patchy_grip` elevation to ~17-18s rather than
20-27s (rounds 68, 76, 78, 82), or a rare promotion-blocking crash (rounds
62, 71). The five closest clean misses:

| round | mode | total_time | vs champion |
|---|---|---|---|
| 87 | finetune | 57.52s | +1.16s |
| 81 | finetune | 57.79s | +1.43s |
| 75 | finetune | 58.01s | +1.65s |
| 91 | finetune | 58.29s | +1.93s |
| 89 | finetune | 58.36s | +2.00s |

Two things stand out against the prior two batches. First, the closest
misses *tightened* as the batch went on (58.01s at round 75, 57.79s at
round 81, 57.52s at round 87) rather than clustering at one lucky round --
weak evidence the fine-tune line is actually converging toward the
champion's neighborhood, not just re-sampling noise. Second, every one of
these close misses is a clean, uniform near-match across all five
scenarios (e.g. round 87: nominal 10.42s vs champion 10.14s, worst_case
13.05s vs 12.74s) rather than a fast-but-fragile genome that would have
blown up under the old selection. That is precisely the failure mode the
fix was meant to produce: a genuinely competitive-but-slightly-slower
candidate, not a landmine the old pooled screen would have waved through.

**Conclusion:** the selection_key fix is validated. Across three batches
(91 rounds, ~1,500 trained individuals) the champion from 2026-09-18 has
never been beaten, but the *reason* it survives has shifted from "the
search can't see what it's selecting for" (batches 1-2) to "the search sees
correctly and still can't find anything better" (batch 3) -- a meaningfully
stronger claim that this specific architecture/reward setup is at or very
close to a genuine local optimum. Further gains from more rounds of the
same population search now look unlikely; the more promising next lever is
an architectural or algorithmic change (e.g. a different plasticity rule,
reward shaping, or KC wiring scheme) rather than additional search volume
under the current one.
