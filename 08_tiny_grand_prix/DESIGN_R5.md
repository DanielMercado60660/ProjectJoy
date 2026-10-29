# Round 5 — discover the road's grip (turn 1)

**Question.** Does memory improve driving after an invisible surface change beyond a
strong online physics estimator and a feed-forward policy? This is a deterministic
bicycle simulator experiment, not evidence about real tire identification.

**Task and information.** Strict collisions terminate a 60-second, one-lap attempt.
Train on a balanced geometry mixture with 40% spatial wet patches, 40% mid-lap
switches (both directions), and 20% dry controls. High grip is U(.9,1.1), low grip
U(.4,.65); entry is at .22–.38 lap and patches span .24–.32 lap. No appearance
channel exists; render paint is independent of grip. All new methods see the same
14 sensors: ten rays, speed, previous action, and *previous-step measured yaw rate*.
No method receives true grip, curvature, lateral error, or a change indicator.

**Comparisons.** (1) MLX PPO with a 32-unit GRU; (2) matched PPO with the identical
cell's hidden state zeroed every step; (3) classical ray follower plus scalar RLS
grip estimation, using clipped yaw/acceleration as equality evidence and un-clipped
motion only as a lower bound. Both PPO arms start identically from
`ppo_dr_r0_rg_best.npz`, with a zero-initialized memory output, equal rollout/gradient
budgets and alternating update order. Their existing pretraining is shared;
classical comparison is explicitly **prior-data-unmatched**. The frozen old 13-input
policy is an extra-data historical reference. Also evaluate the trained GRU with
memory disabled and the tuned estimator frozen. These are ablations, not retuned baselines.

**Baseline strength.** CEM tunes all 13 knobs: steering gain .5–3, ray power 1–8,
centering 0–2, braking fraction .25–1.2, clearance 2–18 m, maximum speed 18–40,
minimum speed 2–8, initial grip .4–1.1, forgetting .8–1, optimistic drift 0–.15/s,
grip safety factor .55–1, slip tolerance .005–.15, steering compensation 0–.8.
Include the old tuned DR controller's knobs, 25% global proposals every generation,
and a variance floor. Candidates share exact scenarios; optimization uses train
only. Noise-free odometry deliberately gives the classical model its full strength.

**Splits and selection.** Train: 64 regular seeds 30000+ and hard seeds 2030000+.
Validation: 16 regular seeds 40000+ and hard seeds 2040000+, crossed with
patch/switch/dry: **96 episodes, all six groups every selection**. Fresh test: 64
regular seeds 1070000+ and hard seeds 2070000+, 384 episodes/method; surface seed
2026102408. On R5 generator exhaustion only, retry effective seeds
`2**32 + 256*requested_effective_seed + retry` (1–32), retaining the geometry and
validity checks; first success wins, independent of policy results or batch size.
Log each replacement and retain the bank manifest. Successful seeds and the legacy
100/50 split use the original generator unchanged. Old tests are secondary only. Evaluation surfaces are keyed by
track/mode/seed, so subset demos reproduce headline episodes exactly. Selection
maximizes equal-group mean `2*finish - 2*crash + signed_progress_fraction - penalized_time/60`;
DNFs cost 60 s. Neither test split selects checkpoints.

**Measures.** Report completion, collisions/attempt, timeout, finished lap time,
DNF-penalized lap time and paired jointly-finished differences. Adaptation is time
to the onset of five consecutive safe steps: lateral fraction ≤.5 and speed within
.6–1.3 of `clip(sqrt(mu*g/max(abs(curvature),.01)),6,30)`. True physics appears only
in this metric. This measures recovery to an envelope, not proof of grip inference.
Pre-register two events/patch and one/switch; cap failures, interruptions and
unreached events at 5 s. Report exposure/recovery coverage and all censor statuses.

**Budget and artifacts.** Battery: 18 minutes paired PPO + 12 minutes CEM, ≤2-minute
smoke, 2.5-minute test allowance, .5-minute demo allowance. CPU receipt measured
171k recurrent rollout steps/s, 1.24 s/CEM batch and .25 s/full validation; MLX update
rate is unmeasured. Huge ceilings plus measured-rate scheduling fill wall budgets.
Log each arm's active time and sample counts; disclose any compute imbalance.
Only fixed selected exports need committing. Atomic resume state includes both
optimizers, simulators, hidden states, step, RNG keys, configuration and selection.
One seed on battery; independent paired seeds 1/2 on AC. Preserve failures in replay.
