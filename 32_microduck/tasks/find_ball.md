# Task 3: find the missing ball (`find_ball`)

**Instruction:** `{"task": "find_ball", "target": null}`.

**Setup version 2** (2026-10-26, before M3). Version 1 always rested the ball about 0.3 m straight
behind the occluder, so "walk behind the occluder and look" needed no memory (M2 review). Version 2
varies where the ball hides, so watching where it went pays.

## Start conditions (a task-specific start, overriding the config's start pose)
- **Hiding place:** per episode, either behind the **occluder** (40 × 20 cm) or behind an open end
  of the **partition** (30 cm tall). Each episode prefers one at random; the other is the fallback.
- **The roll:** at t = 0 the ball (70 mm, 15 g; rolling resistance like a ball on carpet) starts
  beside one end of the chosen object, on the duck's side. It is launched rolling, so it passes that
  end and stops at a randomly drawn point behind the object. Within ~6 s it is 0.05–0.35 m back from
  the end and 0.22–0.45 m behind.
- **The duck:** stands 0.7–1.3 m away on the open side, facing between the ball and the end, and
  watches it go.
- **Spread** over 440 episodes (k = 0…3 of 60 training + 50 instances-dev configs,
  `../results/m2/battery.json → find_ball_v2_spread`):
  - hidden behind the occluder 214, behind the partition 226
  - ends up to the duck's left in 224, to its right in 216
  - back from the end 0.005–0.40 m (median 0.19); behind the object 0.18–0.48 m (median 0.33).
- **Checked per episode** (`tasks._find_ball_setup`):
  - the roll clears the object's end by at least 8 cm
  - the ball is visible from the start at launch and **in view for at least 0.8 s** of the first 6 s
    (median 1.5 s, min 0.9 s)
  - the ball rests hidden from the start, within 8 cm of the drawn point (max error over all setups
    7.9 cm)
  - some viewpoint with a clear view of the rest point is reachable from the start, so every episode
    is solvable without running a controller.
- **Coverage:** **970/970** configs in every split have a valid setup. The one setup v1 had to exclude
  now hides the ball behind the partition.
- Step budget: 600 ticks (60 s).

## What the controller observes
The head camera (it watches the ball roll away), body state (including the tick's 50 Hz IMU samples)
and the instruction. It has to remember where the ball went (which object, which end, roughly how
far) and get another view. Nothing tells it the ball's position.

## Success (judged at `done`)
- The ball has come to rest (< 0.01 m/s) and was out of view at some tick before (M2 review: without
  this, saying `done` at tick 1 passed some episodes).
- The ball's centre is inside the head camera's 45° view with a clear line of sight. The ray ignores
  the duck's own geometry.
- The trunk is within 0.8 m of the ball.
- The duck is stopped.

## Failure
The budget runs out, or at `done` the ball is still rolling, has never been out of view, is not in
view, or is too far.

**Recorded:** the first tick the ball was back in view after being hidden, falls, wall contact, time.

## Does memory matter?
Measured on episode k = 0 of 60 training + 50 instances-dev configs (oracle: `battery.json →
find_ball_v2_baselines`; the two searchers: the train + instances-dev rows of
`results/m3/find_ball_baselines.json`, re-run after the M3 review fix below). All three controllers
get the true pose and layout:
- **oracle:** knows where the ball is.
- **search everywhere:** no memory. It visits candidate hiding regions behind every object (on the
  side away from the start), nearest first, each from a spot with a clear view, recognises the ball
  perfectly when it's in view, and then walks to about 0.65 m from it if it is farther than 0.75 m.
- **look behind the occluder:** the same, but only behind the occluder (what setup v1 rewarded).

| controller | found within 20 s | 30 s | 40 s | 60 s (the budget) |
|---|---|---|---|---|
| oracle (knows where it is) | 103 | 110 | 110 | **110** |
| search everywhere, no memory | 54 | 69 | 98 | **104** |
| look behind the occluder only | 49 | 51 | 51 | **51** (3/58 when the ball is behind the partition) |

So the v1 shortcut now fails more than half the time. An exhaustive memoryless search with *perfect*
recognition still finds the ball in 95% of episodes within 60 s, but needs much longer: 54 vs 103
found within 20 s. Knowing where the ball went buys speed and the last ~5%.

Two caveats. "No memory" is not quite "no information": the duck starts 0.7–1.3 m from the hiding
place, so the searcher's nearest-first order visits the right object and end first in 114 of the 250
M3 evaluation episodes, against ~96 expected from a random first pick among its 4 regions. And in M3 the reference bot's imperfect recognition did not widen the gap in its
favour: it found the ball in 228/250 evaluation episodes against the searcher's 236/250, and more
slowly (`../RESULTS.md`, M3). M6 should report time to success next to success, and include the
memoryless searcher as a reference line.

*(M3 review: until then the searchers said `done` as soon as the ball was in view, sometimes
0.81–1.32 m away, outside the judge's 0.8 m; they scored 52/66/91/96 and 49/49/49/49 here.)*

## Permitted variations by split
| factor | what changes for this task |
|---|---|
| instances | new rooms from the training rules: occluder and partition placement, so new hiding places, viewpoints and rolls |
| arrangements | the occluder moves to reserved places (training: edge cells; dev: corners; test: the room centre); the partition is attached to another wall (dev) or free-standing with two open ends (test) |
| appearances | training rooms with reserved ball shades (magenta band), floors and lighting |
| distractor | a flickering wall screen that sometimes shows ball-like colours |

## Feasibility
The oracle (reads the true ball position, plans a viewpoint 0.55–0.75 m away with a clear line of
sight, turns to it, small head sweep) succeeded in **60/60** training and **50/50** instances-dev
episodes, with 0 falls. Median 16 s.

Trivial controllers score 0/440: `done` at once, stand 59 s then `done`, or look away and back
(`results/m2/review_checks.json`, re-run on setup v2). That file's `find_ball_rest_point` block
measures the rest point relative to the *occluder* only. It describes setup v1 and doesn't fit
partition hides; use `find_ball_v2_spread` instead.

## A satisfying demo shows
The ball rolling round the end of the occluder or the partition and out of sight while the duck
watches; the duck going to the right side of the right object; the ball back in frame; then `done`.
Replay: what the bot remembers ("ball went behind the partition, past its open end, about 0.3 m in")
and when it saw the ball again.
