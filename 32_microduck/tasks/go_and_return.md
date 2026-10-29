# Task 1: go there and come home (`go_and_return`)

**Instruction:** `{"task": "go_and_return", "target": "red" | "green" | "blue" | "yellow"}`. The target
is one of the 3–4 pads present in the room, drawn per episode.

## Start conditions
- The duck stands on the home mat: the config's start pose, within 8 cm of the mat centre, facing a
  random direction.
- The lamp and ball are as the config sets them; neither matters here.
- Step budget: 1200 ticks (120 s).

## What the controller observes
The head camera, body state and the instruction. No map and no pose: the controller has to find the
named pad by its colour and remember where home is. The oracle has the true layout and pose.

## Success (judged at `done`)
1. At some tick the trunk was on the target pad: inside the 30 cm pad square grown by 5 cm.
2. **After** that, the trunk is on the home mat (40 cm square grown by 5 cm).
3. At `done` the duck is stopped (planar speed < 0.05 m/s), within the budget.

## Failure
- The budget runs out.
- `done` comes before the pad was reached, or while off the mat.

Falls and wall contact are not failures in themselves (velstand gets up), but they are recorded.
**Also recorded:** which other pads were entered before the target ("wrong pad visits"), the time to
reach the pad, and the time home.

## The partition
In most rooms the partition blocks some straight lines. The controller must go around it: the
oracle plans on the true map; the reference bot (M3) must cope with what it has seen.

## Permitted variations by split
| factor | what changes for this task |
|---|---|
| instances | new rooms from the training rules: pad positions and colours, partition, start pose and heading |
| arrangements | dev: home mat on a N/S edge cell, partition from the E or W wall, exactly 1 pad in the room's interior. Test: home on an E/W edge cell, a free-standing partition, 2+ interior pads |
| appearances | the exact rooms of training configs with reserved pad shades (hue shifted 8–14°, or darker), floors and lighting. The colour names stay true |
| distractor | a training room plus a wall screen flickering random colours at 10 Hz, sometimes pad-like |

## Feasibility (M2, `results/m2/battery.json`)
The oracle (skills + true pose and layout) succeeded in **60/60** training and **50/50** instances-dev
episodes, with 0 falls. Median episode: 39 s on training. Saying `done` at once scores 0/440
(`results/m2/review_checks.json`).

## A satisfying demo shows
The duck leaving the mat, looking for the named colour, walking around the partition to the right
pad, pausing on it, then walking home and stopping on the mat. In the replay: the instruction, the
target, the current skill and what the bot remembers about where home and the pad are.
