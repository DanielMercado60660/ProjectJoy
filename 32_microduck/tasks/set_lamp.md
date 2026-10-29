# Task 2: set the lamp (`set_lamp`)

**Instruction:** `{"task": "set_lamp", "target": "on" | "off"}`.

## Start conditions
- The duck stands on the home mat (the config's start pose).
- The lamp's initial state comes from the config (an instance factor).
- The target is drawn per episode, so that **about half the episodes already match** (the right answer
  there is to look, see it's right, and press nothing).
- Step budget: 900 ticks (90 s).

## The mechanism
- **Lamp:** a box on one wall, dark grey when off and bright warm white when on.
- **Button:** a 24 cm plate in a dark frame, 5 mm proud of the floor.
  - Pressing it 2.5 mm (about half the duck's weight) toggles the lamp.
  - The plate re-arms only after it has been unloaded for 1 s, so one walk across = one toggle.
  - Turning in place on the plate's edge can register a second press (M1 review).
  - Walking across the centre and turning only off the plate gives exactly one press: 40/40 in M2's
    battery.

## What the controller observes
The head camera (it must *see* the lamp to know its state), body state and the instruction. It does
not observe presses or the lamp flag.

## Success (judged at `done`)
The lamp is in the target state at `done`, within the budget.

## Failure
- The wrong lamp state at `done`.
- The budget runs out.

**Recorded:**
- presses
- presses needed (0 or 1)
- extra presses
- plate visits without a press (missed presses)
- falls, wall contact, time.

"Press only when needed" is scored through these: an episode that toggles twice and lands right
still counts as a success, but its 2 extra presses are reported.

## Permitted variations by split
| factor | what changes for this task |
|---|---|
| instances | new rooms from the training rules; unseen initial lamp states and start poses |
| arrangements | the button moves to reserved places (training: corner cells; dev: E/W edge cells; test: N/S edge cells), and so do home and the partition. The lamp's wall position is drawn the same way everywhere |
| appearances | training rooms with reserved shades (the button's cyan shifts within its band), floors and lighting; the lamp's on/off colours don't change |
| distractor | a flickering wall screen: a bright, changing patch that a careless lamp detector could mistake for the lamp |

## Feasibility (M2)
The oracle (reads the lamp flag, presses via `press_button`) succeeded in **60/60** training and
**50/50** instances-dev episodes, with 0 falls. Only 25 (training) and 28 (dev) of those needed a
press; the oracle passed all of them with exactly one press, in a median of 26 s (training) and 27 s
(dev). The rest end at the first tick.

**Floor:** a controller that says `done` at once succeeds on every episode that needs no press: 56%
(134/240) of training and 46% (92/200) of instances-dev episodes, k = 0–3
(`results/m2/review_checks.json`). Report set-the-lamp success split by presses needed, next to this
floor. The judge cannot see whether the controller *looked* at the lamp; only the camera record can.

## A satisfying demo shows
The duck turning to look at the lamp. If it's wrong, walking across the plate, the lamp toggling,
the duck turning back to look and confirm, then stopping with `done`. If it's already right, a look
and a stop, with no press. Replay: expected outcome ("lamp toggles") vs observed (what the camera
showed).
