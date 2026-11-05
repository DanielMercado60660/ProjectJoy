"""Round 7 Stage 1D: a map-free 64 px pixel servo for set_lamp (design/R7_CAMPAIGN.md). A baseline row, not a
learned arm: it settles whether a reactive cyan-blob servo can reach the plate under the 900-tick budget.

    OMP_NUM_THREADS=1 P2_PROCS=4 ../bin/gpu_run ../.venv/bin/python p2_servo_baseline.py [select|confirm|fresh]
        -> results/r7/eval_<rooms>_servo_oracle.json, eval_<rooms>_servo_lamp.json (r7_eval rows)

Servo (both variants): the cyan plate mask at 64 px (perception.masks hue window, the bot's own thresholds),
mass and column offset averaged over 5 ticks; head pitch 0.2 while searching / far, 0.45 once the blob is near
(mass >= 300, about 0.7 m); turn in place at 0.55 while the plate is out of view (a 30-tick walk after a full turn without a sighting); walk at 0.3 steering on the
column offset; stall-and-turn (a 90 deg turn) after 60 ticks without mass growth; when the blob reaches the
frame bottom (row >= 60, about 0.35 m) a straight 55-tick cross with no turning, then 12 wait ticks.
Latch variants: "oracle" reads the task's needed_presses (simulator truth: labelled ORACLE; scores the servo
alone; done at once when no press is needed, done after the cross otherwise); "lamp" is sensors-only and
map-free: the bot's geometry-gated lamp detector + memory (belief.BeliefTracker) decide: a full turn first,
done if the remembered lamp matches the target, else servo + cross, then a turn until the lamp is re-seen
(done when it matches; another try if not; at most 3; done after a full turn without a sighting, as the bot
assumes the press worked).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from interface import Command  # noqa: E402
from p2_learner import prep_frame  # noqa: E402
from perception import masks  # noqa: E402

TURN, WALK = 0.55, 0.3
SEARCH_PITCH, NEAR_PITCH = 0.2, 0.45
NEAR_MASS, CROSS_ROW, CROSS_TICKS, WAIT_TICKS, STALL_TICKS = 300, 60, 55, 12, 60
SEEN_MASS = 0.8             # 5-tick mean cyan mass at 64 px: a 24 cm plate is 1-2 px at 2.4 m, ~30 px at 1.5 m
EXPLORE_TICKS = 30          # after a full turn without a sighting: walk ~0.4 m, then turn again
FULL_TURN_TICKS = 70        # ~360 deg at the body's ~0.8 rad/s achieved
CURVE_WZ = -0.22            # the body curves left ~0.22 rad/s when commanded straight (M0): a fixed trim, not a sensor


class Servo:
    """One episode's controller state; `needed` (oracle) or a BeliefTracker (lamp) decides the latch."""

    def __init__(self, latch, target, needed=None):
        self.latch, self.target, self.needed = latch, target, needed
        self.mass, self.off = [], []
        self.tracker = None
        if latch == "lamp":
            from belief import BeliefTracker
            self.tracker = BeliefTracker()
        self.tries = 0
        self.log = []

    def cue(self, frame):
        m = masks(prep_frame(frame, 64))["button"]
        ii, jj = np.nonzero(m)
        self.mass.append(int(m.sum()))
        self.off.append(float((jj.mean() - 31.5) / 32.0) if len(jj) else 0.0)
        bottom = int(ii.max()) if len(ii) else -1
        k = min(5, len(self.mass))
        return float(np.mean(self.mass[-k:])), float(np.mean(self.off[-k:])), bottom

    def lamp_matches(self):
        st = self.tracker.mem.lamp_state()
        return None if st is None else (st[0] == self.target)

    def factory(self, iface, instruction):
        obs = iface.observe()
        tick = 0

        def step(cmd):
            nonlocal obs, tick
            obs = yield cmd
            if obs is None:
                obs = iface.observe()
            tick += 1
            if self.tracker is not None and obs.tick > 0:
                self.tracker.update(obs)

        if self.latch == "oracle":
            if not self.needed:
                yield Command(done=True)
                return
        else:
            for _ in range(FULL_TURN_TICKS):                       # look around for the lamp first
                yield from step(Command(wz=TURN, head=(0.0, 0.0, 0.0, 0.0)))
            if self.lamp_matches():
                yield Command(done=True)
                return
        while True:
            # ---- servo to the plate
            best, best_t, stall, unseen = 0.0, tick, 0, 0
            while True:
                mass, off, bottom = self.cue(obs.frame())
                pitch = NEAR_PITCH if mass >= NEAR_MASS else SEARCH_PITCH
                if bottom >= CROSS_ROW and mass >= NEAR_MASS:
                    break
                if mass < SEEN_MASS:                                 # out of view: turn in place; after a full
                    unseen += 1                                      # turn without a sighting, walk a step
                    if unseen > FULL_TURN_TICKS:
                        self.log.append({"tick": tick, "event": "explore_step"})
                        for _ in range(EXPLORE_TICKS):
                            yield from step(Command(vx=WALK, head=(0.0, SEARCH_PITCH, 0.0, 0.0)))
                        unseen = 0
                    yield from step(Command(wz=TURN, head=(0.0, pitch, 0.0, 0.0)))
                    best, best_t = 0.0, tick
                    continue
                unseen = 0
                if mass > best + 20:
                    best, best_t = mass, tick
                if tick - best_t > STALL_TICKS:                      # stall-and-turn
                    stall += 1
                    self.log.append({"tick": tick, "event": "stall_turn"})
                    for _ in range(20):
                        yield from step(Command(wz=TURN, head=(0.0, pitch, 0.0, 0.0)))
                    best, best_t = 0.0, tick
                    continue
                yield from step(Command(vx=WALK, wz=float(np.clip(-1.2 * off, -0.6, 0.6)), head=(0.0, pitch, 0.0, 0.0)))
            self.log.append({"tick": tick, "event": "cross"})
            for _ in range(CROSS_TICKS):                             # across without turning in place: steer on the
                mass, off, _ = self.cue(obs.frame())                 # blob while it is visible, then hold the body's
                wz = float(np.clip(-1.2 * off, -0.6, 0.6)) if mass >= 50 else CURVE_WZ   # measured left-curve trim
                yield from step(Command(vx=WALK, wz=wz, head=(0.0, NEAR_PITCH, 0.0, 0.0)))
            for _ in range(WAIT_TICKS):
                yield from step(Command(head=(0.0, NEAR_PITCH, 0.0, 0.0)))
            self.tries += 1
            if self.latch == "oracle":
                yield Command(done=True)
                return
            # ---- re-check the lamp: turn until it is re-seen
            seen_before = self.tracker.mem.lamp_state()
            t_seen = -1 if seen_before is None else seen_before[1]
            for _ in range(FULL_TURN_TICKS):
                yield from step(Command(wz=TURN, head=(0.0, 0.0, 0.0, 0.0)))
                st = self.tracker.mem.lamp_state()
                if st is not None and st[1] > t_seen and st[1] >= tick - 2:
                    break
            st = self.tracker.mem.lamp_state()
            fresh = st is not None and st[1] > t_seen
            self.log.append({"tick": tick, "event": "lamp_recheck", "fresh": fresh, "state": None if st is None else st[0]})
            if not fresh or st[0] == self.target or self.tries >= 3:
                yield Command(done=True)
                return


def controller_factory(variant):
    """r7_eval's `who` hook: a factory per episode (the oracle variant reads the task's needed_presses)."""
    def make(cfg, ep):
        needed = None
        if variant == "oracle":
            needed = (ep["instruction"]["target"] == "on") != bool(cfg["layout"]["lamp_initial_on"])
        return Servo(variant, ep["instruction"]["target"], needed)
    return make


def main():
    import r7_eval
    which = sys.argv[1] if len(sys.argv) > 1 else "select"
    variants = sys.argv[2].split(",") if len(sys.argv) > 2 else ("oracle", "lamp")
    r7_eval.evaluate(which, {f"servo_{v}": f"servo:{v}" for v in variants}, procs=int(os.environ.get("P2_PROCS", "4")))


if __name__ == "__main__":
    main()
