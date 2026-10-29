"""Reviewer-added scripted opponents (cross-review, 2026-10-24). The original scripts in sumo_env.py are unchanged.

Motivation: head-on pushes between equal robots stall for the full 10 s, and all four original scripts
(and the PPO policy) steer their nose straight at the opponent's centre. These scripts try to break the
stall by getting an angle. Same 15-number egocentric observation as everything else (matched information).
Obs indices: 0-1 own vel, 2 yaw rate, 3-4 centre (own frame), 5 own edge margin,
6-7 opponent rel pos, 8 dist, 9-10 rel vel, 11-12 opponent heading (cos, sin) in own frame,
13 opp edge margin, 14 up.
"""
import numpy as np
from sumo_env import _drive, angle_of


class Hooker:
    """Approach a point beside the opponent (offset along its side normal, on the side facing me and,
    when there is a choice, the ring-centre side), then ram it from that angle once within `switch`.
    While in contact, steer with a small bias `lean` (radians added to the aim angle) so the nose
    stays on the opponent's corner instead of squaring up.
    Params: offset (m), switch distance (m), gain k, lean (rad)."""
    name = "hooker"

    def __init__(self, offset=0.3, switch=0.35, k=2.5, lean=0.0):
        self.offset, self.switch, self.k, self.lean = offset, switch, k, lean
        self.side = None

    def reset(self):
        self.side = None

    def __call__(self, o):
        ctr, rel, dist = o[3:5], o[6:8], o[8]
        h = o[11:13]
        nrm = np.array([-h[1], h[0]])                 # opponent's left-side normal (my frame)
        if self.side is None:
            # pick the side facing me; tie-break toward the ring-centre side of the opponent
            s_me = np.dot(nrm, -rel)
            s_ctr = np.dot(nrm, ctr - rel)
            self.side = 1.0 if (s_me + 0.5 * s_ctr) >= 0 else -1.0
        if dist > self.switch:
            target = rel + self.side * nrm * self.offset
            if np.dot(self.side * nrm, -rel) < -0.05:  # target is on the far side: aim wide of the opponent
                target = target + self.side * nrm * self.offset
            ang = angle_of(target)
        else:
            ang = angle_of(rel) + self.side * self.lean
        fwd = 1.0 if abs(ang) < 1.2 else 0.2
        return _drive(fwd, self.k * ang)


class Leaner:
    """Charger that squares up from range, but once in contact leans its push off-centre by `lean`
    radians (side chosen so the push drives the opponent toward its nearer edge), so the opponent's
    nose slides and it gets turned. Params: gain k, lean (rad), contact distance (m)."""
    name = "leaner"

    def __init__(self, k=1.0, lean=0.4, contact=0.4):
        self.k, self.lean, self.contact = k, lean, contact

    def __call__(self, o):
        ctr, rel, dist = o[3:5], o[6:8], o[8]
        ang = angle_of(rel)
        if dist < self.contact:
            # rotate the push direction toward "away from ring centre" around the opponent
            out_dir = rel - ctr                       # centre -> opponent
            side = np.sign(rel[0] * out_dir[1] - rel[1] * out_dir[0]) or 1.0
            ang = ang + side * self.lean
        fwd = 1.0 if abs(ang) < 1.2 else 0.2
        return _drive(fwd, self.k * ang)


REVIEW_CLASSES = {"hooker": Hooker, "leaner": Leaner}


class Corner:
    """Aim at a point beside the opponent's centre, offset perpendicular to the line between us, so the
    noses meet corner-to-corner instead of square. Two robots pushing square head-on stall; an 8 cm offset
    turns the opponent (~50 degrees in 4 s in the reviewer's probe) so its push starts to glance off.
    Side is picked once per bout: 'init' = the side the opponent is already offset to; 'edge' = the side
    that pushes the opponent toward its nearer edge (re-evaluated each step).
    Params: off_far (m, offset while approaching), off_near (m, offset inside d1), d1 (m), gain k, side."""
    name = "corner"

    def __init__(self, off_far=0.15, off_near=0.08, d1=0.5, k=2.0, side="init"):
        self.off_far, self.off_near, self.d1, self.k, self.side_mode = off_far, off_near, d1, k, side
        self.side = None

    def reset(self):
        self.side = None

    def __call__(self, o):
        ctr, rel, dist = o[3:5], o[6:8], o[8]
        u = rel / (dist + 1e-6)
        perp = np.array([-u[1], u[0]])                # left of the line to the opponent
        if self.side_mode == "edge":
            out_dir = rel - ctr                       # centre -> opponent, in my frame
            # aiming to the right of its centre turns it left (and vice versa); push it toward the edge
            self.side = -1.0 if np.dot(out_dir, perp) > 0 else 1.0
        elif self.side is None:
            self.side = 1.0 if rel[1] >= 0 else -1.0
        off = self.off_far if dist > self.d1 else self.off_near
        target = rel + self.side * perp * off
        ang = angle_of(target)
        fwd = 1.0 if abs(ang) < 1.2 else 0.2
        return _drive(fwd, self.k * ang)


REVIEW_CLASSES["corner"] = Corner


class Rehit(Corner):
    """Corner, plus a stall breaker: if in contact (dist < 0.4) and own speed < `v_stall` for `stall`
    control steps, reverse with a turn for `back` steps (switching attack side), then attack again.
    Params: Corner's, plus stall (steps), back (steps), v_stall (m/s)."""
    name = "rehit"

    def __init__(self, stall=12, back=8, v_stall=0.1, **kw):
        super().__init__(**kw)
        self.stall, self.back, self.v_stall = stall, back, v_stall
        self.cnt, self.rev = 0, 0

    def reset(self):
        super().reset()
        self.cnt, self.rev = 0, 0

    def __call__(self, o):
        if self.rev > 0:
            self.rev -= 1
            return _drive(-1.0, 0.5 * (self.side or 1.0))
        if o[8] < 0.4 and np.linalg.norm(o[0:2]) < self.v_stall:
            self.cnt += 1
        else:
            self.cnt = 0
        if self.cnt >= self.stall:
            self.cnt, self.rev = 0, self.back
            if self.side_mode == "init":
                self.side = -(self.side or 1.0)
        return super().__call__(o)


REVIEW_CLASSES["rehit"] = Rehit
