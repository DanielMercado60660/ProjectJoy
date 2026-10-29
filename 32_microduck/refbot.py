"""The reference bot (M3): colour segmentation on the head camera, an explicit memory, leg odometry,
a camera-built occupancy map, and a hand-written state machine per task, all through the frozen
interface and the M2 skills. Contract: results/m3/bot_contract.md (no world, no config, no judge).

Every tick the bot senses before its skill acts: odometry from the encoders and IMU, the frame
segmented into colour blobs projected on the floor, memory tracks updated, and (every other tick) the
occupancy map. Objects enter memory only after `confirm` consistent sightings, which keeps the
flickering screen's one-tick colours out. Memory is a plain JSON-able dict with named slots and
times since seen.

Tunable parameters live in PARAMS (tuned on train/dev only; results/m3/tuning.jsonl).
"""
import numpy as np

from bodymodel import BodyModel
from botmap import FloorModel, OccupancyGrid, walkable_mask
from gaze import Gaze
from interface import Command
from odometry import Odometry
from perception import CameraPose, detect
from skills import Ctx, go_to, go_to_planned, look, press_button, stop, turn_to, wrap

PARAMS = {
    "confirm": 3,               # sightings before a track is trusted
    "assoc_m": {"pad": 0.30, "button": 0.25, "mat": 0.35, "ball": 0.35, "lamp": 0.45},
    "scan_pitch": 0.15,         # head pitch during map/search scans
    "learn_pitch": 0.12,        # head pitch for the first look-around: sees the floor ring (>= 0.30 m),
                                # the map out to the walls and the lamp (~2 deg above the horizon)
    "scan_wz": 0.55,            # turn command during a continuous look-around
    "approach_pitch": 0.3,      # head pitch while walking to a floor target (keeps it in view)
    "servo_ticks": 25,          # re-plan towards the freshest estimate every N ticks
    "map_every": 2,             # occupancy update every N ticks
    "explore_radius": 0.9,      # exploration viewpoints on a ring around the start
    "lamp_confirm": 3,          # consecutive lamp-on ticks with a steady bearing and colour
    "lamp_bearing_tol": np.radians(6),
    "max_press_tries": 3,
    "gaze": False,              # chicken-head reflex on the outgoing head commands (gaze.py); the frozen
                                # M3 bot (eval_final.json) ran without it
    "anchor_n": 10,             # a landmark seen this often anchors the pose (drift correction)
    "anchor_dist": 1.6,         # only near, whole sightings correct the pose
    "anchor_gain": 0.15,        # fraction of the landmark residual applied per tick
    "anchor_max_step": 0.02,    # metres per tick, at most
    "anchor_max_resid": 0.35,   # larger mismatches are probably the wrong landmark
    "button_avoid_r": 0.22,     # no-go disc around the plate when not pressing
    "ball_min_area_frac": 0.70, # a "found" ball blob must be this fraction of a full ball's size
    "ball_lost_ticks": 4,       # ball unseen this long = it went out of sight
    "ball_view_radius": (0.55, 0.65, 0.75),
}
FLOOR_CLASSES = ("pad_red", "pad_yellow", "pad_green", "pad_blue", "button", "ball", "mat")


def _single(cmd):
    """A one-tick skill: send cmd once (so the bot's drive() senses that tick)."""
    yield cmd
    return {"skill": "tick", "ok": True}


class Memory:
    """Named slots built from confirmed tracks. Everything is in the odometry frame."""

    def __init__(self, P, grid=None):
        self.P, self.grid = P, grid
        self.tracks = []        # {cls, xy, n, w, last, first, border_only}
        self.lamp = {"state": None, "bearing": None, "from": None, "last_on": None, "run": [], "xy": None}
        self.ball_trail = []    # (tick, xy) of ball sightings
        self.visited = []

    def update(self, dets, tick, cam_xy):
        """Associate detections with tracks. Returns residuals (track - detection) for detections of
        well-established anchor tracks (used by the bot to correct odometry drift)."""
        residuals = []
        for d in dets:
            if d["cls"].startswith("lamp") and "xy" not in d:
                continue
            key = "pad" if d["cls"].startswith("pad") else ("lamp" if d["cls"].startswith("lamp") else d["cls"])
            w = 0.3 if d["border"] else 1.0
            xy = np.array(d["xy"])
            if d["cls"] == "ball":
                self.ball_trail.append((tick, xy.tolist()))
            best = None
            for t in self.tracks:
                if t["cls"] == d["cls"]:
                    dist = np.linalg.norm(np.array(t["xy"]) - xy)
                    if dist < self.P["assoc_m"][key] and (best is None or dist < best[0]):
                        best = (dist, t)
            if best is None:
                self.tracks.append({"cls": d["cls"], "xy": xy.tolist(), "n": 1, "w": w, "last": tick,
                                    "first": tick, "consec": 1})
            else:
                t = best[1]
                anchor = (key in ("pad", "button", "mat") and t["n"] >= self.P["anchor_n"] and not d["border"]
                          and d.get("dist", 9) < self.P["anchor_dist"])
                if anchor:
                    residuals.append(np.array(t["xy"]) - xy)
                if d["cls"] == "ball":            # the ball moves: follow it, don't average
                    t["xy"] = xy.tolist()
                elif anchor:
                    pass                          # anchors stay put: they correct the pose instead
                else:
                    a = w / (t["w"] + w) if t["n"] < 30 else w / (30 + w)
                    t["xy"] = ((1 - a) * np.array(t["xy"]) + a * xy).tolist()
                t["consec"] = t["consec"] + 1 if t["last"] >= tick - 1 else 1
                t["n"] += 1
                t["w"] += w
                t["last"] = tick
        return residuals

    def best(self, cls, near=None):
        cands = [t for t in self.tracks if t["cls"] == cls and t["n"] >= self.P["confirm"]]
        if not cands:
            return None
        if near is not None:
            return min(cands, key=lambda t: np.linalg.norm(np.array(t["xy"]) - near))
        return max(cands, key=lambda t: t["w"])

    def lamp_update(self, dets, tick, cam_xy):
        ons = [d for d in dets if d["cls"] == "lamp_on"]
        run = self.lamp["run"]
        if ons:
            d = max(ons, key=lambda x: x["area"])
            if run and (tick - run[-1]["tick"] > 1 or abs(wrap(d["bearing"] - run[-1]["bearing"])) > self.P["lamp_bearing_tol"]
                        or np.abs(np.array(d["rgb"]) - np.array(run[-1]["rgb"])).max() > 25):
                run.clear()
            run.append({"tick": tick, "bearing": d["bearing"], "rgb": d["rgb"], "from": list(cam_xy)})
            if len(run) >= self.P["lamp_confirm"]:
                self.lamp.update(state="on", bearing=d["bearing"], last_on=tick, **{"from": list(cam_xy)})
                if self.grid is not None:          # where on the wall: first mapped obstacle along the bearing
                    occ = self.grid.occupied()
                    u = np.array([np.cos(d["bearing"]), np.sin(d["bearing"])])
                    for r in np.arange(0.3, 3.2, 0.05):
                        c = self.grid.idx(np.asarray(cam_xy) + r * u)[0]
                        if occ[c[1], c[0]]:
                            self.lamp["xy"] = (np.asarray(cam_xy) + r * u).tolist()
                            break
        else:
            run.clear()

    def lamp_state(self, since=-1):
        """Latest confirmed lamp sighting since tick `since`: ("on"|"off", tick, xy) or None."""
        best = None
        for t in self.tracks:
            if t["cls"] in ("lamp_on", "lamp_off") and t["n"] >= self.P["confirm"] and t["consec"] >= self.P["confirm"] \
                    and t["last"] > since:
                if best is None or t["last"] > best[1]:
                    best = (t["cls"][5:], t["last"], t["xy"])
        return best

    def lamp_xy(self):
        ts = [t for t in self.tracks if t["cls"] in ("lamp_on", "lamp_off") and t["n"] >= self.P["confirm"]]
        if not ts:
            return None
        w = np.array([t["w"] for t in ts])
        return (np.array([t["xy"] for t in ts]) * w[:, None]).sum(0) / w.sum()

    def to_dict(self):
        return {"tracks": self.tracks, "lamp": {k: v for k, v in self.lamp.items() if k != "run"},
                "ball_trail": self.ball_trail[-50:], "visited": self.visited}


class RefBot:
    def __init__(self, params=None):
        self.P = dict(PARAMS, **(params or {}))
        self.body = BodyModel()
        self.odo = Odometry(self.body)
        self.floor = FloorModel()
        self.grid = OccupancyGrid()
        self.mem = Memory(self.P, self.grid)
        self.corrections = []
        self.skill_log = []
        self.skill_stack = []
        self.gaze = Gaze()
        self.last_time = None
        self.learning_floor = False
        self.buffer = []          # frames seen while the floor model was being learned
        self.tick = 0
        self.cam = None
        self.dets = []
        self.log = []           # (tick, event)

    # ------------------------------------------------------------------ sensing (every tick)
    def sense(self, obs):
        self.tick = obs.tick
        self.odo.update(obs)
        self.cam = CameraPose(self.body, obs, self.odo.pose())
        self.gaze.update(obs, self.cam.mat)
        self.last_time = obs.sim_time
        frame = obs.frame()
        self.dets = detect(frame, self.cam)
        res = self.mem.update(self.dets, self.tick, self.cam.pos[:2])
        if res and self.P["anchor_gain"] > 0:
            r = np.mean(res, 0)
            if np.linalg.norm(r) < self.P["anchor_max_resid"]:
                step = self.P["anchor_gain"] * r
                n = np.linalg.norm(step)
                if n > self.P["anchor_max_step"]:
                    step *= self.P["anchor_max_step"] / n
                self.odo.x += float(step[0])
                self.odo.y += float(step[1])
                self.corrections.append(float(np.linalg.norm(step)))
        self.mem.lamp_update(self.dets, self.tick, self.cam.pos[:2])
        if self.learning_floor:
            self.floor.add_samples(frame, self.cam, walkable_mask(frame))
            if self.tick % self.P["map_every"] == 0:
                self.buffer.append((frame, self.cam))
        elif self.floor.centres is not None and self.tick % self.P["map_every"] == 0:
            self.grid.update(frame, self.cam, self.floor)

    def stabilise(self, cmd):
        """The chicken-head reflex: add the gaze counter-swing to the head command (when enabled)."""
        if not self.P["gaze"] or self.last_time is None:
            return cmd
        oy, op = self.gaze.offsets(self.last_time)
        h = list(cmd.head)
        return Command(vx=cmd.vx, vy=cmd.vy, wz=cmd.wz, head=(h[0], h[1] + op, h[2] + oy, h[3]), done=cmd.done)

    def event(self, what, **kw):
        self.log.append({"tick": self.tick, "event": what, **kw})

    def pose(self):
        return self.odo.pose()

    def is_free(self, xy):
        from scipy import ndimage
        occ = ndimage.binary_dilation(self.grid.occupied(), iterations=3)
        c = self.grid.idx(xy)[0]
        return not occ[c[1], c[0]]

    def plan(self, start, goal):
        """The bot's planner: A* on its own map; with no path (often a spurious obstacle) walk straight
        and let go_to's stall detection decide."""
        wps = self.grid.plan(start, goal, blocked=self.no_go())
        return wps if wps is not None else [list(map(float, goal))]

    def no_go(self):
        """Walking over the plate presses it: plan around the button (press_button crosses it on its
        own straight line, not through the planner)."""
        b = self.mem.best("button")
        return [(b["xy"], self.P["button_avoid_r"])] if b is not None else []

    def ctx(self, iface):
        return Ctx(iface, self.pose, planner=self.plan, is_free=self.is_free)

    def drive(self, skill):
        """Run a skill, sensing on every tick before the skill sees the observation. Each finished skill's
        report is logged (name, start/end tick, ok, reason) for the M4 recordings; bookkeeping only."""
        start = self.tick
        self.skill_stack.append(getattr(skill, "__name__", "?"))
        try:
            cmd = next(skill)
            while True:
                obs = yield cmd
                if obs is not None:
                    self.sense(obs)
                cmd = skill.send(obs)
        except StopIteration as e:
            rep = e.value if isinstance(e.value, dict) else {}
            self.skill_log.append({"skill": getattr(skill, "__name__", "?"), "start": start, "end": self.tick,
                                   "ok": rep.get("ok"), "reason": rep.get("reason")})
            self.skill_stack.pop()
            return e.value

    def activity(self):
        """The chain of generators currently running, task first (e.g. set_lamp > find > scan_here >
        drive > _single): read for the M4 recordings, bookkeeping only."""
        out, g = [], getattr(self, "task_gen", None)
        while g is not None:
            out.append(g.gi_code.co_name)
            g = g.gi_yieldfrom
        return out

    # -------------------------------------------------------------------- building blocks
    def learn_floor(self, ctx, stop_when=None):
        """One 360 deg look-around with the head tilted down: it sees the guaranteed-floor ring near the
        feet (floor model) and the floor out to ~1.7 m (map). Frames are buffered, the floor model is
        fitted, then the buffered frames are replayed into the occupancy grid. Returns stop_when()."""
        self.learning_floor = True
        hit = yield from self.scan_here(ctx, pitch=self.P["learn_pitch"], stop_when=stop_when)
        self.learning_floor = False
        ok = self.floor.fit()
        if ok:
            for fr, cam in self.buffer:
                self.grid.update(fr, cam, self.floor)
        self.buffer = []
        self.event("floor_model", ok=ok, clusters=0 if self.floor.centres is None else len(self.floor.centres))
        return hit

    def scan_here(self, ctx, full=True, pitch=None, stop_when=None):
        """Look around: one continuous in-place turn (360 deg, or a head sweep only if not full) with the
        head at `pitch`, perceiving every tick. Returns True if stop_when() fired."""
        pitch = self.P["scan_pitch"] if pitch is None else pitch
        x, y, _ = self.pose()
        self.mem.visited.append([x, y])
        if not full:
            for hy in (-1.1, -0.55, 0.0, 0.55, 1.1):
                yield from self.drive(look(ctx, (0.0, pitch, hy, 0.0)))
                if stop_when is not None and stop_when():
                    return True
            yield from self.drive(look(ctx, (0.0, pitch, 0.0, 0.0)))
            return False
        ctx.head = (0.0, pitch, 0.0, 0.0)
        for _ in range(4):                                     # settle the head first
            yield from self.drive(stop(ctx, max_ticks=1))
        turned, last = 0.0, self.pose()[2]
        while turned < 2 * np.pi - 0.15 and self.tick < 100000:
            yield from self.drive(_single(ctx.cmd(wz=self.P["scan_wz"])))
            yaw = self.pose()[2]
            turned += abs(wrap(yaw - last))
            last = yaw
            if stop_when is not None and stop_when():
                yield from self.drive(stop(ctx))
                return True
        yield from self.drive(stop(ctx))
        return False

    def frontier_points(self):
        """Free cells next to unknown cells (the edge of what has been mapped), clustered; returns
        cluster centres the planner can reach, nearest first."""
        from scipy import ndimage
        L = self.grid.L
        known_free = L < -0.5
        unknown = np.abs(L) <= 0.5
        front = known_free & ndimage.binary_dilation(unknown, iterations=1)
        front &= ~ndimage.binary_dilation(self.grid.occupied(), iterations=3)
        lab, n = ndimage.label(front, structure=np.ones((3, 3)))
        x, y, _ = self.pose()
        out = []
        for k in range(1, n + 1):
            rr, cc = np.nonzero(lab == k)
            if len(rr) < 4:
                continue
            i = len(rr) // 2
            order = np.argsort(rr * 1000 + cc)
            r0, c0 = rr[order[i]], cc[order[i]]
            from botmap import CELL, HALF
            p = np.array([c0 * CELL - HALF + CELL / 2, r0 * CELL - HALF + CELL / 2])
            # stay inside the 3 x 3 m room prior (the start is somewhere in it)
            if np.hypot(*p) > 2.0 or any(np.linalg.norm(p - np.array(v)) < 0.5 for v in self.mem.visited):
                continue
            if self.grid.plan((x, y), p) is None:
                continue
            out.append((len(rr), float(np.hypot(*(p - [x, y]))), p))
        out.sort(key=lambda t: t[1] - 0.02 * t[0])          # near first, bigger frontiers slightly preferred
        return [t[2] for t in out]

    def go(self, ctx, target_fn, stop_radius=0.08, max_ticks=600, pitch=None):
        """Walk to a target that may be refined as it comes into view: re-plan to target_fn() every
        servo_ticks. Returns the final distance to the latest estimate."""
        pitch = self.P["approach_pitch"] if pitch is None else pitch
        ctx.head = (0.0, pitch, 0.0, 0.0)
        t0 = self.tick
        while self.tick - t0 < max_ticks:
            tgt = np.asarray(target_fn(), float)
            x, y, _ = self.pose()
            if np.hypot(*(tgt - [x, y])) < stop_radius:
                break
            wps = self.grid.plan((x, y), tgt, blocked=self.no_go())
            wps = wps[:-1] if wps else []
            rep = yield from self.drive(go_to(ctx, tgt, stop_radius=stop_radius, waypoints=wps,
                                              max_ticks=self.P["servo_ticks"], final_stop=False))
            if rep["reason"] == "blocked":
                self.event("blocked", target=tgt.tolist())
        yield from self.drive(stop(ctx))
        x, y, _ = self.pose()
        return float(np.hypot(*(np.asarray(target_fn()) - [x, y])))

    def explore_points(self):
        """Unvisited free points on two rings around the start (the room is 3 x 3 m and the start
        is inside it), that the planner can reach."""
        pts = [np.array([r * np.cos(a), r * np.sin(a)]) for r in (0.7, 1.3)
               for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
        seen = [np.array(v) for v in self.mem.visited]
        x, y, _ = self.pose()
        out = [p for p in pts if self.is_free(p) and all(np.linalg.norm(p - v) > 0.5 for v in seen)
               and self.grid.plan((x, y), p) is not None]
        return out

    def find(self, ctx, cls, budget_tick):
        """Scan here; if `cls` isn't confirmed, visit exploration points and scan until it is."""
        if self.mem.best(cls) is None:
            yield from self.scan_here(ctx, stop_when=lambda: self.mem.best(cls) is not None)
        while self.mem.best(cls) is None and self.tick < budget_tick:
            pts = self.frontier_points() or self.explore_points()
            if not pts:
                break
            x, y, _ = self.pose()
            p = min(pts, key=lambda q: np.hypot(*(q - [x, y])))
            yield from self.go(ctx, lambda p=p: p, stop_radius=0.15, max_ticks=300, pitch=self.P["scan_pitch"])
            yield from self.scan_here(ctx, stop_when=lambda: self.mem.best(cls) is not None)
        return self.mem.best(cls)

    # ------------------------------------------------------------------------------- tasks
    def go_and_return(self, iface, instruction, budget):
        ctx = self.ctx(iface)
        yield from self.learn_floor(ctx)
        cls = "pad_" + instruction["target"]
        track = yield from self.find(ctx, cls, budget - 500)
        if track is None:
            self.event("target_not_found")
        else:
            d = yield from self.go(ctx, lambda: self.mem.best(cls)["xy"], stop_radius=0.07)
            self.event("at_pad", dist=d)
            for _ in range(5):
                yield from self.drive(stop(ctx, max_ticks=3))
        home = lambda: (self.mem.best("mat", near=np.zeros(2)) or {"xy": [0.0, 0.0]})["xy"]  # noqa: E731
        d = yield from self.go(ctx, home, stop_radius=0.07)
        self.event("home", dist=d, used_mat=self.mem.best("mat", near=np.zeros(2)) is not None)
        yield ctx.cmd(done=True)

    def look_at_lamp(self, ctx, frm=None):
        """Turn towards the remembered lamp (walking to `frm` first if given) and hold still for a few
        ticks. Returns the fresh confirmed state ("on"/"off") or None."""
        start = self.tick
        if frm is not None:
            yield from self.go(ctx, lambda: np.asarray(frm), stop_radius=0.12, max_ticks=250, pitch=0.0)
        lxy = self.mem.lamp_xy()
        if lxy is None:
            return None
        x, y, _ = self.pose()
        yield from self.drive(turn_to(ctx, np.arctan2(lxy[1] - y, lxy[0] - x), tol=0.2))
        for hy in (0.0, 0.35, -0.35):
            yield from self.drive(look(ctx, (0.0, 0.0, hy, 0.0)))
            for _ in range(self.P["confirm"] + 1):
                yield from self.drive(stop(ctx, max_ticks=1))
            st = self.mem.lamp_state(since=start)
            if st is not None:
                return st[0]
        return None

    def set_lamp(self, iface, instruction, budget):
        ctx = self.ctx(iface)
        target = instruction["target"]
        yield from self.learn_floor(ctx)              # also the first lamp check (on or off)
        seen_from = None
        st = self.mem.lamp_state()
        while st is None and self.tick < budget - 300:   # lamp not seen yet: explore until it is
            pts = self.frontier_points() or self.explore_points()
            if not pts:
                break
            x, y, _ = self.pose()
            p = min(pts, key=lambda q: np.hypot(*(q - [x, y])))
            yield from self.go(ctx, lambda p=p: p, stop_radius=0.15, max_ticks=250, pitch=0.0)
            yield from self.scan_here(ctx, pitch=0.0, stop_when=lambda: self.mem.lamp_state() is not None)
            st = self.mem.lamp_state()
        if st is not None:
            seen_from = list(self.pose()[:2])
        self.event("lamp_check", state=None if st is None else st[0])
        state = None if st is None else st[0]
        tries = 0
        while state != target and tries < self.P["max_press_tries"] and self.tick < budget - 200:
            tries += 1
            button = yield from self.find(ctx, "button", budget - 250)
            if button is None:
                self.event("button_not_found")
                break
            # walk to a spot 0.6 m before the plate looking down at it, so its position is fresh
            # (odometry drift can make a stale estimate miss the 24 cm plate)
            x, y, _ = self.pose()
            bxy = np.array(self.mem.best("button")["xy"])
            u = (bxy - [x, y]) / max(np.linalg.norm(bxy - [x, y]), 1e-6)
            look_pt = bxy - 0.6 * u
            if self.is_free(look_pt):
                yield from self.go(ctx, lambda: look_pt, stop_radius=0.12, max_ticks=250, pitch=0.45)
                x, y, _ = self.pose()
                b2 = np.array(self.mem.best("button")["xy"])
                yield from self.drive(turn_to(ctx, np.arctan2(b2[1] - y, b2[0] - x), tol=0.2))
                for _ in range(4):
                    yield from self.drive(stop(ctx, max_ticks=1))
            ctx.head = (0.0, 0.45, 0.0, 0.0)
            rep = yield from self.drive(press_button(ctx, self.mem.best("button")["xy"]))
            self.event("press", ok=rep["ok"], reason=rep["reason"])
            for _ in range(12):                     # off the plate, let it re-arm
                yield from self.drive(stop(ctx, max_ticks=1))
            new = yield from self.look_at_lamp(ctx)
            if new is None and seen_from is not None and self.tick < budget - 120:
                new = yield from self.look_at_lamp(ctx, frm=seen_from)   # hidden from here: go where it was seen
            self.event("lamp_check", state=new)
            if new is not None:
                state = new
            elif state is not None:
                state = "on" if state == "off" else "off"    # assume the press worked if it can't be seen
        yield ctx.cmd(done=True)

    def find_ball(self, iface, instruction, budget):
        ctx = self.ctx(iface)
        P = self.P
        # 1. watch the roll: keep the head on the ball until it has been out of sight a while
        last_seen, lost = None, 0
        hy = 0.0
        for _ in range(80):
            obs = yield ctx.cmd(head=(0.0, 0.1, hy, 0.0))
            self.sense(obs)
            balls = [d for d in self.dets if d["cls"] == "ball" and not d["border"]] or \
                    [d for d in self.dets if d["cls"] == "ball"]
            if last_seen is not None:
                balls = [d for d in balls if np.linalg.norm(np.array(d["xy"]) - last_seen[1]) < 0.6]
            if balls:
                b = max(balls, key=lambda d: d["area"]) if last_seen is None else \
                    min(balls, key=lambda d: np.linalg.norm(np.array(d["xy"]) - last_seen[1]))
                last_seen, lost = (self.tick, b["xy"]), 0
                x, y, yaw = self.pose()
                bearing = np.arctan2(b["xy"][1] - y, b["xy"][0] - x)
                hy = float(np.clip(wrap(bearing - yaw), -1.1, 1.1))
            else:
                lost += 1
                if last_seen is not None and lost >= P["ball_lost_ticks"] + 10:
                    break
        trail = [(t, np.array(p)) for t, p in self.mem.ball_trail]
        if last_seen is None:
            self.event("ball_never_seen")
            guess = None
        else:
            # where did it go? last sighting plus the recent direction of travel
            pts = [p for t, p in trail if t >= last_seen[0] - 6]
            v = (pts[-1] - pts[0]) if len(pts) >= 2 else np.zeros(2)
            v = v / max(np.linalg.norm(v), 1e-6)
            guess = np.array(last_seen[1]) + 0.3 * v
            self.event("ball_lost", last=last_seen[1], direction=v.tolist(), guess=guess.tolist())
        # 2. look around once: learns the floor model and maps the obstacles (viewpoints need them);
        #    stops early if the ball comes into view
        found = lambda: self.ball_now(full=True) is not None  # noqa: E731
        yield from self.learn_floor(ctx, stop_when=found)
        # 3. viewpoints of the guessed spot, re-chosen from the current map each time
        targets = [guess] if guess is not None else []
        tried = 0
        while targets and self.tick < budget - 60 and not found():
            tgt = targets.pop(0)
            for attempt in range(3):
                vps = [v for v in self.viewpoints(tgt)
                       if all(np.linalg.norm(v - np.array(u)) > 0.2 for u in self.mem.visited)]
                if not vps or self.tick > budget - 60:
                    break
                vp = vps[0]
                self.mem.visited.append(vp.tolist())
                tried += 1
                yield from self.go(ctx, lambda vp=vp: vp, stop_radius=0.08, max_ticks=300, pitch=P["scan_pitch"])
                x, y, _ = self.pose()
                yield from self.drive(turn_to(ctx, np.arctan2(tgt[1] - y, tgt[0] - x), tol=0.2))
                for head in ((0.0, 0.3, 0.0, 0.0), (0.0, 0.3, 0.5, 0.0), (0.0, 0.3, -0.5, 0.0), (0.0, 0.0, 0.0, 0.0)):
                    yield from self.drive(look(ctx, head))
                    if found():
                        break
                if found():
                    break
            if found():
                break
            if not targets and self.tick < budget - 150:
                yield from self.scan_here(ctx, stop_when=found)
                if found():
                    break
        if found():
            bn = self.ball_now(full=True) or self.ball_now()
            b = np.array(bn["xy"]) if bn is not None else np.array(self.mem.best("ball")["xy"])
            x, y, _ = self.pose()
            if np.hypot(*(b - [x, y])) > 0.75:          # get within the judge's 0.8 m
                yield from self.go(ctx, lambda: b + 0.65 * (np.array(self.pose()[:2]) - b) / max(np.linalg.norm(np.array(self.pose()[:2]) - b), 1e-6),
                                   stop_radius=0.08, max_ticks=200)
            x, y, _ = self.pose()
            yield from self.drive(turn_to(ctx, np.arctan2(b[1] - y, b[0] - x), tol=0.15))
            # centre the ball with the head before stopping (the judge wants it in view at done)
            hy, hp = 0.0, 0.3
            for _ in range(25):
                yield from self.drive(_single(ctx.cmd(head=(0.0, hp, hy, 0.0))))
                bs = [d for d in self.dets if d["cls"] == "ball"]
                if bs:
                    d = max(bs, key=lambda q: q["area"])
                    ei, ej = (d["px"][0] - 63.5) / 64, (d["px"][1] - 63.5) / 64   # rows: down, cols: right
                    if abs(ei) < 0.35 and abs(ej) < 0.35:
                        break
                    hp = float(np.clip(hp + 0.25 * ei, -0.3, 0.8))
                    hy = float(np.clip(hy - 0.25 * ej, -1.1, 1.1))
                else:
                    hy = float(np.clip(-hy if abs(hy) > 0.2 else hy + 0.3, -1.1, 1.1))
            self.event("ball_found", xy=b.tolist(), centred=bool(bs))
        yield from self.drive(stop(ctx))
        yield ctx.cmd(done=True)

    def ball_now(self, full=False):
        """The ball as seen this tick, if it belongs to a track seen on >= confirm consecutive ticks;
        with full=True only if the blob is about the size of a whole ball at that distance (a sliver
        peeking past an edge doesn't count: its centre is likely hidden)."""
        for t in self.mem.tracks:
            if t["cls"] != "ball" or t["last"] != self.tick or t["consec"] < self.P["confirm"]:
                continue
            for d in self.dets:
                if d["cls"] == "ball" and np.linalg.norm(np.array(d["xy"]) - t["xy"]) < 0.05:
                    if full and (d["area"] < self.P["ball_min_area_frac"] * d["full_area"] or d["border"]):
                        continue
                    return d
        return None

    def viewpoints(self, target):
        """Free points 0.55-0.75 m from target with a clear line of sight in the bot's own map."""
        out = []
        occ = self.grid.occupied()
        for r in self.P["ball_view_radius"]:
            for a in np.linspace(-np.pi, np.pi, 16, endpoint=False):
                p = np.asarray(target) + r * np.array([np.cos(a), np.sin(a)])
                if not self.is_free(p):
                    continue
                clear = True
                for t in np.linspace(0.1, 0.9, 12):
                    c = self.grid.idx(p + t * (np.asarray(target) - p))[0]
                    if occ[c[1], c[0]]:
                        clear = False
                        break
                if clear:
                    out.append(p)
        x, y, _ = self.pose()
        out.sort(key=lambda p: np.hypot(*(p - [x, y])))
        return out


def controller(task, budget, params=None):
    """A run_episode-compatible controller factory. Returns (factory, bot) so callers can read the
    bot's memory and log afterwards."""
    bot = RefBot(params)

    def factory(iface, instruction):
        gen = getattr(bot, task)(iface, instruction, budget)
        bot.task_gen = gen
        first = True
        try:
            cmd = next(gen)
            while True:
                bot.proposed = cmd                       # the bot's own command (recorded as the label)
                obs = yield bot.stabilise(cmd)           # executed: plus the gaze reflex if enabled
                if first or obs is None:
                    first = False
                # sensing happens inside the bot's drive(); find_ball's watch loop senses itself
                cmd = gen.send(obs)
        except StopIteration:
            return
    return factory, bot
