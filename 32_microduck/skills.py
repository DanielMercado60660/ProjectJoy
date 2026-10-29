"""Skills on the frozen command interface. Each skill is a 10 Hz state machine (a generator that
yields Commands and receives Observations) and ends by returning a report:

    {"skill", "requested", "achieved", "ok", "reason", "ticks", ...}

Skills never touch the World: they act only through Interface.tick() and read their position from a
pose hook, ctx.pose() -> (x, y, yaw). In M2 the hook is TruePose (simulator ground truth: an ORACLE
pose); M3's reference bot plugs its own odometry into the same hook. Skills cannot observe whether the
button registered or where the ball is: they report what they did ("walked across the plate") and
the expected outcome; checking the outcome is the caller's job.

Body facts the skills work around (M0): no gait below ~0.2 m/s from standing, in-place turns need
|wz| >= 0.5 and come in ~0.27 rad steps with a variable coast, achieved speeds are not the commanded
ones, and the duck curves left when commanded straight. So every motion closes the loop on pose.
"""
from collections import deque

import numpy as np

from configs import DUCK_CLEARANCE, GRID, _point_free, footprints
from interface import BOUNDS, Command

WALK_VX = 0.3          # the gait-tested walking command (achieves ~0.13 m/s)
TURN_WZ = 0.55         # in-place turn command (starts from standing; ~0.8 rad/s achieved)
STEER_GAIN = 1.5       # wz per rad of heading error while walking (as m0_gate.goto_command)
ALIGN = 0.5            # turn in place first if the heading error exceeds this (rad)
TURN_TOL = 0.20        # turn_to success tolerance (rad)
TURN_LEAD = 0.15       # stop turning this far before the target (coast)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class TruePose:
    """ORACLE pose hook: the duck's true planar pose from the simulator."""
    label = "oracle_true_pose"

    def __init__(self, world):
        self.world = world

    def __call__(self):
        p = self.world.duck.pos()
        return float(p[0]), float(p[1]), self.world.duck.yaw()


class Ctx:
    """What a skill can use: the interface, a pose hook and the current head command.

    `ctx.obs` is always the observation of the interface's current tick, whoever drives the skill
    generator (skills.run, tasks.run_episode, a recorder that sends None). Skills discard the value
    their `yield` returns, so before the M2 review fix `ctx.obs` stayed at tick 0 under run_episode:
    look/scan read stale head encoders and stop ignored the gyro."""

    def __init__(self, iface, pose, layout=None, planner=None, is_free=None):
        """`planner(start, goal) -> waypoints | None` and `is_free(xy) -> bool` are the caller's map.
        The oracle passes the true layout (both are built from it); the reference bot passes its own
        occupancy grid. Without either, skills walk straight lines."""
        self.iface, self.pose, self.layout = iface, pose, layout
        if layout is not None:
            X, Y, obst = footprints(layout, pad=DUCK_CLEARANCE)
            planner = planner or (lambda s, g: plan_path(layout, s, g))
            is_free = is_free or (lambda xy: _point_free(X, Y, obst, np.asarray(xy, float)))
        self.planner, self.is_free = planner, is_free
        self.head = (0.0, 0.0, 0.0, 0.0)
        self._obs = iface.observe()

    @property
    def obs(self):
        if self._obs is None or self._obs.tick != self.iface.tick_count:
            self._obs = self.iface.observe()
        return self._obs

    @obs.setter
    def obs(self, o):
        self._obs = o

    def cmd(self, vx=0.0, wz=0.0, head=None, done=False):
        if head is not None:
            self.head = tuple(head)
        return Command(vx=vx, wz=wz, head=self.head, done=done)


def run(ctx, skill):
    """Drive a skill generator to completion through the interface; return its report."""
    try:
        cmd = next(skill)
        while True:
            ctx.obs = ctx.iface.tick(cmd)
            cmd = skill.send(ctx.obs)
    except StopIteration as e:
        return e.value


def _report(skill, requested, achieved, ok, reason, ticks, **extra):
    return {"skill": skill, "requested": requested, "achieved": achieved, "ok": bool(ok),
            "reason": reason, "ticks": int(ticks), **extra}


# ---------------------------------------------------------------------------------------- stop
def stop(ctx, settle_ticks=5, max_ticks=30):
    """Zero twist until the body is still (gyro and pose change small) for settle_ticks."""
    calm, last, t = 0, ctx.pose(), 0
    while t < max_ticks:
        yield ctx.cmd()
        t += 1
        p = ctx.pose()
        moved = np.hypot(p[0] - last[0], p[1] - last[1])
        spin = float(np.abs(ctx.obs.body["gyro"]).max())
        calm = calm + 1 if (moved < 0.004 and spin < 0.3) else 0
        last = p
        if calm >= settle_ticks:
            return _report("stop", {}, {"pose": list(p)}, True, "settled", t)
    return _report("stop", {}, {"pose": list(ctx.pose())}, False, "not settled", t)


# ------------------------------------------------------------------------------------- turn_to
def turn_to(ctx, bearing, tol=TURN_TOL, max_attempts=6, max_ticks=150):
    """Turn in place to a world bearing: turn, stop short by TURN_LEAD, settle, re-check, correct."""
    t, attempts = 0, 0
    while attempts < max_attempts and t < max_ticks:
        err = wrap(bearing - ctx.pose()[2])
        if abs(err) <= tol:
            break
        attempts += 1
        sign = np.sign(err)
        burst = 0
        # at least 6 ticks: shorter pulses spring back to nothing (results/m2 battery)
        while (burst < 6 or sign * wrap(bearing - ctx.pose()[2]) > TURN_LEAD) and t < max_ticks:
            yield ctx.cmd(wz=sign * TURN_WZ)
            t += 1
            burst += 1
            if burst >= 6 and sign * wrap(bearing - ctx.pose()[2]) < 0:   # passed it
                break
        rep = yield from stop(ctx)
        t += rep["ticks"]
    err = wrap(bearing - ctx.pose()[2])
    return _report("turn_to", {"bearing": float(bearing)}, {"yaw": ctx.pose()[2], "error": float(err)},
                   abs(err) <= tol, "within tolerance" if abs(err) <= tol else "out of tolerance",
                   t, attempts=attempts)


# ------------------------------------------------------------------------------- path planning
def plan_path(layout, start, goal, clearance=DUCK_CLEARANCE):
    """A* on the 2 cm occupancy grid (walls, partition, occluder grown by `clearance`), then
    line-of-sight shortcutting. Returns waypoints (excluding start, including goal) or None."""
    import heapq
    X, Y, obst = footprints(layout, pad=clearance)
    n = X.shape[0]
    idx = lambda xy: (int(np.clip((xy[1] - Y[0, 0]) / GRID + 0.5, 0, n - 1)),  # noqa: E731
                      int(np.clip((xy[0] - X[0, 0]) / GRID + 0.5, 0, n - 1)))
    s, g = idx(start), idx(goal)
    free = ~obst
    free[s] = free[g] = True
    h = lambda a: np.hypot(a[0] - g[0], a[1] - g[1])  # noqa: E731
    openq, came, cost = [(h(s), 0.0, s)], {s: None}, {s: 0.0}
    steps = [(1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1), (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
    while openq:
        _, c, cur = heapq.heappop(openq)
        if cur == g:
            break
        if c > cost[cur]:
            continue
        for dr, dc, w in steps:
            nb = (cur[0] + dr, cur[1] + dc)
            if 0 <= nb[0] < n and 0 <= nb[1] < n and free[nb]:
                nc = c + w
                if nc < cost.get(nb, 1e18):
                    cost[nb], came[nb] = nc, cur
                    heapq.heappush(openq, (nc + h(nb), nc, nb))
    if g not in came:
        return None
    if s == g:                       # already in the goal's grid cell (M2 review: used to IndexError)
        return [np.asarray(goal, float).round(3).tolist()]
    cells = []
    cur = g
    while cur is not None:
        cells.append(cur)
        cur = came[cur]
    cells.reverse()
    pts = [np.array([X[0, c], Y[r, 0]]) for r, c in cells]

    def clear(a, b):
        k = int(np.hypot(*(b - a)) / (GRID / 2)) + 2
        for t in np.linspace(0, 1, k):
            r, c = idx(a + t * (b - a))
            if obst[r, c]:
                return False
        return True

    out, i = [], 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not clear(pts[i], pts[j]):
            j -= 1
        out.append(pts[j])
        i = j
    out[-1] = np.asarray(goal, float)
    return [p.round(3).tolist() for p in out]


# --------------------------------------------------------------------------------------- go_to
def go_to(ctx, target, stop_radius=0.10, waypoints=None, allow_turn_in_place=True,
          vx=WALK_VX, max_ticks=900, stall_ticks=40, final_stop=True):
    """Walk to `target` (world x, y) through optional waypoints, steering on pose; stop at the end.
    Fails as 'blocked' if the distance to the current waypoint hasn't shrunk 5 cm in stall_ticks."""
    pts = [np.asarray(p, float) for p in (waypoints or [])] + [np.asarray(target, float)]
    t, path_len, last = 0, 0.0, np.array(ctx.pose()[:2])
    reason, ok = "reached", True
    for k, wp in enumerate(pts):
        final = k == len(pts) - 1
        radius = stop_radius if final else 0.15
        best, best_t = 1e9, t
        while True:
            x, y, yaw = ctx.pose()
            d = float(np.hypot(*(wp - [x, y])))
            if d < radius:
                break
            if d < best - 0.05:
                best, best_t = d, t
            if t - best_t > stall_ticks:
                reason, ok = "blocked", False
                break
            if t >= max_ticks:
                reason, ok = "timeout", False
                break
            err = wrap(np.arctan2(wp[1] - y, wp[0] - x) - yaw)
            if allow_turn_in_place and abs(err) > ALIGN:
                rep = yield from turn_to(ctx, np.arctan2(wp[1] - y, wp[0] - x), tol=0.3)
                t += rep["ticks"]
                best_t = t   # turning isn't stalling
                continue
            yield ctx.cmd(vx=vx, wz=float(np.clip(STEER_GAIN * err, *BOUNDS["wz"])))
            t += 1
            p = np.array(ctx.pose()[:2])
            path_len += float(np.hypot(*(p - last)))
            last = p
        if not ok:
            break
    if final_stop:
        rep = yield from stop(ctx)
        t += rep["ticks"]
    x, y, yaw = ctx.pose()
    dist = float(np.hypot(*(pts[-1] - [x, y])))
    return _report("go_to", {"target": [float(v) for v in target], "waypoints": len(pts) - 1},
                   {"pose": [x, y, yaw], "distance": dist, "path_length": path_len},
                   ok and dist <= stop_radius + 0.05, reason, t)


def go_to_planned(ctx, target, **kw):
    """go_to around obstacles using the caller's planner (ctx.planner: the true layout for the oracle,
    the bot's own occupancy grid for the reference bot); a straight line without one."""
    x, y, _ = ctx.pose()
    wps = ctx.planner((x, y), target) if ctx.planner is not None else None
    if ctx.planner is not None and wps is None:
        return _report("go_to", {"target": list(target)}, {}, False, "no path", 0)
    rep = yield from go_to(ctx, target, waypoints=(wps or [])[:-1], **kw)
    return rep


# ---------------------------------------------------------------------------------------- look
def look(ctx, head_pose, settle_ticks=3, max_ticks=20):
    """Hold still and move the head; report the pose actually reached (from the encoders)."""
    req = tuple(float(v) for v in Command(head=head_pose).clipped().head)
    prev, calm, t = ctx.obs.head_joints(), 0, 0
    while t < max_ticks:
        yield ctx.cmd(head=req)
        t += 1
        cur = ctx.obs.head_joints()
        calm = calm + 1 if np.abs(cur - prev).max() < 0.01 else 0
        prev = cur
        if calm >= settle_ticks:
            break
    ach = ctx.obs.head_joints()
    err = ach - np.asarray(req)
    return _report("look", {"head": list(req)}, {"head": ach.round(4).tolist(), "error": err.round(4).tolist()},
                   bool(np.abs(err[2]) < 0.15), "settled" if calm >= settle_ticks else "still moving", t)


# ---------------------------------------------------------------------------------------- scan
SCAN_HEAD_YAWS = (-1.1, -0.55, 0.0, 0.55, 1.1)
HALF_FOV = np.radians(22.5)


def scan(ctx, full=False, on_frame=None, head_pitch=0.0):
    """Look around with the head (yaw stops +-1.1 rad, ~170 deg of view); with full=True also turn
    the body ~120 deg twice for 360 deg. on_frame(obs, camera_heading) is called at every stop.
    Reports the headings covered (in world terms, from the pose hook and the head encoders)."""
    views, t = [], 0
    sweeps = 3 if full else 1
    base = ctx.pose()[2]
    for s in range(sweeps):
        if s > 0:
            rep = yield from turn_to(ctx, wrap(base + s * 2 * np.pi / 3), tol=0.3)
            t += rep["ticks"]
        order = SCAN_HEAD_YAWS if s % 2 == 0 else SCAN_HEAD_YAWS[::-1]
        for hy in order:
            rep = yield from look(ctx, (0.0, head_pitch, hy, 0.0))
            t += rep["ticks"]
            heading = wrap(ctx.pose()[2] + ctx.obs.head_joints()[2])
            views.append({"camera_heading": float(heading), "head_yaw": float(ctx.obs.head_joints()[2]),
                          "tick": ctx.obs.tick})
            if on_frame is not None:
                on_frame(ctx.obs, heading)
    rep = yield from look(ctx, (0.0, 0.0, 0.0, 0.0))
    t += rep["ticks"]
    covered = _coverage([v["camera_heading"] for v in views])
    return _report("scan", {"full": full}, {"views": views, "covered_deg": covered},
                   covered >= (300 if full else 150), f"{covered:.0f} deg covered", t)


def _coverage(headings, bins=360):
    hit = np.zeros(bins, bool)
    for h in headings:
        lo, hi = np.degrees(h - HALF_FOV), np.degrees(h + HALF_FOV)
        for d in range(int(np.floor(lo)), int(np.ceil(hi))):
            hit[d % bins] = True
    return float(hit.sum())


# -------------------------------------------------------------------------------- press_button
def press_button(ctx, button_xy, approach=0.35, beyond=0.35):
    """Walk straight across the plate centre: go to a point `approach` before it, face it, walk to a
    point `beyond` after it without turning in place, then stop. Turning happens only off the plate
    (turning on its edge can press it again; M1 review). The skill can't see the press: it reports
    whether its path crossed the plate centre; the expected outcome is that the lamp toggles."""
    b = np.asarray(button_xy, float)
    x, y, _ = ctx.pose()
    d = b - [x, y]
    d = d / max(np.linalg.norm(d), 1e-9)
    if ctx.is_free is not None:
        # with a map: the approach direction closest to the straight one whose approach and exit
        # points are both free floor (a plate near a wall can't be crossed from every side)
        base = np.arctan2(d[1], d[0])
        for off in sorted(np.radians(np.arange(-180, 180, 15)), key=abs):
            dd = np.array([np.cos(base + off), np.sin(base + off)])
            if ctx.is_free(b - approach * dd) and ctx.is_free(b + beyond * dd):
                d = dd
                break
    pre, post = b - approach * d, b + beyond * d
    t = 0
    rep = yield from go_to_planned(ctx, pre, stop_radius=0.08)
    t += rep["ticks"]
    if not rep["ok"]:
        return _report("press_button", {"button": b.tolist()}, {"approach": rep}, False, "approach failed", t)
    rep = yield from turn_to(ctx, np.arctan2(d[1], d[0]))
    t += rep["ticks"]
    closest = 9.0
    track = deque(maxlen=2000)
    gen = go_to(ctx, post, stop_radius=0.10, allow_turn_in_place=False, final_stop=False, max_ticks=200)
    try:
        cmd = next(gen)
        while True:
            obs = yield cmd
            p = np.array(ctx.pose()[:2])
            track.append(p)
            closest = min(closest, float(np.linalg.norm(p - b)))
            cmd = gen.send(obs)
    except StopIteration as e:
        cross = e.value
    t += cross["ticks"]
    rep = yield from stop(ctx)
    t += rep["ticks"]
    crossed = closest < 0.08 and cross["ok"]
    return _report("press_button", {"button": b.tolist()},
                   {"closest_to_centre": closest, "beyond_reached": cross["ok"]},
                   crossed, "crossed the plate centre" if crossed else "missed the plate", t,
                   expected_outcome="lamp toggles")


# ------------------------------------------------------------------------------- return_home
def return_home(ctx, home_xy, mat_half=0.20):
    rep = yield from go_to_planned(ctx, home_xy, stop_radius=0.10)
    x, y, _ = ctx.pose()
    inside = abs(x - home_xy[0]) <= mat_half and abs(y - home_xy[1]) <= mat_half
    return _report("return_home", {"home": list(home_xy)}, {"pose": list(ctx.pose()), "inside_mat": inside},
                   rep["ok"] and inside, rep["reason"], rep["ticks"])
