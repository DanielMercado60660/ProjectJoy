"""The three Phase 1 tasks: episode setup, the structured instruction, and ground-truth judges.

An episode is fully determined by (config, task, k): make_episode() derives everything else from a
seed built from the config hash, so episodes are reproducible and stay inside their config's split.
The instruction is structured in Phase 1: {"task": <id>, "target": <pad colour | "on"/"off" | null>}.

Judges read simulator truth (pose, lamp flag, presses, ball position, contacts). They are scorekeepers,
not controllers: a controller only gets the Interface observation and the instruction. Specs are in
tasks/<task>.md.
"""
import zlib

import mujoco
import numpy as np

from configs import footprints, _point_free
from interface import Interface
from room import World

TASKS = ("go_and_return", "set_lamp", "find_ball")
BUDGET_TICKS = {"go_and_return": 1200, "set_lamp": 900, "find_ball": 600}   # 120 / 90 / 60 s
FALL_TILT = np.radians(45)
PAD_MARGIN = 0.05        # "on the pad": trunk within the pad square grown by 5 cm
MAT_MARGIN = 0.05
STOPPED_SPEED = 0.05     # m/s planar speed counted as stopped at the end
BALL_FIND_DIST = 0.8     # task 3: ball within 0.8 m of the trunk at the end
BALL_REST_SPEED = 0.01   # task 3: the ball must have come to rest (m/s) and have been out of view
                         # at some tick: without these a controller that said done at tick 1 passed
                         # 54/240 training episodes, the launched ball still in view (M2 review)
HALF_FOV = np.radians(22.5)


def episode_rng(cfg, task, k):
    return np.random.default_rng([int(cfg["hash"], 16) % (2 ** 32), TASKS.index(task), int(k)])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------------------------- visibility
def ball_visibility(world):
    """Ground truth: is the ball's centre inside the head camera's 45 deg view with a clear line of
    sight? Rays ignore the robot's own geom groups (2, 3): the camera sits inside the head mesh."""
    m, d = world.model, world.duck.data
    c = d.cam_xpos[world.duck.cam_id].copy()
    R = d.cam_xmat[world.duck.cam_id].reshape(3, 3)
    b = world.ball_pos()
    v = R.T @ (b - c)                       # camera frame: looks along -z
    in_fov = v[2] < 0 and abs(v[0] / -v[2]) < np.tan(HALF_FOV) and abs(v[1] / -v[2]) < np.tan(HALF_FOV)
    gid = np.array([-1], np.int32)
    groups = np.array([1, 1, 0, 0, 0, 0], np.uint8)
    dist = mujoco.mj_ray(m, d, c, b - c, groups, 1, -1, gid)
    clear = dist >= 0 and gid[0] == world.ball_geom
    return bool(in_fov and clear), bool(clear)


# ---------------------------------------------------------------------------------- episodes
def make_episode(cfg, task, k=0):
    """Instruction and start conditions for episode k of `task` in this config."""
    rng = episode_rng(cfg, task, k)
    lay = cfg["layout"]
    ep = {"config_id": cfg["id"], "config_hash": cfg["hash"], "task": task, "k": int(k),
          "budget_ticks": BUDGET_TICKS[task], "start": None, "ball_launch": None}
    if task == "go_and_return":
        pad = lay["pads"][int(rng.integers(len(lay["pads"])))]
        ep["instruction"] = {"task": task, "target": pad["color"]}
    elif task == "set_lamp":
        initial = bool(lay["lamp_initial_on"])
        target = initial if rng.random() < 0.5 else not initial
        ep["instruction"] = {"task": task, "target": "on" if target else "off"}
    else:
        ep["instruction"] = {"task": task, "target": None}
        ep.update(_find_ball_setup(cfg, rng))
    return ep


SETUP_VERSION = 2   # v2 (M3 start): two hiders, varied ends, sides and rest points
MIN_WATCH_TICKS = 8  # the ball is in view for >= 0.8 s of the first 6 s after launch


def _hiders(lay):
    """Things a ball can hide behind: the occluder (both ends open) and the partition (its open ends;
    an end within 10 cm of a wall is closed)."""
    hx, hy = lay["room"]["size"][0] / 2, lay["room"]["size"][1] / 2
    out = []
    for name in ("occluder", "partition"):
        o = lay[name]
        if o is None:
            continue
        C = np.array(o["pos"], float)
        a = np.array([np.cos(o["yaw"]), np.sin(o["yaw"])])
        n = np.array([-np.sin(o["yaw"]), np.cos(o["yaw"])])
        L = o["length"] / 2
        ends = [e for e in (1.0, -1.0) if abs((C + e * L * a)[0]) < hx - 0.1 and abs((C + e * L * a)[1]) < hy - 0.1]
        if ends:
            out.append({"name": name, "C": C, "a": a, "n": n, "L": L, "ends": ends})
    return out


def ball_viewpoints(world, ball_xy, me_xy=None, radii=(0.55, 0.6, 0.7, 0.75)):
    """Free floor points 0.55-0.75 m from a floor position (ball height) with a clear line of sight at
    camera height from the point and 8 cm either side (the duck stops within ~8 cm). Closer rings first,
    then nearest to me_xy. "Clear" = the ray reaches the target point (or hits the ball sitting there). Used by the task-3 setup (solvability), the oracle and the no-memory baselines."""
    from configs import DUCK_CLEARANCE
    lay = world.cfg["layout"]
    X, Y, obst = footprints(lay, pad=DUCK_CLEARANCE + 0.03)
    m, d = world.model, world.duck.data
    me = np.zeros(2) if me_xy is None else np.asarray(me_xy, float)
    cands = []
    for r in radii:
        for ang in np.linspace(-np.pi, np.pi, 24, endpoint=False):
            p = np.asarray(ball_xy, float) + r * np.array([np.cos(ang), np.sin(ang)])
            if not _point_free(X, Y, obst, p):
                continue
            perp = np.array([-np.sin(ang), np.cos(ang)])
            ok = True
            for off in (0.0, 0.08, -0.08):
                for z in (0.22, 0.28):
                    src = np.array([*(p + off * perp), z])
                    tgt = np.array([*ball_xy, 0.035])
                    gid = np.array([-1], np.int32)
                    dist = mujoco.mj_ray(m, d, src, tgt - src, np.array([1, 1, 0, 0, 0, 0], np.uint8), 1, -1, gid)
                    # mj_ray returns the hit distance as a multiple of the direction vector, not metres
                    ok &= bool(dist < 0 or gid[0] == world.ball_geom or dist >= 1.0 - 0.005 / np.linalg.norm(tgt - src))
            if ok:
                cands.append((r, float(np.linalg.norm(p - me)), p))
    cands.sort(key=lambda c: (c[0] > 0.62, c[1]))
    return [c[2] for c in cands]


def _find_ball_setup(cfg, rng, n_draws=60):
    """Task 3 start (setup v2). Hiding place: behind the occluder OR behind an open end of the
    partition (each episode prefers one at random; the other is the fallback). The ball starts beside
    the chosen end on the duck's side and is launched at t=0 to roll round that end and stop at a
    randomly drawn point behind it (0.05-0.35 m back from the end, 0.22-0.45 m behind). The duck stands
    0.7-1.3 m away on the open side, facing between the ball and the end, and watches it go.
    Every setup is checked: the rolling path clears the end, the ball is visible at launch and hidden
    at rest, the launch speed (fitted by simulation) lands it within 8 cm of the drawn point, and some
    viewpoint with a clear view of the rest point is reachable from the start (so the episode is
    solvable without running any controller)."""
    from configs import DUCK_CLEARANCE, GRID, _reachable
    lay = cfg["layout"]
    hiders = _hiders(lay)
    order = list(rng.permutation(len(hiders)))
    X, Y, obst = footprints(lay, pad=0.20)
    _, _, obst_ball = footprints(lay, pad=0.06)
    _, _, obst_walk = footprints(lay, pad=DUCK_CLEARANCE)
    world = World(cfg)
    rejects = {}

    def rej(k):
        rejects[k] = rejects.get(k, 0) + 1

    for hi in order:
        h = hiders[hi]
        C, a, n, L = h["C"], h["a"], h["n"], h["L"]
        for _ in range(n_draws):
            e = float(rng.choice(h["ends"]))
            hs = float(rng.choice([1.0, -1.0]))            # the side the ball hides on
            E = C + e * L * a
            P0 = E + e * rng.uniform(0.30, 0.50) * a - hs * rng.uniform(0.12, 0.28) * n
            back = rng.uniform(0.05, min(0.35, 2 * L - 0.05))
            P1 = E - e * back * a + hs * rng.uniform(0.22, 0.45) * n
            V = E + e * rng.uniform(-0.3, 0.3) * a - hs * rng.uniform(0.7, 1.3) * n
            # the straight roll must pass the end with room for the ball (r = 3.5 cm)
            denom = (P1 - P0) @ n
            t_cross = -((P0 - C) @ n) / denom if abs(denom) > 1e-9 else -1
            if not 0 < t_cross < 1 or ((P0 + t_cross * (P1 - P0) - E) @ a) * e < 0.08:
                rej("path hits the end")
                continue
            if not (_point_free(X, Y, obst, V) and _point_free(X, Y, obst_ball, P0)
                    and _point_free(X, Y, obst_ball, P1)):
                rej("floor")
                continue
            aim = P0 + 0.35 * (E - P0)
            start = {"pos": V.round(3).tolist(), "yaw": round(float(np.arctan2(*(aim - V)[::-1])), 4)}
            world.reset(start=start)
            _place_ball(world, P0)
            if not ball_visibility(world)[0]:
                rej("not visible at launch")
                continue
            _place_ball(world, P1)
            if ball_visibility(world)[1]:
                rej("rest not hidden")
                continue
            vps = ball_viewpoints(world, P1)
            seen, idx = _reachable(*footprints(lay, pad=DUCK_CLEARANCE)[:2], ~obst_walk, V)
            if not any(seen[idx(vp)] for vp in vps):
                rej("no reachable view of the rest point")
                continue
            launch = _fit_launch(world, start, P0, P1)
            if launch["status"] != "ok":
                rej(launch["status"])
                continue
            # left or right as the duck sees it at the start (+ = to its left)
            fwd = np.array([np.cos(start["yaw"]), np.sin(start["yaw"])])
            rel = np.asarray(launch["rest"]) - V
            launch.update(setup_version=SETUP_VERSION, hider=h["name"], end=e, hidden_side=hs,
                          rest_planned=P1.round(4).tolist(),
                          rest_left_of_start_view=bool(fwd[0] * rel[1] - fwd[1] * rel[0] > 0))
            return {"start": start, "ball_launch": launch}
    raise RuntimeError(f"{cfg['id']}: no valid find_ball setup {rejects}")


def _settle_ball(world, max_steps=1000):
    """Step with the duck standing still until the ball stops; return (rest xy, steps)."""
    world.duck.set_command(0, 0, 0)
    for k in range(max_steps):
        world.step()
        v = world.duck.data.qvel[world.ball_vadr:world.ball_vadr + 3]
        if k > 25 and np.hypot(v[0], v[1]) < 0.003:
            return world.ball_pos()[:2], k + 1
    return world.ball_pos()[:2], max_steps


def _place_ball(world, xy):
    d = world.duck.data
    d.qpos[world.ball_qadr:world.ball_qadr + 3] = [xy[0], xy[1], 0.035]
    d.qpos[world.ball_qadr + 3:world.ball_qadr + 7] = [1, 0, 0, 0]
    d.qvel[world.ball_vadr:world.ball_vadr + 6] = 0
    mujoco.mj_forward(world.model, d)
    world.duck.has_pre_state = False


def start_episode(world, ep):
    """Reset the world to the episode's start: start pose, ball placed and launched if task 3."""
    world.reset(start=ep["start"])
    if ep["ball_launch"] is not None:
        L = ep["ball_launch"]
        _place_ball(world, L["from"])
        world.launch_ball(L["velocity"])


def _roll(world, start, P0, velocity):
    world.reset(start=start)
    _place_ball(world, P0)
    world.launch_ball(velocity)
    return _settle_ball(world)


def _fit_launch(world, start, P0, P1, tries=4):
    """Launch speed so the ball rests at P1: rolling distance grows smoothly with speed, so a few
    proportional corrections from 0.45 m/s converge. Then check: rest hidden from the viewpoint."""
    direction = (P1 - P0) / np.linalg.norm(P1 - P0)
    target = float(np.linalg.norm(P1 - P0))
    sp = 0.45 * target / 0.72          # ~0.72 m at 0.45 m/s on this floor
    for _ in range(tries):
        rest, steps = _roll(world, start, P0, (sp * direction).tolist())
        travelled = float((rest - P0) @ direction)
        err = float(np.linalg.norm(rest - P1))
        if err < 0.06 or travelled <= 0:
            break
        sp = float(np.clip(sp * target / travelled, 0.05, 1.0))
    _, clear_end = ball_visibility(world)
    # the duck must get to watch the ball go: in view for >= MIN_WATCH_TICKS of the first 6 s
    watch = 0
    if err < 0.08 and not clear_end:
        world.reset(start=start)
        _place_ball(world, P0)
        world.launch_ball((sp * direction).tolist())
        world.duck.set_command(0, 0, 0)
        for _ in range(60):
            for _ in range(5):
                world.step()
            watch += ball_visibility(world)[0]
    ok = err < 0.08 and not clear_end and steps < 1000 and watch >= MIN_WATCH_TICKS
    status = "ok" if ok else ("rest visible" if clear_end else
                              f"missed rest point by {err:.2f} m" if err >= 0.08 else f"watched only {watch} ticks")
    return {"status": status, "from": P0.round(4).tolist(), "velocity": (sp * direction).round(5).tolist(),
            "speed": round(sp, 5), "rest": rest.round(4).tolist(), "rest_error": round(err, 4),
            "rest_steps": int(steps), "watch_ticks": int(watch)}


# -------------------------------------------------------------------------------------- judges
class Judge:
    """Ground-truth scorekeeper, updated after every tick."""

    def __init__(self, world, ep):
        self.world, self.ep, self.task = world, ep, ep["task"]
        m = world.model
        self.robot_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] >= world.duck.trunk_id
                            and m.body_rootid[m.geom_bodyid[g]] == world.duck.trunk_id}
        names = ("wall_N", "wall_S", "wall_E", "wall_W", "partition", "occluder")
        self.obstacles = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in names} - {-1}
        self.falls, self.fallen, self.collision_ticks, self.ticks = 0, False, 0, 0
        self.events = []
        lay = world.cfg["layout"]
        self.pads = {p["color"]: p for p in lay["pads"]}
        self.home = lay["home"]
        self.reached_pad_tick, self.home_after_pad_tick = None, None
        self.wrong_pads = set()
        self.presses0 = world.press_count
        self.lamp0 = world.lamp_on
        self.plate_visits, self.on_plate = 0, False
        self.ball_hidden_seen, self.reacquired_tick = False, None

    def _inside(self, xy, centre, half):
        return abs(xy[0] - centre[0]) <= half and abs(xy[1] - centre[1]) <= half

    def update(self, iface: Interface):
        w = self.world
        self.ticks = iface.tick_count
        tilt = w.duck.tilt()
        if not self.fallen and tilt > FALL_TILT:
            self.falls += 1
            self.fallen = True
            self.events.append(("fall", self.ticks))
        elif self.fallen and tilt < np.radians(20):
            self.fallen = False
        d = w.duck.data
        if any((c.geom1 in self.robot_geoms and c.geom2 in self.obstacles) or
               (c.geom2 in self.robot_geoms and c.geom1 in self.obstacles) for c in d.contact[:d.ncon]):
            self.collision_ticks += 1
        xy = w.duck.pos()[:2]
        if self.task == "go_and_return":
            tgt = self.pads[self.ep["instruction"]["target"]]
            if self.reached_pad_tick is None:
                if self._inside(xy, tgt["pos"], tgt["size"] / 2 + PAD_MARGIN):
                    self.reached_pad_tick = self.ticks
                for c, p in self.pads.items():
                    if c != tgt["color"] and self._inside(xy, p["pos"], p["size"] / 2 + PAD_MARGIN):
                        self.wrong_pads.add(c)
            elif self.home_after_pad_tick is None and self._inside(xy, self.home["pos"], self.home["size"] / 2 + MAT_MARGIN):
                self.home_after_pad_tick = self.ticks
        elif self.task == "set_lamp":
            b = w.cfg["layout"]["button"]
            on = self._inside(xy, b["pos"], b["size"] / 2)
            if on and not self.on_plate:
                self.plate_visits += 1
            self.on_plate = on
        else:
            vis, _ = ball_visibility(w)
            if not vis and self.ticks > 0:
                self.ball_hidden_seen = True
            if vis and self.ball_hidden_seen and self.reacquired_tick is None:
                self.reacquired_tick = self.ticks

    def result(self, iface: Interface):
        """Score at the end (done bit or budget)."""
        w = self.world
        xy = w.duck.pos()[:2]
        vx, vy, _ = w.duck.planar_velocity()
        stopped = float(np.hypot(vx, vy)) < STOPPED_SPEED
        declared = iface.done
        res = {"task": self.task, "instruction": self.ep["instruction"], "config_id": self.ep["config_id"],
               "k": self.ep["k"], "declared_done": bool(declared), "ticks": int(iface.tick_count),
               "sim_time": round(float(w.duck.time), 2), "falls": self.falls,
               "collision_ticks": self.collision_ticks, "stopped": stopped}
        if self.task == "go_and_return":
            on_mat = self._inside(xy, self.home["pos"], self.home["size"] / 2 + MAT_MARGIN)
            ok = declared and self.reached_pad_tick is not None and on_mat and stopped
            res.update(reached_pad_tick=self.reached_pad_tick, home_after_pad_tick=self.home_after_pad_tick,
                       on_mat=on_mat, wrong_pad_visits=sorted(self.wrong_pads))
        elif self.task == "set_lamp":
            target = self.ep["instruction"]["target"] == "on"
            presses = w.press_count - self.presses0
            needed = int(self.lamp0 != target)
            ok = declared and w.lamp_on == target
            res.update(lamp_initial=self.lamp0, lamp_final=w.lamp_on, target=target, presses=presses,
                       needed_presses=needed, extra_presses=max(0, presses - needed),
                       plate_visits=self.plate_visits, visits_without_press=max(0, self.plate_visits - presses))
        else:
            vis, clear = ball_visibility(w)
            dist = float(np.linalg.norm(w.ball_pos()[:2] - xy))
            bv = w.duck.data.qvel[w.ball_vadr:w.ball_vadr + 2]
            ball_at_rest = float(np.hypot(*bv)) < BALL_REST_SPEED
            ok = (declared and vis and dist <= BALL_FIND_DIST and stopped and ball_at_rest
                  and self.ball_hidden_seen)
            res.update(ball_visible_at_end=vis, ball_distance=round(dist, 3), ball_at_rest=ball_at_rest,
                       ball_hidden_seen=self.ball_hidden_seen,
                       reacquired_tick=self.reacquired_tick, ball_rest=w.ball_pos()[:2].round(3).tolist())
        res["success"] = bool(ok)
        return res


def run_episode(cfg, ep, controller, world=None):
    """Run one episode: controller(ctx-like state) is a generator factory taking (iface, instruction)
    and yielding Commands. Returns the judge's result."""
    world = world or World(cfg)
    start_episode(world, ep)
    iface = Interface(world)
    judge = Judge(world, ep)
    gen = controller(iface, ep["instruction"])
    obs = iface.observe()
    try:
        cmd = next(gen)
        while not iface.done and iface.tick_count < ep["budget_ticks"]:
            obs = iface.tick(cmd)
            judge.update(iface)
            if iface.done:
                break
            cmd = gen.send(obs)
    except StopIteration:
        pass
    return judge.result(iface), world, iface
