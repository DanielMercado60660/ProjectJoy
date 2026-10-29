"""ORACLE controllers: the M2 skills fed simulator ground truth (true pose, true layout, true lamp
state, true ball position). Labelled "oracle" everywhere. They exist to prove each task is
physically solvable through the frozen interface (lessons file, item 4) and to show where perception
will cost the reference bot (M3). They are not a baseline.
"""
import mujoco
import numpy as np

from configs import DUCK_CLEARANCE, footprints, _point_free
from skills import (Ctx, TruePose, go_to_planned, look, press_button, return_home, stop, turn_to,
                    wrap)
from tasks import ball_viewpoints, ball_visibility

LABEL = "oracle"


def _finish(ctx):
    yield ctx.cmd(done=True)


def go_and_return(world, iface, instruction):
    lay = world.cfg["layout"]
    ctx = Ctx(iface, TruePose(world), lay)
    pad = next(p for p in lay["pads"] if p["color"] == instruction["target"])
    yield from go_to_planned(ctx, pad["pos"], stop_radius=0.08)
    yield from return_home(ctx, lay["home"]["pos"])
    yield from _finish(ctx)


def set_lamp(world, iface, instruction, max_tries=3):
    lay = world.cfg["layout"]
    ctx = Ctx(iface, TruePose(world), lay)
    target = instruction["target"] == "on"
    tries = 0
    while world.lamp_on != target and tries < max_tries:     # oracle: reads the lamp flag directly
        tries += 1
        yield from press_button(ctx, lay["button"]["pos"])
        for _ in range(12):                                   # let the plate re-arm before re-checking
            yield ctx.cmd()
    yield from _finish(ctx)


def _ball_viewpoint(world, ball_xy, me_xy):
    return ball_viewpoints(world, ball_xy, me_xy)


def find_ball(world, iface, instruction):
    ctx = Ctx(iface, TruePose(world), world.cfg["layout"])
    # watch the ball until it stops (it starts rolling at t=0)
    for _ in range(80):
        v = world.duck.data.qvel[world.ball_vadr:world.ball_vadr + 3]
        if np.hypot(v[0], v[1]) < 0.01:
            break
        yield ctx.cmd()
    ball = world.ball_pos()[:2]                               # oracle: true ball position
    for vp in _ball_viewpoint(world, ball, np.array(ctx.pose()[:2]))[:8]:
        rep = yield from go_to_planned(ctx, vp, stop_radius=0.08)
        if not rep["ok"]:
            continue
        x, y, _ = ctx.pose()
        yield from turn_to(ctx, np.arctan2(ball[1] - y, ball[0] - x), tol=0.15)
        # small head sweep (the head bobs; a single check can miss): stop at the first clear view
        for head in ((0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.3, 0.0), (0.0, 0.0, -0.3, 0.0), (0.0, 0.3, 0.0, 0.0)):
            if ball_visibility(world)[0]:
                break
            yield from look(ctx, head)
        if ball_visibility(world)[0] or iface.tick_count > 520:   # leave time to stop and say done
            break
    yield from stop(ctx)
    yield from _finish(ctx)


CONTROLLERS = {"go_and_return": go_and_return, "set_lamp": set_lamp, "find_ball": find_ball}


def controller_for(world, task):
    fn = CONTROLLERS[task]
    return lambda iface, instruction: fn(world, iface, instruction)


# ------------------------------------------------------------------------------------------------
# No-memory BASELINES for task 3 (labelled baseline, not oracle): true pose and layout, and they
# recognise the ball perfectly when it's in view (ground-truth visibility), but they don't use where
# the ball went. They visit candidate hiding regions behind each hider (the side away from where the
# duck started), nearest first, each from a viewpoint with a clear view of it, and look around.
# "occluder_only" searches only behind the occluder (what setup v1 rewarded); "all_hiders" searches
# behind every hiding place.
def no_memory_search(world, iface, instruction, which="all_hiders"):
    from tasks import _hiders, ball_viewpoints
    from skills import look
    lay = world.cfg["layout"]
    ctx = Ctx(iface, TruePose(world), lay)
    me0 = np.array(ctx.pose()[:2])
    # candidate hiding regions: behind each hider (side away from the start), near each open end and,
    # for the occluder, its middle; 0.33 m behind (the middle of the setup's 0.22-0.45 m range)
    regions = []
    for h in _hiders(lay):
        if which == "occluder_only" and h["name"] != "occluder":
            continue
        hs = -1.0 if (me0 - h["C"]) @ h["n"] > 0 else 1.0
        backs = [0.12, 0.30] if h["name"] == "partition" else [0.2]
        for e in h["ends"]:
            E = h["C"] + e * h["L"] * h["a"]
            for back in backs:
                regions.append(E - e * back * h["a"] + hs * 0.33 * h["n"])
    found = False
    while regions and not found and iface.tick_count < 560:
        here = np.array(ctx.pose()[:2])
        regions.sort(key=lambda r: np.linalg.norm(r - here))     # nearest unvisited region first
        target = regions.pop(0)
        vps = ball_viewpoints(world, target, here)
        if not vps:
            continue
        rep = yield from go_to_planned(ctx, vps[0], stop_radius=0.08)
        if not rep["ok"]:
            continue
        x, y, _ = ctx.pose()
        yield from turn_to(ctx, np.arctan2(target[1] - y, target[0] - x), tol=0.2)
        for hy in (0.0, 0.6, -0.6):                               # perfect recognition, small head sweep
            yield from look(ctx, (0.0, 0.0, hy, 0.0))
            if ball_visibility(world)[0]:
                found = True
                break
        if found:
            yield from look(ctx, (0.0, 0.0, 0.0, 0.0))
            b = world.ball_pos()[:2]
            here = np.array(ctx.pose()[:2])
            if np.linalg.norm(b - here) > 0.75 and iface.tick_count < 540:
                # M3 review: it used to say done on sight, sometimes 0.81-1.32 m away (the judge wants
                # <= 0.8 m). Get within range along the line of sight, as the reference bot does.
                u = (here - b) / max(np.linalg.norm(here - b), 1e-6)
                yield from go_to_planned(ctx, b + 0.65 * u, stop_radius=0.08, max_ticks=max(30, 540 - iface.tick_count))
                x, y, _ = ctx.pose()
            yield from turn_to(ctx, np.arctan2(b[1] - y, b[0] - x), tol=0.15)
            if not ball_visibility(world)[0]:                     # keep the head on it if turning lost it
                for hy in (0.3, -0.3):
                    yield from look(ctx, (0.0, 0.0, hy, 0.0))
                    if ball_visibility(world)[0]:
                        break
    yield from stop(ctx)
    yield from _finish(ctx)


def baseline_for(world, which):
    return lambda iface, instruction: no_memory_search(world, iface, instruction, which)
