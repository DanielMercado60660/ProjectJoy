"""Round 6 opponents. All controllers take a `view` (anything with n, pos (n,2,2), th (n,2), ball (n,2),
bvel (n,2), last_touch (n,)) and return (n, 2) wheel commands for the robot they drive.

TRAINING POOL (randomized families; `goalie`, `keeper`, `chaser` from scripted.py are inside these ranges,
so in round 6 those three are dev-visible, not held out):
  goalie_r  guard on the ball->goal line at standoff s, rush the ball inside radius r
  keeper_r  line keeper at depth d in front of the goal line, mirrors the ball's y predicted h s ahead
  chaser_r  striker: lines up behind the ball and pushes it toward the learner's goal
  marker    body-blocks the learner: sits on the learner->goal segment m metres from the learner,
            rushes the ball only when it is within r of its goal
  wall      near-static blocker at a random spot in front of the goal, drifting toward the ball's y
Every family also gets a speed scale, a reaction delay of 0-6 steps (it sees the ball as it was 0-0.3 s ago)
and Gaussian action noise std U[0, 0.3], drawn in sample_params; r6_common.R6Arena.reset then makes 35% of
episodes nominal (speed 1, no delay, no noise) and maps the speed of the rest to U[0.75, 1] (pilot-1 fix).

HELD-OUT (eval only; specs written to STATUS.md at 18:08 before any r6 code; never trained or tuned on):
  interceptor, zone, presser  (fixed parameters, speed 1.0, no delay, no noise)

Also: attacker_param, a parametric version of scripted.attacker for the CEM-tuned baseline.
"""
import numpy as np
from env import W, H, GOAL_HALF, R_ROBOT, R_BALL, DT, BALL_FRICTION, E_WALL, V_OPP, wrap


def drive_to_v(pos, th, target, gain=4.0, reverse_ok=True, slow_radius=0.05):
    """scripted.drive_to with per-env gain / reverse_ok allowed (arrays or scalars)."""
    d = target - pos
    dist = np.linalg.norm(d, axis=1)
    ang = np.arctan2(d[:, 1], d[:, 0])
    err = wrap(ang - th)
    rev = np.broadcast_to(np.asarray(reverse_ok, dtype=bool), dist.shape)
    back = rev & (np.abs(err) > np.pi / 2)
    fwd = np.where(back, -1.0, 1.0)
    err = np.where(back, wrap(err + np.pi), err)
    v = fwd * np.clip(dist / slow_radius, 0, 1) * np.clip(np.cos(err), 0, 1)
    w = np.clip(np.asarray(gain) * err / np.pi, -1, 1)
    return np.stack([np.clip(v - w, -1, 1), np.clip(v + w, -1, 1)], 1)


def clamp_field(t):
    t[:, 0] = np.clip(t[:, 0], -W + R_ROBOT, W - R_ROBOT)
    t[:, 1] = np.clip(t[:, 1], -H + R_ROBOT, H - R_ROBOT)
    return t


def behind_ball(pos, th, ball, goal, standoff=0.13, aim_tol=0.05, push=0.3, detour=0.08, gain=4.0,
                reverse_ok=False):
    """scripted._behind_ball with per-env parameters (arrays of shape (n,) or scalars)."""
    n = len(pos)
    sv = lambda x: np.broadcast_to(np.asarray(x, dtype=float), (n,))[:, None]
    g = goal - ball
    gu = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-9)
    rb = ball - pos
    along = np.sum(rb * gu, 1)
    cross = rb[:, 0] * gu[:, 1] - rb[:, 1] * gu[:, 0]
    aligned = (along > 0.02) & (np.abs(cross) < sv(aim_tol)[:, 0])
    behind_pt = ball - gu * sv(standoff)
    side = np.sign(cross + 1e-9)
    perp = np.stack([-gu[:, 1], gu[:, 0]], 1) * side[:, None]
    wrong = along < 0.0
    det = ball - gu * 0.05 + perp * (R_ROBOT + R_BALL + sv(detour))
    target = np.where(wrong[:, None], det, behind_pt)
    target = np.where(aligned[:, None], ball + gu * sv(push), target)
    return drive_to_v(pos, th, clamp_field(target), gain=gain, reverse_ok=reverse_ok)


# ----------------------------------------------------------------------------- training pool
def goalie_r(v, P, r=1):
    goal = np.array([W, 0.0])
    gb = v.ball - goal
    dist = np.linalg.norm(gb, axis=1, keepdims=True) + 1e-9
    guard = goal + gb / dist * P["standoff"][:, None]
    guard[:, 0] = np.minimum(guard[:, 0], W - R_ROBOT)
    guard[:, 1] = np.clip(guard[:, 1], -GOAL_HALF - 0.05, GOAL_HALF + 0.05)
    target = np.where((dist[:, 0] < P["rush"])[:, None], v.ball, guard)
    return drive_to_v(v.pos[:, r], v.th[:, r], target)


def keeper_r(v, P, r=1):
    pred = v.ball + v.bvel * P["horizon"][:, None]
    target = np.stack([W - R_ROBOT - 0.03 - P["depth"], np.clip(pred[:, 1], -P["yclip"], P["yclip"])], 1)
    return drive_to_v(v.pos[:, r], v.th[:, r], target)


def chaser_r(v, P, r=1):
    goal = np.tile(np.array([-W, 0.0]), (v.n, 1))
    return behind_ball(v.pos[:, r], v.th[:, r], v.ball, goal, standoff=P["standoff"])


def marker(v, P, r=1):
    goal = np.array([W, 0.0])
    me, opp = v.pos[:, r], v.pos[:, 1 - r]
    d = goal - opp
    du = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
    block = opp + du * P["mark"][:, None]
    rush = np.linalg.norm(v.ball - goal, axis=1) < P["rush"]
    target = np.where(rush[:, None], v.ball, block)
    return drive_to_v(me, v.th[:, r], clamp_field(target))


def wall(v, P, r=1):
    target = np.stack([P["wx"], P["wy"] + P["drift"] * (np.clip(v.ball[:, 1], -0.4, 0.4) - P["wy"])], 1)
    return drive_to_v(v.pos[:, r], v.th[:, r], target)


POOL_FAMILIES = {"goalie_r": goalie_r, "keeper_r": keeper_r, "chaser_r": chaser_r, "marker": marker, "wall": wall}


def sample_params(rng, k):
    """Per-env parameters for every family (cheap to draw all; each env uses its own family's)."""
    u = lambda lo, hi: rng.uniform(lo, hi, k)
    return dict(standoff=u(0.08, 0.40), rush=u(0.20, 0.70), depth=u(0.0, 0.15), horizon=u(0.0, 0.6),
                yclip=u(0.2, 0.35), cstand=u(0.08, 0.20), mark=u(0.15, 0.40), mrush=u(0.20, 0.40),
                wx=u(0.5, 0.85), wy=u(-0.3, 0.3), drift=u(0.0, 0.5),
                speed=u(0.7, 1.0), delay=rng.integers(0, 7, k), noise=u(0.0, 0.3))


def family_params(name, P):
    if name == "chaser_r":
        return dict(standoff=P["cstand"])
    if name == "marker":
        return dict(mark=P["mark"], rush=P["mrush"])
    return P


# ----------------------------------------------------------------------------- held-out (eval only)
def _roll_ball(ball, bvel, steps=40):
    """Predict the ball path with the env's friction and wall bounces (no robots). Returns (steps, n, 2)."""
    b, vv = ball.copy(), bvel.copy()
    out = np.zeros((steps,) + b.shape)
    f = np.exp(-BALL_FRICTION * DT)
    for k in range(steps):
        vv = vv * f
        b = b + vv * DT
        for sgn in (1, -1):
            m = sgn * b[:, 1] > H - R_BALL
            b[m, 1] = sgn * (H - R_BALL)
            vv[m, 1] = -E_WALL * np.abs(vv[m, 1]) * sgn
            mouth = np.abs(b[:, 1]) < GOAL_HALF
            m = (sgn * b[:, 0] > W - R_BALL) & ~mouth
            b[m, 0] = sgn * (W - R_BALL)
            vv[m, 0] = -E_WALL * np.abs(vv[m, 0]) * sgn
        b[:, 0] = np.clip(b[:, 0], -W - 0.05, W + 0.05)
        out[k] = b
    return out


def interceptor(v, r=1):
    me, th = v.pos[:, r], v.th[:, r]
    goal = np.array([W, 0.0])
    path = _roll_ball(v.ball, v.bvel, 40)                                  # 2.0 s
    tk = (np.arange(1, 41) * DT)[:, None]
    reach = np.linalg.norm(path - me[None], axis=2) / V_OPP + 0.15 <= tk   # (40, n)
    reach &= path[:, :, 0] > 0.1      # rev 1 (band fix, ~18:10): only intercepts in its own half
    first = np.argmax(reach, 0)
    anyr = reach.any(0)
    tgt_i = path[first, np.arange(v.n)]
    gb = v.ball - goal
    guard = goal + gb / (np.linalg.norm(gb, axis=1, keepdims=True) + 1e-9) * 0.30
    target = np.where(anyr[:, None], tgt_i, guard)
    close = np.linalg.norm(v.ball - me, axis=1) < 0.18
    clear_pt = np.stack([np.full(v.n, -0.3), 0.65 * np.sign(v.ball[:, 1] + 1e-9)], 1)
    cu = clear_pt - v.ball
    cu /= np.linalg.norm(cu, axis=1, keepdims=True) + 1e-9
    target = np.where(close[:, None], v.ball + cu * 0.3, target)
    return drive_to_v(me, th, clamp_field(target))


def zone(v, r=1):
    me, th = v.pos[:, r], v.th[:, r]
    bx, by = v.ball[:, 0], v.ball[:, 1]
    # rev 2 (band fix, ~18:11): home line 0.45 -> 0.62, zone starts at x > 0.30 (was 0.15), and the step-up
    # clears diagonally toward the near touchline (was straight upfield into the attacker).
    HL = 0.62
    home = np.stack([np.full(v.n, HL), np.clip(0.6 * by, -0.35, 0.35)], 1)
    in_zone = (bx > 0.30) & (bx < me[:, 0] + 0.05) & (np.abs(by - me[:, 1]) < 0.35)
    clear = np.stack([np.full(v.n, -0.2), 0.7 * np.sign(by + 1e-9)], 1) - v.ball
    clear /= np.linalg.norm(clear, axis=1, keepdims=True) + 1e-9
    step_up = v.ball + clear * 0.3
    behind = bx > me[:, 0] + 0.05
    # recovery mode persists while the ball is beyond the home line
    rec = getattr(v, "zone_recover", None)
    if rec is None:
        rec = np.zeros(v.n, dtype=bool)
    rec = (rec | behind) & (bx > HL)
    if hasattr(v, "zone_recover"):
        v.zone_recover[:] = rec
    gl = np.stack([np.full(v.n, 0.88), np.clip(by, -0.3, 0.3)], 1)
    # rev 1 (bug fix, ~18:10): the straight recovery run went through the ball into its own goal (83% of the
    # goals it conceded were its own touches). Route around: if the ball is within 0.2 m of the path, aim at a
    # waypoint beside the ball on the far side from the ball's y first.
    seg = gl - me
    sl = np.sum(seg * seg, 1) + 1e-9
    tproj = np.clip(np.sum((v.ball - me) * seg, 1) / sl, 0, 1)
    near_path = np.linalg.norm(me + tproj[:, None] * seg - v.ball, axis=1) < 0.2
    side = np.where(me[:, 1] > by, 1.0, -1.0)
    wp = np.stack([bx, by + side * 0.22], 1)
    gl = np.where((near_path & (tproj < 0.95))[:, None], wp, gl)
    target = np.where(in_zone[:, None], step_up, home)
    target = np.where(rec[:, None], gl, target)
    return drive_to_v(me, th, clamp_field(target))


def presser(v, r=1):
    me, th = v.pos[:, r], v.th[:, r]
    opp = v.pos[:, 1 - r]
    goal = np.array([W, 0.0])
    d_opp = np.linalg.norm(v.ball - opp, axis=1)
    d_me = np.linalg.norm(v.ball - me, axis=1)
    charge = (d_opp < 0.30) | (d_me < d_opp)
    bg = goal - v.ball
    shadow = v.ball + bg / (np.linalg.norm(bg, axis=1, keepdims=True) + 1e-9) * 0.25
    target = np.where(charge[:, None], v.ball, shadow)
    act = drive_to_v(me, th, clamp_field(target))
    counter = (v.last_touch == r) & (me[:, 0] > v.ball[:, 0])
    if counter.any():
        lg = np.tile(np.array([-W, 0.0]), (v.n, 1))
        act[counter] = behind_ball(me, th, v.ball, lg)[counter]
    return act


HELDOUT = {"interceptor": interceptor, "zone": zone, "presser": presser}


# ----------------------------------------------------------------------------- parametric scripted attacker
ATT_DEFAULT = dict(standoff=0.13, aim=0.0, aim_tol=0.05, push=0.3, detour=0.08, gain=4.0, rev=-1.0)
ATT_BOUNDS = dict(standoff=(0.06, 0.30), aim=(0.0, 1.0), aim_tol=(0.01, 0.15), push=(0.05, 0.6),
                  detour=(0.02, 0.20), gain=(1.5, 10.0), rev=(-1.0, 1.0))


def attacker_param(v, P, r=0):
    """scripted.attacker with tunable knobs. aim: shoot at y = -sign(defender y)*aim*GOAL_HALF (the reviewer's
    off-centre aim); rev > 0 allows reversing. ATT_DEFAULT reproduces scripted.attacker exactly."""
    n = v.n
    p = {k: np.broadcast_to(np.asarray(P[k], dtype=float), (n,)) for k in ATT_DEFAULT}
    gy = -np.sign(v.pos[:, 1 - r, 1] + 1e-9) * p["aim"] * GOAL_HALF
    goal = np.stack([np.full(n, W), gy], 1)
    return behind_ball(v.pos[:, r], v.th[:, r], v.ball, goal, standoff=p["standoff"], aim_tol=p["aim_tol"],
                       push=p["push"], detour=p["detour"], gain=p["gain"], reverse_ok=p["rev"] > 0)
