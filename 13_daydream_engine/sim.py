"""Bouncing-balls world: N balls in a unit box under gravity.

Vectorised over a batch of scenes with numpy. State per ball is (x, y, vx, vy);
a scene state is a flat vector of length 4*N ordered ball by ball.

Physics (fixed for every scene):
  * box [0,1]^2, ball radius R, gravity G pointing down (-y)
  * walls are inelastic: normal velocity is reversed and scaled by E_WALL
  * ball-ball collisions are elastic (equal masses), resolved only when the pair
    overlaps AND is approaching, then the pair is pushed apart along the normal
  * each recorded frame is FRAME_DT seconds, integrated with SUBSTEPS
    semi-implicit Euler substeps

`simulate` also returns per-frame collision flags (wall hit, ball-ball hit) so
evaluation can split errors around bounces.
"""
import numpy as np

N_BALLS = 3
R = 0.06
G = 3.0
E_WALL = 0.85
E_BALL = 1.0
FRAME_DT = 0.02
SUBSTEPS = 10
STATE_DIM = 4 * N_BALLS


def random_init(n_scenes, rng, n_balls=N_BALLS, vmax=1.5):
    """Random non-overlapping positions and uniform velocities in [-vmax, vmax]."""
    pos = np.zeros((n_scenes, n_balls, 2))
    for i in range(n_balls):
        p = rng.uniform(R, 1 - R, size=(n_scenes, 2))
        for _ in range(200):  # rejection sample until no overlap with earlier balls
            bad = np.zeros(n_scenes, bool)
            for j in range(i):
                bad |= np.linalg.norm(p - pos[:, j], axis=1) < 2.2 * R
            if not bad.any():
                break
            p[bad] = rng.uniform(R, 1 - R, size=(bad.sum(), 2))
        pos[:, i] = p
    vel = rng.uniform(-vmax, vmax, size=(n_scenes, n_balls, 2))
    return np.concatenate([pos, vel], axis=-1).reshape(n_scenes, -1)


def _substep(p, v, h, balls=True):
    """One substep in place. p, v: (B, N, 2). Returns (wall_hit, ball_hit) bool (B,).
    balls=False skips ball-ball collisions (used by the walls-only baseline)."""
    v[..., 1] -= G * h
    p += v * h
    wall = np.zeros(p.shape[0], bool)
    for d in range(2):
        lo = p[..., d] < R
        hi = p[..., d] > 1 - R
        hit_lo = lo & (v[..., d] < 0)
        hit_hi = hi & (v[..., d] > 0)
        v[..., d] = np.where(hit_lo | hit_hi, -E_WALL * v[..., d], v[..., d])
        p[..., d] = np.where(lo, 2 * R - p[..., d], p[..., d])
        p[..., d] = np.where(hi, 2 * (1 - R) - p[..., d], p[..., d])
        wall |= (hit_lo | hit_hi).any(axis=1)
    ball = np.zeros(p.shape[0], bool)
    n = p.shape[1] if balls else 0
    for i in range(n):
        for j in range(i + 1, n):
            dp = p[:, j] - p[:, i]
            dist = np.linalg.norm(dp, axis=1) + 1e-12
            nrm = dp / dist[:, None]
            vrel = np.sum((v[:, j] - v[:, i]) * nrm, axis=1)
            hit = (dist < 2 * R) & (vrel < 0)
            if hit.any():
                imp = (-(1 + E_BALL) * vrel / 2)[:, None] * nrm * hit[:, None]
                v[:, i] -= imp
                v[:, j] += imp
                over = (2 * R - dist) * hit / 2
                p[:, i] -= nrm * over[:, None]
                p[:, j] += nrm * over[:, None]
                ball |= hit
    return wall, ball


def simulate(init, n_frames, substeps=SUBSTEPS, balls=True):
    """Roll out `n_frames` frames from `init` (B, 4N).

    Returns states (B, n_frames+1, 4N) including the initial frame, and
    collision flags (B, n_frames, 2): [wall, ball] for the transition t -> t+1.
    """
    B = init.shape[0]
    n = init.shape[1] // 4
    s = init.reshape(B, n, 4).astype(np.float64).copy()
    p, v = s[..., :2].copy(), s[..., 2:].copy()
    out = np.zeros((B, n_frames + 1, n, 4))
    out[:, 0, :, :2], out[:, 0, :, 2:] = p, v
    flags = np.zeros((B, n_frames, 2), bool)
    h = FRAME_DT / substeps
    for t in range(n_frames):
        for _ in range(substeps):
            w, b = _substep(p, v, h, balls)
            flags[:, t, 0] |= w
            flags[:, t, 1] |= b
        out[:, t + 1, :, :2], out[:, t + 1, :, 2:] = p, v
    return out.reshape(B, n_frames + 1, -1), flags


# ---------- cheap baselines (no learning) ----------

def constant_velocity(init, n_frames):
    """x_{t+1} = x_t + v_t dt, velocity frozen. Returns (B, n_frames+1, 4N)."""
    B = init.shape[0]
    s = init.reshape(B, -1, 4)
    t = np.arange(n_frames + 1)[None, :, None] * FRAME_DT
    p = s[:, None, :, :2] + s[:, None, :, 2:] * t[..., None]
    v = np.broadcast_to(s[:, None, :, 2:], p.shape)
    return np.concatenate([p, v], -1).reshape(B, n_frames + 1, -1)


def ballistic(init, n_frames):
    """Exact free fall under gravity, ignoring every wall and ball collision."""
    B = init.shape[0]
    s = init.reshape(B, -1, 4)
    t = (np.arange(n_frames + 1) * FRAME_DT)[None, :, None]
    p = s[:, None, :, :2] + s[:, None, :, 2:] * t[..., None]
    v = np.broadcast_to(s[:, None, :, 2:], p.shape).copy()
    p[..., 1] -= 0.5 * G * t ** 2
    v[..., 1] -= G * t
    return np.concatenate([p, v], -1).reshape(B, n_frames + 1, -1)


def walls_only(init, n_frames):
    """Known-physics baseline added in review: the true integrator with gravity and wall
    bounces (true restitution E_WALL), but no ball-ball collisions."""
    return simulate(init, n_frames, balls=False)[0]


def make_dataset(n_scenes, n_frames, seed):
    rng = np.random.default_rng(seed)
    init = random_init(n_scenes, rng)
    states, flags = simulate(init, n_frames)
    return states.astype(np.float32), flags


if __name__ == "__main__":
    import argparse, os, time
    ap = argparse.ArgumentParser(description="generate train/test bouncing-ball datasets")
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    for name, n, seed in [("train", a.n_train, 0), ("test", a.n_test, 1)]:
        s, f = make_dataset(n, a.frames, seed)
        np.savez_compressed(os.path.join(a.out, f"{name}.npz"), states=s, flags=f, seed=seed)
        print(f"{name}: {s.shape}, wall-hit frames {f[...,0].mean():.3f}, ball-hit frames {f[...,1].mean():.3f}")
    print(f"done in {time.time()-t0:.1f}s")
