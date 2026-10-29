"""One-lap time trials on a set of tracks.

Each car starts at rest on the start line and must complete one full lap
(cumulative signed progress >= track length) before the time limit
(track length / MIN_AVG_SPEED). Two rule sets:

  strict   touching a wall ends the run (DNF, counted as a crash)
  respawn  touching a wall counts one collision; the car is put back on the
           centerline at that spot, at rest, facing along the track, and drives on.
           Shows whether a controller that crashes can still get round.
  soft     (reward-hacking demo only) walls push the car back and it slides along
           them; collisions = number of steps spent scraping a wall.
"""
import numpy as np

from .sim import CAR_R, DT, N_RAYS, RAY_LEN, VecRace

MIN_AVG_SPEED = 4.0  # m/s; time limit = length / this


def run_lap(controller, bank, grip=1.0, mode="strict", record=False, track_ids=None):
    T = len(bank.tracks) if track_ids is None else len(track_ids)
    tids = np.arange(len(bank.tracks)) if track_ids is None else np.asarray(track_ids)
    env = VecRace(bank, T, soft_walls=(mode == "soft"))
    obs = env.reset(tids, grip=grip)
    length = bank.length[tids]
    limit = np.ceil(length / MIN_AVG_SPEED / DT).astype(int)
    finished = np.zeros(T, bool)
    lap_time = np.full(T, np.nan)
    collisions = np.zeros(T, int)
    active = np.ones(T, bool)
    max_prog = np.zeros(T)
    min_prog = np.zeros(T)
    traj, events = [], []
    lat_hist, ray_min, jitter = [], np.full(T, np.inf), []
    hw_eff = bank.half_width[tids] - CAR_R
    crash_inside = np.zeros(T)
    for t in range(int(limit.max())):
        act = controller(obs, env)
        steer_before = env.steer_prev.copy()
        act[~active] = 0
        env.done[~active] = True
        obs, dprog, done = env.step(act)
        if record:
            traj.append(env.snapshot())
        drive = active & ~env.crashed  # cars still driving normally this step
        lat_hist.append(np.where(drive, np.abs(env.lat) / hw_eff, np.nan))
        jitter.append(np.where(drive & (t > 0), np.abs(env.steer_prev - steer_before), np.nan))
        ray_min = np.where(drive, np.minimum(ray_min, obs[:, :N_RAYS].min(1) * RAY_LEN), ray_min)
        max_prog = np.maximum(max_prog, env.prog)
        min_prog = np.minimum(min_prog, env.prog)
        # a wall hit on the step that completes the lap is still a wall hit (review fix):
        # under strict rules it is a DNF; under respawn rules the car finishes after the respawn
        newly = active & (env.prog >= length) & ~env.crashed
        finished |= newly
        lap_time[newly] = (t + 1) * DT
        active &= ~newly
        crash = active & env.crashed
        collisions += crash
        for e in np.where(crash)[0]:  # which wall? inside of the bend (apex) or outside
            k = bank.tracks[tids[e]].curvature
            i = env.idx[e]
            kk = np.take(k, np.arange(i - 5, i + 6), mode="wrap").mean()
            crash_inside[e] += float(np.sign(env.lat[e]) == np.sign(kk))
        if record:
            for e in np.where(crash)[0]:
                events.append((t, int(e), float(env.x[e]), float(env.y[e])))
        if mode == "soft":
            collisions = env.wall_hits.copy()
        elif mode == "strict":
            active &= ~crash
        elif crash.any():
            b = bank
            for e in np.where(crash)[0]:
                i = env.idx[e]
                tid = env.track_id[e]
                env.x[e], env.y[e] = b.center[tid, i]
                env.yaw[e] = np.arctan2(b.tangent[tid, i, 1], b.tangent[tid, i, 0])
                env.v[e] = 0.0
                env.s_frac[e] = 0.0
                env.lat[e] = 0.0
                env.done[e] = False
                env.crashed[e] = False
            obs = env.observe()
        active &= (t + 1) < limit
        if not active.any():
            break
    res = dict(finished=finished, lap_time=lap_time, collisions=collisions,
               progress_frac=np.clip(max_prog / length, 0, 1), min_prog=min_prog,
               length=length, crashed=collisions > 0)
    # wall-hugging diagnostics: |lateral offset| as a fraction of the room to the wall
    # (0 = on the centerline, 1 = touching), closest ray reading, steering jitter
    lh = np.array(lat_hist)
    res["lat_mean"] = np.nanmean(lh, 0)
    res["lat_p95"] = np.nanpercentile(lh, 95, axis=0)
    res["near_wall_frac"] = np.nanmean(np.where(np.isnan(lh), np.nan, lh > 0.8), 0)
    res["ray_min"] = ray_min
    res["crash_inside"] = crash_inside  # number of crashes into the inside (apex) wall
    res["steer_jitter"] = np.nanmean(np.array(jitter), 0)
    if record:
        res["lat_hist"] = np.nan_to_num(lh, nan=0.0)
        res["traj"] = np.stack(traj, 0)  # (steps, T, 6)
        res["events"] = events
    return res


def summarize(res):
    f = res["finished"]
    return dict(
        completion=float(f.mean()),
        crash_rate=float((res["collisions"] > 0).mean()),
        collisions_per_lap=float(res["collisions"].mean()),
        mean_lap_time=float(np.nanmean(res["lap_time"])) if f.any() else float("nan"),
        mean_avg_speed=float(np.nanmean(res["length"][f] / res["lap_time"][f])) if f.any() else float("nan"),
        mean_progress=float(res["progress_frac"].mean()),
        lat_mean=float(np.nanmean(res["lat_mean"])),
        lat_p95=float(np.nanmean(res["lat_p95"])),
        near_wall_time=float(np.nanmean(res["near_wall_frac"])),
        steer_jitter=float(np.nanmean(res["steer_jitter"])),
        inside_wall_share=float(res["crash_inside"].sum() / max(1, res["collisions"].sum()))
        if "crash_inside" in res else float("nan"),
        n=int(len(f)),
    )
