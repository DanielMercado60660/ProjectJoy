"""How do the two flyers pass waypoints? Closest-approach distance and speed at each pass,
peak speed, and thrust saturation, on the 500 held-out courses (wind sigma 2).
  ../.venv/bin/python analyze_flight.py -> results/flight_style.json"""
import json
import numpy as np
import flight as F
import train_flight as TF


def run(courses, fn, seed=99):
    n = len(courses)
    env = F.FlightEnv(n, seed, fixed_courses=courses)
    obs = env.obs(); alive = np.ones(n, bool)
    pass_d, pass_v, sat, vmax = [], [], [], np.zeros(n)
    for _ in range(F.MAX_STEPS):
        a = fn(env, obs)
        cur, _, _ = env.targets(); p0 = env.p.copy()
        an = np.clip(a, -1, 1); sat.append(np.linalg.norm(an, axis=1)[alive] >= 0.98)
        obs, r, done, info = env.step(np.where(alive[:, None], a, 0))
        seg = env.p - p0; w = cur - p0
        u = np.clip((w * seg).sum(1) / np.maximum((seg * seg).sum(1), 1e-9), 0, 1)
        dmin = np.linalg.norm(p0 + u[:, None] * seg - cur, axis=1)
        m = info["reached"] & alive
        pass_d += list(dmin[m]); pass_v += list(np.linalg.norm(env.v[m], axis=1))
        vmax = np.where(alive, np.maximum(vmax, np.linalg.norm(env.v, axis=1)), vmax)
        alive &= ~done; env.v[~alive] = 0; env.thr[~alive] = np.array([0.0, F.G])
        if not alive.any():
            break
    pass_d, pass_v = np.array(pass_d), np.array(pass_v)
    return dict(n_passes=int(len(pass_d)), mean_pass_dist_m=float(pass_d.mean()),
                frac_pass_in_outer_half_of_ring=float((pass_d > 0.5 * F.WP_R).mean()),
                mean_speed_at_pass=float(pass_v.mean()), mean_peak_speed=float(vmax.mean()),
                thrust_saturated_frac=float(np.concatenate(sat).mean()))


if __name__ == "__main__":
    C = F.heldout_courses(500)
    prm = json.load(open("checkpoints/pd_params.json"))
    out = dict(ppo=run(C, TF.policy_fn(TF.load("checkpoints/flight_ppo"))),
               pd_tuned=run(C, lambda e, o: F.pd_action(e, prm)))
    print(json.dumps(out, indent=1)); json.dump(out, open("results/flight_style.json", "w"), indent=1)
