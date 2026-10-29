"""Cross-review checks (2026-10-24). Writes only results/review_*.json; never touches results/tournament.json.

  ../.venv/bin/python review_eval.py select            # pick the Corner script's params (training scripts, seed 302)
  ../bin/gpu_run ../.venv/bin/python review_eval.py contact   # replay the round robin, classify ring-outs by contact
  ../bin/gpu_run ../.venv/bin/python review_eval.py rr        # Corner vs every entrant, n=200, eval seeds 6000+
  ../.venv/bin/python review_eval.py probe                    # push physics: head-on vs offset vs side (1 s)
  ../bin/gpu_run ../.venv/bin/python review_eval.py physics   # chassis height / uprightness / speed / MuJoCo warnings

`contact` replays results/tournament.json's round robin bout for bout (same seeds and side alternation as
tournament.run_cell), checks that W/L/D reproduce exactly, and records for every ring-out whether the two robots
were touching (MuJoCo contact list, any geom of robot a against any geom of robot b) in the last 0.5 s / 1 s.
"""
import itertools, json, pickle, sys, time
from multiprocessing import Pool
import numpy as np
import tournament
from tournament import make, wilson
from sumo_env import SumoEnv, make_script
from review_scripts import REVIEW_CLASSES

CKPT = "checkpoints/main.pkl"
CORNER_GRID = [  # top round-1 configs (seed 300, 24 bouts/opp vs charger/flanker/circler), see REVIEW.md
    dict(off_far=0.2, off_near=0.1, d1=0.45, k=2.5, side="init"),
    dict(off_far=0.1, off_near=0.1, d1=0.45, k=2.5, side="init"),
    dict(off_far=0.1, off_near=0.1, d1=0.6, k=2.5, side="init"),
    dict(off_far=0.1, off_near=0.05, d1=0.45, k=1.0, side="init"),
    dict(off_far=0.2, off_near=0.15, d1=0.45, k=2.5, side="init"),
    dict(off_far=0.1, off_near=0.05, d1=0.6, k=1.0, side="edge"),
]


def corner_params():
    try:
        return json.load(open("results/review_corner_select.json"))["best"]
    except FileNotFoundError:
        return CORNER_GRID[0]


def make_any(spec, seed):
    if spec == "corner":
        return REVIEW_CLASSES["corner"](**corner_params())
    return make(spec, seed)


def robot_of_geom(m):
    root = m.body_rootid[m.geom_bodyid]
    ia = m.body("a").id
    ib = m.body("b").id
    return np.where(root == ia, 0, np.where(root == ib, 1, -1))


def bout_with_contact(env, pa, pb, R=None, fric=None):
    """Same as sumo_env.play_bout, plus per-step a-b contact from MuJoCo's contact list."""
    rg = robot_of_geom(env.m)
    oa, ob = env.reset(R, fric)
    for p in (pa, pb):
        if hasattr(p, "reset"):
            p.reset()
    touch = []
    while True:
        (oa, ob), out, info = env.step(pa(oa), pb(ob))
        nc = env.d.ncon
        g1, g2 = rg[np.asarray(env.d.contact.geom1)[:nc]], rg[np.asarray(env.d.contact.geom2)[:nc]]
        touch.append(bool(np.any((g1 >= 0) & (g2 >= 0) & (g1 != g2))))
        if out is not None:
            return dict(outcome=out, steps=info["t"], out_a=bool(info["out_a"]), out_b=bool(info["out_b"]),
                        tip_a=bool(info["tip_a"]), tip_b=bool(info["tip_b"]), touch=touch)


def contact_cell(task):
    a, b, n, seed, R, fric = task
    env = SumoEnv(seed=seed)
    W = L = D = 0
    rows = []
    for i in range(n):
        pa, pb = make_any(a, seed * 7 + i), make_any(b, seed * 11 + i)
        if i % 2 == 0:
            r = bout_with_contact(env, pa, pb, R, fric); o = r["outcome"]; lo_a, lo_b = r["out_a"], r["out_b"]
        else:
            r = bout_with_contact(env, pb, pa, R, fric); o = -r["outcome"]; lo_a, lo_b = r["out_b"], r["out_a"]
        W += o == 1; L += o == -1; D += o == 0
        t = r["touch"]
        rows.append(dict(o=int(o), out_a=lo_a, out_b=lo_b, steps=r["steps"], touch_any=float(np.mean(t)),
                         touch_05s=any(t[-12:]), touch_1s=any(t[-25:]), touched_ever=any(t)))
    return dict(a=a, b=b, n=n, seed=seed, W=int(W), L=int(L), D=int(D), bouts=rows)


def summarize(cells):
    out = []
    for c in cells:
        dec = [r for r in c["bouts"] if r["o"] != 0]
        wins = [r for r in dec if r["o"] == 1]
        losses = [r for r in dec if r["o"] == -1]
        out.append(dict(a=c["a"], b=c["b"], W=c["W"], L=c["L"], D=c["D"],
                        a_wins_no_touch_05s=sum(not r["touch_05s"] for r in wins),
                        a_wins_no_touch_1s=sum(not r["touch_1s"] for r in wins),
                        a_wins_never_touched=sum(not r["touched_ever"] for r in wins),
                        a_losses_no_touch_05s=sum(not r["touch_05s"] for r in losses),
                        a_losses_no_touch_1s=sum(not r["touch_1s"] for r in losses),
                        draw_touch_frac=float(np.mean([r["touch_any"] for r in c["bouts"] if r["o"] == 0] or [0]))))
    return out


def physics_cell(task):
    """Physics sanity per bout: max chassis height, min uprightness, max planar speed, MuJoCo warnings."""
    a, b, n, seed, R, fric = task
    env = SumoEnv(seed=seed)
    zmax, upmin, vmax, warn, nonfinite = [], [], [], 0, 0
    for i in range(n):
        pa, pb = make_any(a, seed * 7 + i), make_any(b, seed * 11 + i)
        oa, ob = env.reset(R, fric)
        for p in (pa, pb):
            if hasattr(p, "reset"):
                p.reset()
        w0 = sum(int(w.number) for w in env.d.warning)
        zm, um, vm = 0.0, 1.0, 0.0
        while True:
            (oa, ob), out, info = env.step(pa(oa), pb(ob))
            for k in range(2):
                q, v = env.qadr[k], env.vadr[k]
                zm = max(zm, float(env.d.qpos[q + 2]))
                um = min(um, float(env.state(k)[2]))
                vm = max(vm, float(np.linalg.norm(env.d.qvel[v:v + 2])))
            if not np.all(np.isfinite(env.d.qpos)):
                nonfinite += 1
                break
            if out is not None:
                break
        warn += sum(int(w.number) for w in env.d.warning) - w0
        zmax.append(zm); upmin.append(um); vmax.append(vm)
    return dict(a=a, b=b, R=R, fric=fric, n=n, max_chassis_z=max(zmax), frac_bouts_z_over_0p10=float(np.mean(np.array(zmax) > 0.10)),
                min_up=min(upmin), frac_bouts_up_below_0p8=float(np.mean(np.array(upmin) < 0.8)),
                max_speed=max(vmax), mujoco_warnings=warn, nonfinite=nonfinite)


def score_vs_training(task):
    p, seed, n = task
    env = SumoEnv(seed=seed)
    res = {}
    for opp in ["charger", "flanker", "circler"]:
        W = L = 0
        for i in range(n):
            a, b = REVIEW_CLASSES["corner"](**p), make_script(opp, tournament.TUNED.get(opp))
            o = tournament.play_bout_any(env, a, b, None, None)["outcome"] if i % 2 == 0 else \
                -tournament.play_bout_any(env, b, a, None, None)["outcome"]
            W += o == 1; L += o == -1
        res[opp] = [int(W), int(L)]
    return dict(params=p, res=res, score=sum(w - l for w, l in res.values()) / (3 * n))


def push_probe():
    """Robot A at full throttle for 4 s against robot B placed 0.4 m ahead (R 1.0, friction 1.0)."""
    import mujoco
    from sumo_env import yaw_quat
    env = SumoEnv(seed=0)
    for name, thb, actb, yb in [("head-on, B idle", np.pi, [0, 0], 0.0), ("head-on, B pushing", np.pi, [1, 1], 0.0),
                                ("side-on, B idle", np.pi / 2, [0, 0], 0.0),
                                ("8 cm offset, B pushing", np.pi, [1, 1], 0.08)]:
        env.set_conditions(1.0, 1.0)
        mujoco.mj_resetData(env.m, env.d)
        qa, qb = env.qadr
        env.d.qpos[qa:qa + 3] = [-0.4, 0, 0.06]; env.d.qpos[qa + 3:qa + 7] = yaw_quat(0)
        env.d.qpos[qb:qb + 3] = [0.0, yb, 0.06]; env.d.qpos[qb + 3:qb + 7] = yaw_quat(thb)
        mujoco.mj_forward(env.m, env.d); env.t = 0
        for _ in range(100):
            _, out, _ = env.step(np.array([1.0, 1.0]), np.array(actb, float))
            if out is not None:
                break
        pb, yaw, _, _, _ = env.state(1)
        print(f"{name:24s} B moved {np.linalg.norm(pb - [0, yb]):.2f} m in {env.t * 0.04:.1f} s, "
              f"B turned {abs(np.degrees(np.angle(np.exp(1j * (yaw - thb))))):.0f} deg, outcome {out}")


def main():
    mode = sys.argv[1]
    if mode == "probe":
        return push_probe()
    t0 = time.time()
    with Pool(4, initializer=tournament.init, initargs=(CKPT,)) as pool:
        if mode == "select":
            rows = pool.map(score_vs_training, [(p, 302, 48) for p in CORNER_GRID])
            rows.sort(key=lambda r: -r["score"])
            out = dict(best=rows[0]["params"], rows=rows, seed=302, n_per_opp=48,
                       opponents=["charger", "flanker", "circler"], note="counter excluded (held out)")
            json.dump(out, open("results/review_corner_select.json", "w"), indent=1)
            for r in rows:
                print(f"{r['score']:+.3f} {r['res']} {r['params']}")
        elif mode == "contact":
            T = json.load(open("results/tournament.json"))
            cells = pool.map(contact_cell, [(r["a"], r["b"], r["n"], r["seed"], r["R"], r["fric"]) for r in T["rr"]],
                             chunksize=1)
            for c, r in zip(cells, T["rr"]):
                assert (c["W"], c["L"], c["D"]) == (r["W"], r["L"], r["D"]), (c["a"], c["b"], c["W"], r["W"])
            summ = summarize(cells)
            json.dump(dict(summary=summ, cells=cells, reproduced=True), open("results/review_contact.json", "w"))
            for s in summ:
                print(s)
        elif mode == "physics":
            tasks = []
            for k, (R, f) in enumerate([(None, None), (1.0, 1.6)]):
                for j, b in enumerate(["charger", "flanker", "circler", "counter", "corner"]):
                    tasks.append(("ppo", b, 100, 6100 + 10 * k + j, R, f))
            tasks.append(("charger", "flanker", 100, 6120, None, None))
            res = pool.map(physics_cell, tasks, chunksize=1)
            json.dump(res, open("results/review_physics.json", "w"), indent=1)
            for r in res:
                print(r)
        elif mode == "rr":
            ents = ["ppo", "charger", "flanker", "circler", "counter", "random"]
            tasks = [("corner", e, 200, 6000 + k, None, None) for k, e in enumerate(ents)]
            cells = pool.map(contact_cell, tasks, chunksize=1)
            res = []
            for c in cells:
                n = c["n"]
                res.append(dict(a=c["a"], b=c["b"], n=n, seed=c["seed"], W=c["W"], L=c["L"], D=c["D"],
                                win_ci=wilson(c["W"], n), loss_ci=wilson(c["L"], n), score=(c["W"] - c["L"]) / n))
            json.dump(dict(params=corner_params(), rr=res, contact=summarize(cells)),
                      open("results/review_corner_rr.json", "w"), indent=1)
            for r in res:
                print(f"corner vs {r['b']:<8} W {r['W']:3d} L {r['L']:3d} D {r['D']:3d} score {r['score']:+.3f} "
                      f"winCI {r['win_ci']} lossCI {r['loss_ci']}")
    print(f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
