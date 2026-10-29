"""Reviewer probe (REVIEW.md): what gait does the residual policy actually run, and does it exploit physics?

On N held-out layouts (same seeds / init noise as evaluate.py) it records, per method:
  - mean frequency and stride multipliers the policy outputs (a[12], a[13]) and the effective gait
    frequency / commanded stride;
  - share of foot-stone normal force carried by contacts whose normal is more than 45 deg from vertical (side face or edge), and the fraction of
    control steps with a side-face contact carrying > 20 N;
  - foot-stone contacts whose normal is mostly horizontal (stepping on a stone's side face) and
    foot contacts within 2 cm of a stone's front/back edge, as a fraction of foot-contact substeps;
  - stones that no foot ever touched (jumping over a stone);
  - max |qvel| and max contact penetration (sim-instability check).
Usage: ../.venv/bin/python review_probe.py --n 40 --out results/review_probe.json
"""
import argparse, json, os
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))


def probe(task, L, policy, rng):
    env = task.env
    m, d = env.m, env.d
    o = task.reset(L, rng, 0.05)
    fm, sm = [], []
    side, edge, n_fc = 0, 0, 0
    f_all, f_side, side_steps, n_steps = 0.0, 0.0, 0, 0
    fbuf = np.zeros(6)
    touched = np.zeros(len(L), bool)
    max_qvel, max_pen = 0.0, 0.0
    foot_set = set(env.foot_geom)
    stone_of = {g: i for i, g in enumerate(env.stone_geom)}
    import task as T, lava_env as E
    while True:
        a = np.zeros(T.ACT_DIM) if policy is None else np.clip(policy(o), -1, 1)
        fm.append(1 + T.FREQ_MOD * a[12]); sm.append(1 + T.STRIDE_MOD * a[13])
        # step manually so contacts can be inspected every substep (mirrors LavaTask.step)
        o, r, done, info = task.step(a)
        # contact inspection at the post-step state (one sample per control step)
        n_steps += 1
        step_side = False
        for ci, c in enumerate(d.contact[:d.ncon]):
            g1, g2 = c.geom1, c.geom2
            if g1 in foot_set and g2 in stone_of:
                fg, sg = g1, g2
            elif g2 in foot_set and g1 in stone_of:
                fg, sg = g2, g1
            else:
                continue
            n_fc += 1
            i = stone_of[sg]
            touched[i] = True
            mujoco.mj_contactForce(m, d, ci, fbuf)
            fn = abs(float(fbuf[0]))
            f_all += fn
            below_top = abs(c.frame[2]) < 0.7        # normal more than 45 deg from vertical: side face / edge
            if abs(c.frame[2]) < 0.7:          # contact normal is the first row of frame
                side += 1
            if below_top:
                f_side += fn
                if fn > 20.0:
                    step_side = True
            px = c.pos[0]
            if min(abs(px - L[i, 0]), abs(px - L[i, 1])) < 0.02:
                edge += 1
            max_pen = max(max_pen, -float(c.dist))
        side_steps += int(step_side)
        max_qvel = max(max_qvel, float(np.abs(d.qvel).max()))
        if done:
            break
    return dict(success=info["success"], lava=info["lava"], t=env.t,
                freq_mul=float(np.mean(fm)), stride_mul=float(np.mean(sm)),
                side_contact_frac=side / max(n_fc, 1),
                side_face_force_share=f_side / max(f_all, 1e-9),
                steps_with_loaded_side_face_contact=side_steps / max(n_steps, 1), edge2cm_contact_frac=edge / max(n_fc, 1),
                stones_untouched=int((~touched[1:-1]).sum()) if info["success"] else None,
                max_qvel=max_qvel, max_penetration=max_pen, speed=env.progress() / env.t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default="results/review_probe.json")
    a = ap.parse_args()
    os.chdir(HERE)
    import task as T, lava_env as E, ppo
    pol, cfg = ppo.make_policy_fn("checkpoints/policy.npz")
    methods = [("residual_ppo", cfg["cpg_params"], pol, False),
               ("cpg_planner_r4", json.load(open("checkpoints/planner4_params.json"))["params"], None, True)]
    out = {}
    for name, cpg, p, plan in methods:
        task = T.LavaTask(cpg, planner=plan)
        rs = []
        for i in range(a.n):
            seed = E.HELDOUT_SEED_BASE + i
            rs.append(probe(task, E.make_layout(seed), p, np.random.default_rng(seed + 50_000)))
        keys = [k for k in rs[0] if k not in ("success", "lava", "stones_untouched")]
        summ = {k: float(np.mean([r[k] for r in rs])) for k in keys}
        summ["max_qvel_worst"] = float(max(r["max_qvel"] for r in rs))
        summ["max_penetration_worst"] = float(max(r["max_penetration"] for r in rs))
        summ["side_contact_frac_worst"] = float(max(r["side_contact_frac"] for r in rs))
        summ["side_face_force_share_worst"] = float(max(r["side_face_force_share"] for r in rs))
        un = [r["stones_untouched"] for r in rs if r["stones_untouched"] is not None]
        summ["completed"] = len(un)
        summ["episodes_with_a_stone_never_touched"] = int(sum(u > 0 for u in un))
        summ["base_freq_hz"] = cpg["freq"]; summ["base_stride_m"] = cpg["stride"]
        summ["effective_freq_hz"] = cpg["freq"] * summ["freq_mul"]
        summ["effective_cmd_stride_m"] = cpg["stride"] * summ["stride_mul"]
        out[name] = dict(summary=summ, episodes=rs)
        print(name, json.dumps(summ, indent=1))
    json.dump(out, open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
