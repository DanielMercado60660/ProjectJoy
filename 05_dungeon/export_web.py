"""Build the self-contained media/demo.html (dungeon editor + solver replay).

  ../.venv/bin/python export_web.py [--policies solver_cur1 solver_v1]

Embeds the GRU policy weights (float32, base64) and a set of preset dungeons
(designer levels + held-out family examples) into dungeon/demo_template.html.
"""
import argparse, base64, json, os
import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten

from dungeon.data import test_set
from dungeon.mazes import ALL_FAMILIES

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS = {"solver_cur3": "PPO-GRU after 3 curriculum rounds", "solver_ff": "PPO, memoryless ablation",
          "solver_cur1": "PPO-GRU after designer curriculum", "solver_v1": "PPO-GRU (families only)",
          "solver_ctl1": "PPO-GRU control fine-tune", "smoke_solver": "PPO-GRU (smoke, untrained)"}


def policy_spec(name):
    w = mx.load(os.path.join(HERE, "checkpoints", name, "weights.safetensors"))
    params = {}
    for k, v in w.items():
        a = np.array(v, dtype=np.float32)
        params[k] = dict(shape=list(a.shape), b64=base64.b64encode(a.tobytes()).decode())
    hidden = int(w["gru.h2h.weight"].shape[1])
    pj = os.path.join(HERE, "checkpoints", name, "policy.json")
    memoryless = bool(json.load(open(pj)).get("memoryless", False)) if os.path.exists(pj) else False
    return dict(label=LABELS.get(name, name), hidden=hidden, memoryless=memoryless, params=params)


def lv(w, s, g, name, note=""):
    return dict(name=name, note=note, walls=[int(x) for x in np.asarray(w).ravel()],
                start=[int(s[0]), int(s[1])], goal=[int(g[0]), int(g[1])])


def presets():
    out = []
    runs = [("v1_s1", "bred against PPO-GRU v1", 4), ("ctl1_s2", "the limit-cycle trap (vs ctl1)", 2),
            ("v1_md12_s7", "bred against v1, exit >= 12 steps", 2), ("expl_cur3_s8", "bred against the round-3 curriculum solver", 2),
            ("wall_follow_s0", "bred against the wall follower", 2), ("tremaux_s0", "bred against Trémaux", 2)]
    for run, desc, n in runs:
        p = os.path.join(HERE, "results", f"designer_{run}.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p)
        order = np.argsort(-z["fresh_fitness"])
        chosen = []
        for i in order:
            if all((z["walls"][i] != z["walls"][j]).sum() > 12 for j in chosen):
                chosen.append(i)
            if len(chosen) == n:
                break
        for r, i in enumerate(chosen):
            out.append(lv(z["walls"][i], z["start"][i], z["goal"][i], f"Designer: {desc} #{r + 1}",
                          f"Evolved by the designer; regret {z['fresh_fitness'][i]:.2f}, target solver failed "
                          f"{100 * z['fresh_fail'][i]:.0f}% of 32 fresh attempts."))
    for f in ALL_FAMILIES:
        L = test_set(f)
        for i in range(2):
            out.append(lv(L["walls"][i], L["start"][i], L["goal"][i], f"Family: {f} #{i + 1}" +
                          (" (held out)" if f in ("caves", "spiral") else ""),
                          "Procedurally generated test level" + (" from a family the solver never trained on."
                                                                  if f in ("caves", "spiral") else ".")))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", nargs="*", default=["solver_cur3", "solver_v1", "solver_ff"])
    ap.add_argument("--out", default=os.path.join(HERE, "media", "demo.html"))
    a = ap.parse_args(argv)
    pols = {p: policy_spec(p) for p in a.policies if os.path.exists(os.path.join(HERE, "checkpoints", p, "weights.safetensors"))}
    data = dict(policies=pols, presets=presets())
    tpl = open(os.path.join(HERE, "dungeon", "demo_template.html")).read()
    core = open(os.path.join(HERE, "dungeon", "web_core.js")).read()
    html = tpl.replace("/*__CORE__*/", core).replace("/*__DATA__*/", json.dumps(data, separators=(",", ":")))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        f.write(html)
    print(f"wrote {a.out} ({len(html) / 1e6:.2f} MB, {len(pols)} policies, {len(data['presets'])} presets)")


if __name__ == "__main__":
    main()
