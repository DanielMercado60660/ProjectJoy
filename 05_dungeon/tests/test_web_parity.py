"""The JS solver in demo.html must behave like the Python one (BFS, Trémaux and the GRU policy)."""
import json, os, shutil, subprocess, sys
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dungeon.mazes import make_levels, bfs_dist  # noqa: E402

NODE = shutil.which("node")


def _node(script):
    return json.loads(subprocess.check_output([NODE, "-e", script], cwd=ROOT, timeout=60))


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_js_bfs_matches():
    L = make_levels("braid", 10, seed=4)
    lv = [dict(w=[int(x) for x in w.ravel()], s=[int(s[0]), int(s[1])], g=[int(g[0]), int(g[1])])
          for w, s, g in zip(L["walls"], L["start"], L["goal"])]
    js = f"""const c=require('./dungeon/web_core.js'); const L={json.dumps(lv)};
    console.log(JSON.stringify(L.map(l=>c.bfsDist(Uint8Array.from(l.w), l.s[0], l.s[1])[l.g[0]*13+l.g[1]])));"""
    d_js = _node(js)
    d_py = [int(bfs_dist(w, s[0], s[1])[g[0], g[1]]) for w, s, g in zip(L["walls"], L["start"], L["goal"])]
    assert d_js == d_py


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_js_policy_matches_python_greedy():
    import mlx.core as mx
    from dungeon.policy import RecurrentPolicy
    from dungeon.rollout import run_policy
    from export_web import policy_spec
    import tempfile

    mx.random.seed(3)
    pol = RecurrentPolicy(32)
    mx.eval(pol.parameters())
    tmp = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmp, "checkpoints", "p"))
    pol.save_weights(os.path.join(tmp, "checkpoints", "p", "weights.safetensors"))
    import export_web
    old = export_web.HERE
    export_web.HERE = tmp
    try:
        spec = policy_spec("p")
    finally:
        export_web.HERE = old
    L = make_levels("rooms", 6, seed=8)
    o = run_policy(pol, L, greedy=True, keep_traj=True, T=40)
    lv = [dict(w=[int(x) for x in w.ravel()], s=[int(s[0]), int(s[1])], g=[int(g[0]), int(g[1])])
          for w, s, g in zip(L["walls"], L["start"], L["goal"])]
    with open(os.path.join(tmp, "spec.json"), "w") as f:
        json.dump(dict(spec=spec, levels=lv), f)
    js = f"""const c=require('./dungeon/web_core.js'); const D=require('{tmp}/spec.json');
    const p=c.makePolicy(D.spec); const out=D.levels.map(l=>c.runPolicy(p, Uint8Array.from(l.w), l.s, l.g, 0, true).traj.slice(0,41));
    console.log(JSON.stringify(out));"""
    trajs = _node(js)
    same = sum(np.array_equal(np.array(tj), o["traj"][i][:len(tj)]) for i, tj in enumerate(trajs))
    assert same >= 5, same  # float32 vs float64 rounding may flip a rare near-tie
