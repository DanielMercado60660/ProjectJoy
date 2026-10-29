"""The JS port in web/terra_core.js (inlined into media/demo.html) must compute exactly the
same edge costs and optimal path costs as the Python reference. Skipped if node is missing."""
import json, random, shutil, subprocess, sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from terra import terrain, cost, planners  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def test_js_matches_python(tmp_path):
    T = terrain.generate(900003, 48)
    G = cost.Graph(T["elev"], T["kind"])
    rng = random.Random(0)
    passable = [u for u in range(G.n) if G.adj[u]]
    qs = [(rng.choice(passable), rng.choice(passable)) for _ in range(8)]
    inp = {"h": G.h, "w": G.w, "elev": [float(x) for x in G.elev.ravel()], "kind": [int(k) for k in G.kind.ravel()],
           "queries": qs}
    f = tmp_path / "in.json"
    f.write_text(json.dumps(inp))
    out = json.loads(subprocess.check_output(["node", str(ROOT / "tests" / "js_harness.js"), str(f)]))
    # edge costs: identical layout (node*8 + dir) and identical float values
    py = cost.edge_costs(G.elev, G.kind)  # (8, H, W)
    py = np.where(np.isfinite(py), py, -1).reshape(8, -1).T.ravel()
    js = np.array(out["edge_costs"])
    assert np.array_equal(py, js), f"max diff {np.max(np.abs(py - js))}"
    for (s, g), q in zip(qs, out["queries"]):
        ref = planners.dijkstra(G, s, g).cost
        ref = -1 if not np.isfinite(ref) else ref
        for k in ("dijkstra", "astar", "dstar"):
            assert abs(q[k] - ref) < 1e-9, (k, q[k], ref)
    assert out["dstar_checks"] > 50 and out["dstar_bad"] == 0
    assert "rebuild_mismatch" not in out
    assert out["gen_ok"]


@pytest.mark.parametrize("tag", ["hnet", "hnet16"])
def test_js_hnet_matches_mlx(tmp_path, tag):
    """web/hnet.js (float16 weights) reproduces the MLX network's output (both the evaluated
    model and the browser-sized one that demo.html bakes)."""
    from terra import learn
    import mlx.core as mx
    ck = ROOT / "checkpoints" / f"{tag}.safetensors"
    if not ck.exists():
        pytest.skip(f"no {tag} checkpoint yet")
    base = json.loads((ROOT / "checkpoints" / f"{tag}.json").read_text())["base"]
    model = learn.HNet(base=base)
    model.load_weights(str(ck))
    T = terrain.generate(900003, 96)
    gr, gc = 40, 60
    spec = learn.export_js(model, base)
    inp = {"h": 96, "w": 96, "elev": [float(x) for x in T["elev"].ravel()], "kind": [int(k) for k in T["kind"].ravel()],
           "goal": gr * 96 + gc, "spec": spec}
    f = tmp_path / "in.json"
    f.write_text(json.dumps(inp))
    out = json.loads(subprocess.check_output(["node", str(ROOT / "tests" / "hnet_harness.js"), str(f)]))
    st = learn.static_features(T["elev"], T["kind"])
    x = np.concatenate([st, learn.goal_features(96, 96, gr, gc)]).transpose(1, 2, 0)
    xj = np.array(out["features"]).reshape(96, 96, 8)
    assert np.abs(x - xj).max() < 1e-5, np.abs(x - xj).reshape(-1, 8).max(0)
    r = np.array(model(mx.array(x[None]))[0])
    rj = np.array(out["r"]).reshape(96, 96)
    assert np.abs(r - rj).mean() < 0.01 and np.abs(r - rj).max() < 0.1, (np.abs(r - rj).mean(), np.abs(r - rj).max())
