"""Build the self-contained, editable media/demo.html (no network needed).

Inlines web/terra_core.js (the JS port, parity-tested against Python), web/demo_ui.js and a
few Python-generated maps (float32 elevation + kinds, base64) with bridge-crossing queries.
  ../.venv/bin/python build_demo.py
"""
import base64, json
import numpy as np
from terra import terrain, cost, scenarios

MAPS = [(900003, "Lake crossing (held-out seed 900003)"), (900007, "River valley (seed 900007)"),
        (900012, "Highlands (seed 900012)"), (900021, "Delta (seed 900021)")]


def bake(seed, name):
    T = terrain.generate(seed, 96)
    G = cost.Graph(T["elev"], T["kind"])
    q = scenarios.pick_bridge_query(T, G, np.random.default_rng(1), min_dist=50)
    s, g = (q[0], q[1]) if q else (5 * 96 + 5, 90 * 96 + 90)
    return {"name": name, "h": 96, "w": 96, "start": int(s), "goal": int(g),
            "elev": base64.b64encode(T["elev"].astype("<f4").tobytes()).decode(),
            "kind": base64.b64encode(T["kind"].astype(np.int8).tobytes()).decode()}


def hnet_spec(tags=("hnet16", "hnet")):
    """Bake the smallest available trained heuristic net (JS runs it on the CPU)."""
    import os
    from terra import learn
    for tag in tags:
        if os.path.exists(f"checkpoints/{tag}.safetensors"):
            cfg = json.load(open(f"checkpoints/{tag}.json"))
            m = learn.HNet(base=cfg["base"])
            m.load_weights(f"checkpoints/{tag}.safetensors")
            spec = learn.export_js(m, cfg["base"])
            spec["tag"] = tag
            print(f"baking learned heuristic '{tag}' (base {cfg['base']}, step {cfg.get('step')})")
            return spec
    return None


def main(out="media/demo.html"):
    tpl = open("web/demo_template.html").read()
    core = open("web/terra_core.js").read()
    ui = open("web/demo_ui.js").read()
    maps = [bake(s, n) for s, n in MAPS]
    spec = hnet_spec()
    html = (tpl.replace("/*__CORE__*/", core).replace("/*__MAPS__*/", json.dumps(maps))
            .replace("/*__HNET__*/", open("web/hnet.js").read())
            .replace("/*__HNET_SPEC__*/", json.dumps(spec)).replace("/*__UI__*/", ui))
    assert "http://" not in html and "https://" not in html, "demo must not reference the network"
    open(out, "w").write(html)
    print(f"wrote {out} ({len(html) / 1024:.0f} KB, {len(maps)} maps)")


if __name__ == "__main__":
    main()
