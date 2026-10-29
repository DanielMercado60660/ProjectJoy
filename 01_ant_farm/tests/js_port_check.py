"""Check that the JS port inside media/demo.html computes the same observations and the same
action probabilities (hand-written and learned) as the Python code. Needs `node` on PATH.

  ../.venv/bin/python tests/js_port_check.py media/demo.html [checkpoint_dir]
"""
import json, os, subprocess, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sim, handwritten as hw
from evaluate import load_policy

html_path = sys.argv[1]
ck = sys.argv[2] if len(sys.argv) > 2 else None
html = open(html_path).read()
js = html.split("<script>")[1].split("// ------------------------------------------------------------ panels")[0]

w = sim.make_worlds(np.array([50_001]), 64, "heldout")
for _ in range(120):
    pm, pd = hw.compass(sim.observe(w))
    m, d = sim.sample_actions(w, pm, pd)
    sim.step(w, m, d)
obs = sim.observe(w)[0]
hpm, hpd = hw.compass(obs[None])
pm, pd = hw.compass(sim.observe(w))
m_act, d_act = sim.sample_actions(w, pm, pd)
w_after = w.copy()
for _ in range(3):
    sim.step(w_after, m_act, d_act)
state = {"m": m_act[0].tolist(), "d": d_act[0].tolist(), "obst": w.obst[0].ravel().tolist(), "food": w.food[0].ravel().tolist(), "nest": w.nest[0].tolist(),
         "pher": w.pher[0].ravel().tolist(), "y": w.pos[0, :, 0].tolist(), "x": w.pos[0, :, 1].tolist(),
         "h": w.head[0].tolist(), "carry": w.carry[0].tolist(), "energy": w.energy[0].tolist(),
         "alive": w.alive[0].tolist(), "clock": w.clock[0].tolist(), "store": float(w.store[0])}
test = js + """
const S = JSON.parse(require('fs').readFileSync(process.argv[2]));
const w = new World({obst: S.obst, food: S.food, nest: S.nest});
w.pher.set(S.pher); w.y.set(S.y); w.x.set(S.x); w.h.set(S.h); w.carry.set(S.carry); w.energy.set(S.energy);
w.alive.set(S.alive); w.clock.set(S.clock);
const out = {obs: [], hpm: [], hpd: [], npm: [], npd: []};
const netName = Object.keys(DATA.nets)[0];
const nb = netName ? BRAINS[netName] : null;
for (let i = 0; i < 64; i++) {
  const o = observe(w, i, new Float32Array(59)); out.obs.push(Array.from(o));
  const pm = new Float32Array(7), pd = new Float32Array(3);
  handBrain(o, pm, pd); out.hpm.push(Array.from(pm)); out.hpd.push(Array.from(pd));
  if (nb) { const pm2 = new Float32Array(7), pd2 = new Float32Array(3); nb(Float32Array.from(o), pm2, pd2); out.npm.push(Array.from(pm2)); out.npd.push(Array.from(pd2)); }
}
const w2 = new World({obst: S.obst, food: S.food, nest: S.nest});
w2.pher.set(S.pher); w2.y.set(S.y); w2.x.set(S.x); w2.h.set(S.h); w2.carry.set(S.carry); w2.energy.set(S.energy);
w2.alive.set(S.alive); w2.clock.set(S.clock); w2.store = S.store;
for (let s = 0; s < 3; s++) stepWorld(w2, Int32Array.from(S.m), Int32Array.from(S.d));
out.after = {pher: Array.from(w2.pher), y: Array.from(w2.y), x: Array.from(w2.x), energy: Array.from(w2.energy), carry: Array.from(w2.carry), delivered: w2.delivered};
console.log(JSON.stringify(out));
"""
with tempfile.TemporaryDirectory() as td:
    open(os.path.join(td, "t.js"), "w").write(test)
    open(os.path.join(td, "s.json"), "w").write(json.dumps(state))
    res = json.loads(subprocess.run(["node", os.path.join(td, "t.js"), os.path.join(td, "s.json")],
                                    check=True, capture_output=True, text=True).stdout)
ok = True
d_obs = np.abs(np.array(res["obs"]) - obs).max(); print("obs max diff", d_obs); ok &= d_obs < 1e-4
d_h = max(np.abs(np.array(res["hpm"]) - hpm[0]).max(), np.abs(np.array(res["hpd"]) - hpd[0]).max())
print("hand-written probs max diff", d_h); ok &= d_h < 1e-4
A = res["after"]
d_s = max(np.abs(np.array(A["pher"]) - w_after.pher[0].ravel()).max(), np.abs(np.array(A["y"]) - w_after.pos[0, :, 0]).max(),
          np.abs(np.array(A["x"]) - w_after.pos[0, :, 1]).max(), np.abs(np.array(A["energy"]) - w_after.energy[0]).max(),
          np.abs(np.array(A["carry"]) - w_after.carry[0]).max())
print("3 steps of dynamics max diff", d_s); ok &= d_s < 1e-4
if ck and res["npm"]:
    pol = load_policy(ck)
    pm, pd = pol(obs[None])
    d_n = max(np.abs(np.array(res["npm"]) - pm[0]).max(), np.abs(np.array(res["npd"]) - pd[0]).max())
    print("learned probs max diff (weights rounded to 1e-4 in the html)", d_n); ok &= d_n < 5e-3
print("JS PORT OK" if ok else "JS PORT MISMATCH")
sys.exit(0 if ok else 1)
