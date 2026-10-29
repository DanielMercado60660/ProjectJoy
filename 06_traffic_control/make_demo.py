"""Build media/demo.html (self-contained, offline: two controllers side by side on the held-out
rush-hour episode) and media/summary.png from runs/eval_results.json + runs/demo_traces.npz.

Usage: ../.venv/bin/python make_demo.py
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sim

SERIES = "#2a78d6"; HILITE = "#eb6834"; INK = "#0b0b0b"; INK2 = "#52514e"; GRID = "#e4e3df"; SURF = "#fcfcfb"


def short(n):
    return (n.replace("fixed_time (tuned for mean wait)", "fixed-time (mean-wait tuned)")
             .replace("fixed_time (tuned)", "fixed-time (tuned)").replace("longest_queue_first", "longest-queue-first")
             .replace("max_pressure", "max-pressure").replace("PPO shared policy", "PPO (learned)"))


def summary_png(res, path):
    pats = list(res["patterns"])
    names = list(res["patterns"][pats[0]])
    metrics = [("mean_wait", "Mean wait per signal (s)"), ("quiet_max_wait", "Quiet Lane worst wait (s)")]
    fig, axes = plt.subplots(len(metrics), len(pats), figsize=(4.2 * len(pats), 2.0 + 0.42 * len(names) * len(metrics)),
                             facecolor=SURF, squeeze=False)
    for r, (k, label) in enumerate(metrics):
        for c, pat in enumerate(pats):
            ax = axes[r][c]; ax.set_facecolor(SURF)
            vals = [res["patterns"][pat][n][k]["mean"] for n in names]
            errs = [res["patterns"][pat][n][k]["std"] for n in names]
            y = np.arange(len(names))[::-1]
            cols = [HILITE if "PPO" in n else SERIES for n in names]
            ax.barh(y, vals, height=0.62, color=cols, xerr=errs, error_kw=dict(ecolor=INK2, lw=0.8, capsize=2))
            for yi, v, e in zip(y, vals, errs):
                ax.text(v + e, yi, f"  {v:.0f}", va="center", ha="left", fontsize=8, color=INK)
            ax.set_yticks(y); ax.set_yticklabels([short(n) for n in names] if c == 0 else [], fontsize=8, color=INK2)
            ax.set_title(f"{pat} · {label}", fontsize=9, color=INK, loc="left")
            ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
            for s in ("top", "right", "left"):
                ax.spines[s].set_visible(False)
            ax.spines["bottom"].set_color(GRID); ax.tick_params(axis="x", colors=INK2, labelsize=7)
            ax.set_xlim(0, max(v + e for v, e in zip(vals, errs)) * 1.25)
    fig.suptitle(f"Held-out demand, {res['episodes']} arrival seeds per pattern (bars = mean, whiskers = ±1 std)",
                 fontsize=9, color=INK2, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=SURF)
    plt.close(fig)


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Traffic Control From Hell</title>
<style>
:root{--bg:#f6f5f2;--panel:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#8a8984;--line:#e4e3df;--road:#d9d7d1;
--car:#5f6b7a;--warn:#c98500;--bad:#d03b3a;--go:#1a9a5a;--stop:#d03b3a;--amber:#e3a008;--quiet:#6250d6;--accent:#2a78d6}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#121211;--panel:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;
--muted:#8f8e88;--line:#2e2e2c;--road:#34332f;--car:#9aa6b6;--quiet:#9085e9;--accent:#3987e5}}
:root[data-theme="dark"]{--bg:#121211;--panel:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#8f8e88;--line:#2e2e2c;--road:#34332f;--car:#9aa6b6;--quiet:#9085e9;--accent:#3987e5}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1180px;margin:0 auto;padding:20px 16px 40px}
h1{font-size:22px;margin:0 0 4px}
p.lede{color:var(--ink2);margin:0 0 16px;max-width:820px}
.controls{display:flex;flex-wrap:wrap;gap:10px;align-items:center;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px 12px;margin-bottom:12px}
button,select{font:inherit;color:var(--ink);background:var(--bg);border:1px solid var(--line);border-radius:7px;padding:5px 10px;cursor:pointer}
input[type=range]{flex:1;min-width:160px}
.clock{font-variant-numeric:tabular-nums;font-weight:600;min-width:64px}
.panels{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px}
.panel header{display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:6px}
canvas{width:100%;height:auto;display:block}
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-top:6px}
.stat{border:1px solid var(--line);border-radius:8px;padding:6px 8px}
.stat b{display:block;font-size:18px;font-variant-numeric:tabular-nums}
.stat span{font-size:11px;color:var(--ink2)}
.legend{display:flex;flex-wrap:wrap;gap:14px;color:var(--ink2);font-size:12px;margin:10px 0}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:-1px}
table{border-collapse:collapse;width:100%;font-size:12.5px;font-variant-numeric:tabular-nums}
th,td{padding:5px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{color:var(--ink2);font-weight:600}
.tablewrap{overflow-x:auto;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:6px 10px;margin-top:10px}
h2{font-size:16px;margin:22px 0 4px}
.note{color:var(--ink2);font-size:12.5px;max-width:860px}
</style></head><body><main>
<h1>Traffic Control From Hell</h1>
<p class="lede">Four signals on a 2&times;2 grid during a held-out rush hour (17:00&ndash;18:00, same arrivals for every controller).
Two busy avenues run east&ndash;west; Elm St is a medium north&ndash;south street; <b style="color:var(--quiet)">Quiet Lane</b> carries about one car a minute.
Queued cars turn amber after 30&nbsp;s and red after 90&nbsp;s. Watch Quiet Lane under the queue-greedy rules.</p>
<div class="controls">
<button id="play">Pause</button>
<label>speed <select id="speed"><option value="1">1&times;</option><option value="3" selected>3&times;</option><option value="8">8&times;</option><option value="20">20&times;</option></select></label>
<span class="clock" id="clock">17:00:00</span>
<input type="range" id="scrub" min="0" max="1799" value="0" aria-label="time">
</div>
<div class="legend">
<span><i class="sw" style="background:var(--car)"></i>queued car</span>
<span><i class="sw" style="background:var(--warn)"></i>front car waited &gt;30 s</span>
<span><i class="sw" style="background:var(--bad)"></i>&gt;90 s</span>
<span><i class="sw" style="background:var(--go)"></i>green</span>
<span><i class="sw" style="background:var(--amber)"></i>switching (all-red)</span>
<span><i class="sw" style="background:var(--quiet)"></i>Quiet Lane</span>
</div>
<div class="panels" id="panels"></div>
<h2>Held-out results</h2>
<p class="note">Mean &plusmn; std over __EPISODES__ arrival seeds per pattern. Waits are per signal crossing; worst waits include cars still waiting when the hour ends.</p>
<div id="tables"></div>
<p class="note">__NOTE__</p>
</main>
<script>
const DATA = __DATA__;
const RES = __RES__;
const N = 1800, W = 520, H = 520;
const ctrls = Object.keys(DATA);
const quiet = new Set([4,5,12,13]);
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
let t = 0, playing = true, speed = 3, acc = 0;
const panels = [];
function makePanel(defName){
  const el = document.createElement('div'); el.className = 'panel';
  el.innerHTML = `<header><select>${ctrls.map(c=>`<option ${c===defName?'selected':''}>${c}</option>`).join('')}</select></header>
  <canvas width="${W*2}" height="${H*2}"></canvas>
  <div class="stats"><div class="stat"><b data-k="wait">0</b><span>cars waiting now</span></div>
  <div class="stat"><b data-k="quiet">0 s</b><span>Quiet Lane front car</span></div>
  <div class="stat"><b data-k="qmax">0 s</b><span>Quiet Lane worst so far</span></div></div>`;
  document.getElementById('panels').appendChild(el);
  const p = {el, sel: el.querySelector('select'), cv: el.querySelector('canvas')};
  p.ctx = p.cv.getContext('2d'); p.ctx.scale(2,2);
  p.sel.onchange = () => draw();
  panels.push(p);
}
const ix = [170, 350], iy = [170, 350], BOX = 22, OFF = 6, CAR = 6;
function lanes(){ // geometry per lane: stop point + unit vector pointing back up the queue
  const g = [];
  for (let i=0;i<4;i++){ const r=Math.floor(i/2), c=i%2, x=ix[c], y=iy[r];
    g.push({x:x-OFF, y:y-BOX, dx:0, dy:-1});   // southbound: queue extends north
    g.push({x:x+OFF, y:y+BOX, dx:0, dy:1});    // northbound
    g.push({x:x-BOX, y:y+OFF, dx:-1, dy:0});   // eastbound
    g.push({x:x+BOX, y:y-OFF, dx:1, dy:0});    // westbound
  } return g; }
const G = lanes();
function linkLen(l){ const i=Math.floor(l/4), d=l%4, r=Math.floor(i/2), c=i%2;
  const internal = (d===0&&r===1)||(d===1&&r===0)||(d===2&&c===1)||(d===3&&c===0);
  return internal ? 180-2*BOX : 170-BOX; }
function draw(){
  const road = css('--road'), panel = css('--panel'), ink2 = css('--ink2'), qc = css('--quiet');
  for (const p of panels){
    const D = DATA[p.sel.value], ctx = p.ctx;
    ctx.fillStyle = panel; ctx.fillRect(0,0,W,H);
    // roads
    ctx.fillStyle = road;
    for (const y of iy) ctx.fillRect(0, y-14, W, 28);
    ctx.fillRect(ix[0]-14, 0, 28, H);
    ctx.fillStyle = qc; ctx.globalAlpha = 0.18; ctx.fillRect(ix[1]-14, 0, 28, H); ctx.globalAlpha = 1;
    ctx.fillStyle = ink2; ctx.font = '11px -apple-system,sans-serif';
    ctx.fillText('1st Ave', 6, iy[0]-18); ctx.fillText('2nd Ave', 6, iy[1]-18);
    ctx.save(); ctx.translate(ix[0]+18, 12); ctx.fillText('Elm St', 0, 0); ctx.restore();
    ctx.fillStyle = qc; ctx.fillText('Quiet Lane', ix[1]+18, 12);
    const off = t*16, offp = t*4;
    let waiting = 0, qnow = 0;
    for (let l=0;l<16;l++){
      const q = D.q[off+l], c = D.c[off+l], w = D.w[off+l], g = G[l];
      waiting += q; if (quiet.has(l)) qnow = Math.max(qnow, w);
      const col = w>90 ? css('--bad') : w>30 ? css('--warn') : (quiet.has(l) ? qc : css('--car'));
      const maxCars = Math.floor(linkLen(l)/CAR) - 1;
      ctx.fillStyle = col;
      for (let k=0;k<Math.min(q,maxCars);k++){
        const d = 2 + k*CAR; ctx.fillRect(g.x + g.dx*d - 2.5 + (g.dx?(g.dx>0?0:-0):0), g.y + g.dy*d - 2.5, 5, 5);
      }
      if (q > maxCars){ ctx.fillStyle = ink2; ctx.fillText('+'+(q-maxCars), g.x + g.dx*(maxCars*CAR) - (g.dx<0?24:0), g.y + g.dy*(maxCars*CAR) + (g.dy?0:-6)); }
      // cars still driving along the link (position not modelled: drawn as faint dots up-stream)
      const mv = c - q; ctx.globalAlpha = 0.35; ctx.fillStyle = css('--car');
      for (let k=0;k<Math.min(mv,6);k++){ const d = linkLen(l) - 8 - k*CAR; ctx.fillRect(g.x + g.dx*d - 2, g.y + g.dy*d - 2, 4, 4); }
      ctx.globalAlpha = 1;
    }
    // signals
    for (let i=0;i<4;i++){
      const ph = D.p[offp+i], r=Math.floor(i/2), c=i%2, x=ix[c], y=iy[r];
      ctx.fillStyle = css('--line'); ctx.fillRect(x-BOX+2, y-BOX+2, 2*BOX-4, 2*BOX-4);
      const ns = ph===-1 ? css('--amber') : ph===0 ? css('--go') : css('--stop');
      const ew = ph===-1 ? css('--amber') : ph===1 ? css('--go') : css('--stop');
      ctx.fillStyle = ns; ctx.fillRect(x-12, y-BOX-1, 24, 3); ctx.fillRect(x-12, y+BOX-2, 24, 3);
      ctx.fillStyle = ew; ctx.fillRect(x-BOX-1, y-12, 3, 24); ctx.fillRect(x+BOX-2, y-12, 3, 24);
      ctx.fillStyle = ink2; ctx.fillText('I'+i, x-6, y+4);
    }
    let qmax = 0; for (let s=0;s<=t;s++) for (const l of quiet) qmax = Math.max(qmax, D.w[s*16+l]);
    p.el.querySelector('[data-k=wait]').textContent = waiting;
    const qe = p.el.querySelector('[data-k=quiet]'); qe.textContent = Math.round(qnow)+' s';
    qe.style.color = qnow>90 ? css('--bad') : '';
    const me = p.el.querySelector('[data-k=qmax]'); me.textContent = Math.round(qmax)+' s';
    me.style.color = qmax>90 ? css('--bad') : '';
  }
  const s = 17*3600 + t*2; const hh = Math.floor(s/3600), mm = Math.floor(s%3600/60), ss = s%60;
  document.getElementById('clock').textContent = `${hh}:${String(mm).padStart(2,'0')}:${String(ss).padStart(2,'0')}`;
  document.getElementById('scrub').value = t;
}
const defA = ctrls.find(c=>c.startsWith('longest')) || ctrls[0];
const defB = ctrls.find(c=>c.startsWith('PPO')) || ctrls[ctrls.length-1];
makePanel(defA); makePanel(defB);
document.getElementById('play').onclick = e => { playing = !playing; e.target.textContent = playing?'Pause':'Play'; };
document.getElementById('speed').onchange = e => speed = +e.target.value;
document.getElementById('scrub').oninput = e => { t = +e.target.value; draw(); };
const hm = location.hash.match(/t=(\d+)/);
if (hm){ t = Math.min(N-1, +hm[1]); playing = false; document.getElementById('play').textContent = 'Play'; }
let last = performance.now();
function loop(now){ const dt = (now-last)/1000; last = now;
  if (playing){ acc += dt*speed*2; while (acc >= 1){ acc -= 1; t = (t+1)%N; } draw(); }
  requestAnimationFrame(loop); }
draw(); requestAnimationFrame(loop);
// tables
const cols = [['throughput','veh/h',0],['mean_wait','mean wait s',1],['p95_wait','p95 s',0],['max_wait','worst wait s',0],
  ['quiet_mean_wait','Quiet mean s',1],['quiet_max_wait','Quiet worst s',0],['jain_fairness','Jain',3],['spillback_blocked_steps','spillback steps',0]];
let html = '';
for (const [pat, rows] of Object.entries(RES.patterns)){
  html += `<div class="tablewrap"><table><thead><tr><th>${pat}</th>${cols.map(c=>`<th>${c[1]}</th>`).join('')}</tr></thead><tbody>`;
  for (const [n, r] of Object.entries(rows))
    html += `<tr><td>${n}</td>${cols.map(c=>`<td>${r[c[0]].mean.toFixed(c[2])} <span style="color:var(--muted)">&plusmn;${r[c[0]].std.toFixed(c[2])}</span></td>`).join('')}</tr>`;
  html += '</tbody></table></div>';
}
document.getElementById('tables').innerHTML = html;
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="media/demo.html")
    ap.add_argument("--png", default="media/summary.png")
    a = ap.parse_args()
    res = json.load(open(os.path.join(a.runs, "eval_results.json")))
    tr = np.load(os.path.join(a.runs, "demo_traces.npz"))
    names = [str(n) for n in tr["names"]]
    data = {}
    for i, n in enumerate(names):
        data[n] = dict(q=tr[f"q_{i}"].astype(int).ravel().tolist(), c=tr[f"c_{i}"].astype(int).ravel().tolist(),
                       p=tr[f"p_{i}"].astype(int).ravel().tolist(), w=np.round(tr[f"w_{i}"]).astype(int).ravel().tolist())
    slim = dict(episodes=res["episodes"], patterns={p: {n: {k: v for k, v in r.items() if isinstance(v, dict)}
                                                        for n, r in rows.items()} for p, rows in res["patterns"].items()})
    note = ("Demo episode: first held-out rush-hour seed. 'Cars still driving' along a link are drawn as faint dots near "
            "the upstream end; the simulator tracks them as a delay line, not positions. Fixed-time is tuned on training "
            "demand (cycle and split per street); the guard threshold is tuned the same way.")
    html = (HTML.replace("__DATA__", json.dumps(data, separators=(",", ":")))
                .replace("__RES__", json.dumps(slim, separators=(",", ":")))
                .replace("__EPISODES__", str(res["episodes"])).replace("__NOTE__", note))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        f.write(html)
    summary_png(res, a.png)
    print(f"wrote {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB) and {a.png}")


if __name__ == "__main__":
    main()
