"""Offline measured-result table and trace replay; no MLX, plots or network."""
import argparse
from collections import defaultdict
import html
import json
from pathlib import Path

import numpy as np
from r5_core import atomic_json, quantiles


def collect(run):
    root = Path("results/r5") / run
    status_path = root / "run_status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text())
    else:
        state = json.loads(Path(f"checkpoints/r5_{run}_state.json").read_text())
        status = dict(complete=False, done=state["step"], planned=len(state["config"]["plan"]["jobs"]),
                      reason="No terminal run status; interrupted run")
    episodes = []
    for path in sorted(root.glob("episode_*.json")):
        item = json.loads(path.read_text())
        item["file"] = str(path)
        episodes.append(item)
    groups = defaultdict(list)
    for e in episodes:
        j = e["job"]
        groups[tuple(j[k] for k in ("event", "engine", "policy", "load", "delay_ms"))].append(e)
    summaries = []
    for group, es in sorted(groups.items()):
        ages = [a["age_ms"] for e in es for a in e["actions"]]
        scheduled = sum(e["metrics"]["scheduled"] for e in es)
        ontime = sum(e["metrics"]["on_time"] for e in es)
        applied = sum(e["metrics"]["applied"] for e in es)
        if group[0] == "car":
            task = dict(finish_fraction=float(np.mean([e["task"]["finished"] for e in es])),
                        crash_fraction=float(np.mean([e["task"]["crashed"] for e in es])),
                        mean_progress=float(np.mean([e["task"]["progress_fraction"] for e in es])))
        else:
            task = {k: float(np.mean([e["task"][k] for e in es]))
                    for k in ("pursuit_error_px", "tracking_error_px", "within_5px_fraction")}
        summaries.append(dict(group=list(group), episodes=len(es), files=[e["file"] for e in es],
                              age_ms=quantiles(ages), deadline_miss_fraction=1-ontime/max(1, scheduled),
                              active_age_ms=quantiles([a["age_ms"] for e in es for a in e["actions"]
                                                       if a.get("task_active", True)]),
                              action_coverage=applied/max(1, scheduled), task=task))
    return dict(status=status, summaries=summaries), episodes


PAGE = r'''<!doctype html><meta charset="utf-8"><title>Round 5: observation to action</title>
<style>body{background:#101820;color:#e5edf5;font:15px system-ui;margin:28px;max-width:1300px}
table{border-collapse:collapse;font-size:12px;width:100%}td,th{padding:7px;border-bottom:1px solid #344553;text-align:left}
canvas{background:#182733;max-width:100%;border-radius:10px}button,select{padding:8px;margin:5px}
code{color:#8cddd0}.warn{color:#ffce80}h1{font-size:26px}</style>
<h1>Observation → inference → action</h1><p class="warn">__STATUS__</p>
<p>Same frozen weights and seed within each comparison. Ambient includes the rest of the Mac;
loaded adds one 1024² MLX matmul worker. Injected delay is a separate stress scenario.
Age percentiles pool applied actions across the fixed horizon, including after car termination;
raw/summary JSON also separates active-task age. Coverage and misses count dropped and unfinished frames.
No wins are inferred from latency alone. Small seed counts limit generalization.</p>
<p>Car: 10 Hz sensors, ≤10 ms integration, strict walls, 40 s timeout. Tracker: 30 Hz,
40 frames, 90 px/s pursuit cursors; error includes startup and occlusion. Requested Core ML units
can fall back to CPU. Every raw receipt contains uptime, battery/power and stage timings.</p>
<select id="episode"></select><button id="replay">Replay</button><span id="info"></span><br>
<canvas id="view" width="720" height="440"></canvas><p id="details"></p>
<table><thead><tr><th>Event / engine / policy / load / delay ms</th><th>Seeds</th><th>Age p50 / p95 / p99 ms</th>
<th>Misses / coverage</th><th>Task outcome</th></tr></thead><tbody>__ROWS__</tbody></table>
<script>
const episodes=__DATA__;
const sel=document.getElementById('episode'),canvas=document.getElementById('view'),ctx=canvas.getContext('2d');
let current=0,start=0,playing=true;
episodes.forEach((e,i)=>{let o=document.createElement('option');o.value=i;
o.textContent=[e.job.event,e.job.engine,e.job.policy,e.job.load,'delay '+e.job.delay_ms,'seed '+e.job.seed].join(' · ');
sel.appendChild(o)});
function choose(){current=Number(sel.value);start=performance.now();playing=true;
document.getElementById('details').textContent=episodes[current].file+' | '+JSON.stringify(episodes[current].task)}
sel.onchange=choose;document.getElementById('replay').onclick=choose;
function path(points,project,color,close=false){ctx.strokeStyle=color;ctx.lineWidth=2;ctx.beginPath();
points.forEach((p,i)=>{let q=project(p);i?ctx.lineTo(...q):ctx.moveTo(...q)});if(close)ctx.closePath();ctx.stroke()}
function dot(p,project,color,r=5){let q=project(p);ctx.fillStyle=color;ctx.beginPath();ctx.arc(...q,r,0,Math.PI*2);ctx.fill()}
function paint(now){if(!episodes.length)return;const e=episodes[current],t=Math.min((now-start)/1000,e.metrics.horizon_s);
const frames=e.trace.filter(x=>x[0]<=t),f=frames[frames.length-1]||e.trace[0];
ctx.clearRect(0,0,720,440);if(!f){requestAnimationFrame(paint);return}
if(e.job.event==='car'){const points=e.geometry.left.concat(e.geometry.right),xs=points.map(p=>p[0]),ys=points.map(p=>p[1]);
const xmin=Math.min(...xs),ymin=Math.min(...ys),scale=Math.min(660/(Math.max(...xs)-xmin),380/(Math.max(...ys)-ymin));
const p=a=>[30+(a[0]-xmin)*scale,410-(a[1]-ymin)*scale];
path(e.geometry.left,p,'#7298a7',true);path(e.geometry.right,p,'#7298a7',true);
path(frames.map(x=>[x[1],x[2]]),p,'#64e2bf');dot([f[1],f[2]],p,f[5]?'#ff706d':'#f9d96c',6);
}else{const p=a=>[150+a[0]*6,25+a[1]*6],colors=['#fa665b','#65df8b','#729eff'];
e.geometry.occ.filter(o=>o[0]>=0).forEach(o=>{ctx.fillStyle='#65747c';ctx.fillRect(150+o[0]*6,25,(o[1]-o[0])*6,384)});
for(let k=0;k<3;k++)if(e.geometry.present[k]){dot(f[1][k],p,colors[k],9);dot(f[2][k],p,'#fff',4);
let q=p(f[3][k]);ctx.strokeStyle=colors[k];ctx.strokeRect(q[0]-7,q[1]-7,14,14)}}
document.getElementById('info').textContent=t.toFixed(2)+' s — truth: color / cursor: white / estimate: square';
requestAnimationFrame(paint)}
if(episodes.length){let failure=episodes.findIndex(e=>e.task.crashed);if(failure>=0)sel.value=failure;choose();requestAnimationFrame(paint)}
</script>'''


def render_report(run, out):
    summary, episodes = collect(run)
    atomic_json(Path("results/r5") / run / "summary.json", summary)
    rows = []
    for r in summary["summaries"]:
        fmt = lambda x: "not measured" if x is None else f"{x:.3f}"
        age = " / ".join(fmt(r["age_ms"][f"p{p}"]) for p in (50, 95, 99))
        cells = [" / ".join(map(str, r["group"])), str(r["episodes"]), age,
                 f"{r['deadline_miss_fraction']:.1%} / {r['action_coverage']:.1%}",
                 json.dumps(r["task"])]
        rows.append("<tr>"+"".join("<td>"+html.escape(c)+"</td>" for c in cells)+"</tr>")
    compact = [{k: e[k] for k in ("job", "task", "trace", "geometry", "file", "metrics")} for e in episodes]
    status = summary["status"]
    label = ("COMPLETE" if status["complete"] else "PARTIAL — do not compare unmatched cells")
    label += f" · {status['done']}/{status['planned']} episodes · raw receipts: results/r5/{run}/"
    output = PAGE.replace("__STATUS__", html.escape(label)).replace("__ROWS__", "".join(rows))
    output = output.replace("__DATA__", json.dumps(compact, allow_nan=False).replace("<", "\\u003c"))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(output)
    return len(episodes)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    n = render_report(args.run, args.out)
    print(f"REPORT {args.out}: {n} episode traces (raw failures retained)")
