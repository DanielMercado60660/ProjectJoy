"""File-only reporting and a no-OpenGL, offline animated failure demo."""
import json
import os
from pathlib import Path

import numpy as np

from r5_core import DT, TARGETS, TASKS


def report(tag, res, deadline):
    deadline.check()
    data = json.loads((res/'eval.json').read_text())
    ledger = json.loads((res/'training.json').read_text())['ledger']
    cpg = json.loads((res/'cpg.json').read_text())
    examples = json.loads((res/'demo_traces.json').read_text())
    rows = data['rows']
    lines = [f'# Round 5 run {tag}', '',
             'One training seed per run. Fresh test resets are primary; legacy resets are secondary.',
             'Every table below is derived from `eval.json`, `training.json` or `cpg.json` in this directory.',
             'Rewards/targets differ from rounds 1–4, so raw returns are not comparable across rounds.', '']
    for split in ('test', 'legacy'):
        lines += [f'## {split}: final three-skill repertoire', '',
                  '| Method | Isolated tracking | Speed MAE | Raw return / episode | Switch failure | Fall proxy / switch | Stall / switch | Capped settle seconds |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for method in ('adapter','dagger','joint','oracle','cpg','zero'):
            isolated = [r for r in rows if r['split']==split and r['stage']==3 and r['method']==method and r['kind']=='isolated']
            switches = [r for r in rows if r['split']==split and r['stage']==3 and r['method']==method and r['kind']=='transition' and r['switch']]
            if not isolated or not switches:
                continue
            avg = lambda group, key: np.mean([r[key] for r in group])
            lines.append(f'| {method} | {avg(isolated,"tracking_fraction"):.4f} | {avg(isolated,"speed_mae"):.3f} | '
                         f'{avg(isolated,"return_"):.2f} | {avg(switches,"failure"):.4f} | '
                         f'{avg(switches,"fall_proxy"):.4f} | {avg(switches,"stall"):.4f} | {avg(switches,"capped_settle_s"):.3f} |')
        lines += ['']
    lines += ['## Retention (fresh tests; acquisition snapshot is the anchor)', '',
              '| Method | Task | Stage | Acquisition tracking | Current tracking | Loss | Retained ratio |',
              '|---|---|---:|---:|---:|---:|---:|']
    for r in data['retention']:
        ratio = 'unqualified (<5% at acquisition)' if r['retained_ratio'] is None else f'{r["retained_ratio"]:.4f}'
        lines.append(f'| {r["method"]} | {r["task"]} | {r["stage"]} | {r["acquisition"]:.4f} | {r["competence"]:.4f} | {r["loss"]:.4f} | {ratio} |')
    lines += ['', '## Completed training cost (common base charged to each arm)', '',
              'Credits = environment steps + teacher query examples. Equal credits are **not equal FLOPs or wall time**.',
              'Wall times include checkpoint I/O; aborted work after the last checkpoint is only visible in invocation logs.', '',
              '| Method | Stage | Env steps | Of which specialists | Teacher queries | Credits | Gradient steps | Gradient examples | Seconds |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for method, stages in ledger.items():
        for r in stages:
            c = r['cost']
            lines.append(f'| {method} | {r["stage"]} | {c["env_steps"]} | {c["specialist_env_steps_included"]} | '
                         f'{c["teacher_queries"]} | {r["credits"]} | {c["gradient_steps"]} | '
                         f'{c["gradient_examples"]} | {c["wall_s"]:.2f} |')
    lines += ['', f'CPG: {cpg["train_steps"]} search steps + {cpg["val_steps"]} validation steps; '
              f'{cpg["wall_s"]:.2f} seconds. Independently budgeted generous baseline.',
              f'Evaluation: {data["env_steps"]} completed environment steps; {data["wall_s"]:.2f} seconds.', '',
              '## Memory', '', '| Model | Actor parameters | Actor bytes | Training parameters | Checkpoint bytes (including replay/Adam) |',
              '|---|---:|---:|---:|---:|']
    for key, r in data['sizes'].items():
        lines.append(f'| {key} | {r["actor_parameters"]} | {r["actor_bytes"]} | '
                     f'{r.get("training_parameters", "n/a")} | {r.get("checkpoint_bytes", "n/a")} |')
    lines += ['', 'Peak process RSS/GPU memory: not measured by this report. Actor bytes are exact array storage, not resident memory.',
              'Coverage is 100% of scheduled actions. Non-settling transitions receive the entire segment duration.',
              'Fall proxy: torso height <0.3m or wrapped pitch >1.2rad for 5 consecutive steps; HalfCheetah does not terminate on falls.',
              'The demo deliberately shows each method’s worst fresh-test transition episode (failures first, MAE tie-break).', '']
    (res/'summary.md').write_text('\n'.join(lines))
    media = Path('media')
    media.mkdir(exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str((res/'.mplconfig').resolve())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = list(examples)
    fig, axes = plt.subplots(len(names), 1, figsize=(10, 2.1*len(names)), squeeze=False)
    for ax, method in zip(axes[:,0], names):
        ep = examples[method]['episode']
        trace = ep['trace']
        times = np.arange(len(trace['v'])) * DT
        ax.plot(times, trace['v'], label='actual velocity', lw=1)
        ax.step(times, TARGETS[trace['task']], label='target velocity', where='post', lw=1)
        ax.set_title(f'{method}: worst test transition, seed {ep["seed"]}, order ' + ' → '.join(TASKS[t] for t in ep['order']))
        ax.set_ylabel('m/s')
        ax.grid(alpha=.2)
    axes[0,0].legend(loc='upper right')
    axes[-1,0].set_xlabel('seconds; no simulator reset at command switches')
    fig.tight_layout()
    fig.savefig(media/f'r5_{tag}_transitions.png', dpi=120)
    plt.close(fig)
    deadline.check()
    # Compact but actual MuJoCo body trajectories, rendered as stick figures in SVG.
    bundle = {}
    for name, example in examples.items():
        ep = example['episode']
        tr = ep['trace']
        bundle[name] = dict(seed=ep['seed'], order=ep['order'], parents=ep['parents'],
                            frames=[dict(body=np.round(tr['body'][t], 4).tolist(), v=round(tr['v'][t], 3),
                                         tid=tr['task'][t], t=round(t*DT,2)) for t in range(0, len(tr['v']), 2)])
    html = '''<!doctype html><meta charset="utf-8"><title>R5 skill transitions</title>
<style>body{font:16px system-ui;background:#101b2b;color:#eff5ff;margin:28px;max-width:1100px}select,button,input{font:inherit;margin:6px}svg{background:#192b41;width:100%;border-radius:12px}p{line-height:1.5}small{color:#b6c7dd}</style>
<h1>Three skills, one body</h1><p>Worst held-out transition for each method. The simulator is never reset at a switch. These are saved MuJoCo body positions, shown as a stick figure; the camera follows the torso.</p>
<select id="method"></select><button id="play">Pause</button><input id="frame" type="range" min="0" value="0" style="width:45%"><p id="readout"></p>
<svg id="view" viewBox="0 0 1000 300" role="img" aria-label="Animated cheetah body"><line x1="0" y1="260" x2="1000" y2="260" stroke="#73869b"/></svg>
<p><small>Failure includes a sustained fall proxy, stall, or never settling within ±0.5 m/s for 10 steps. A visibly preserved old gait does not prove the new skill was acquired. Tables: results/r5/TAG/summary.md.</small></p>
<script>const data=DATA;const targets=[3,-3,1], names=['forward','backward','slow cruise'];
const sel=document.querySelector('#method'),slider=document.querySelector('#frame'),view=document.querySelector('#view');
for(const name of Object.keys(data)){const o=document.createElement('option');o.value=o.textContent=name;sel.append(o)}
let idx=0,playing=true;function draw(){const d=data[sel.value];idx%=d.frames.length;const f=d.frames[idx], b=f.body, cx=b[1][0];
view.querySelectorAll('.body').forEach(n=>n.remove());for(let j=1;j<b.length;j++){const p=d.parents[j];if(p===0)continue;
const line=document.createElementNS('http://www.w3.org/2000/svg','line');line.setAttribute('class','body');
for(const [k,v]of Object.entries({x1:500+(b[p][0]-cx)*160,y1:260-b[p][1]*160,x2:500+(b[j][0]-cx)*160,y2:260-b[j][1]*160,stroke:'#58dfbd','stroke-width':9,'stroke-linecap':'round'}))line.setAttribute(k,v);view.append(line)}
slider.max=d.frames.length-1;slider.value=idx;document.querySelector('#readout').textContent=`${sel.value} · seed ${d.seed} · ${f.t.toFixed(1)}s · command ${names[f.tid]} (${targets[f.tid]} m/s) · actual ${f.v.toFixed(2)} m/s`;}
sel.onchange=()=>{idx=0;draw()};slider.oninput=()=>{idx=+slider.value;draw()};document.querySelector('#play').onclick=e=>{playing=!playing;e.target.textContent=playing?'Pause':'Play'};setInterval(()=>{if(playing){idx++;draw()}},100);draw();</script>'''
    html = html.replace('TAG', tag).replace('DATA', json.dumps(bundle))
    (media/f'r5_{tag}_demo.html').write_text(html)
    print(f'R5_REPORT_COMPLETE {res}/summary.md media/r5_{tag}_demo.html', flush=True)
