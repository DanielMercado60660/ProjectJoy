"""Headless self-contained race replay from the selected round-5 checkpoints."""
import argparse
import json
from hashlib import sha256
from pathlib import Path
import numpy as np
from r5_common import Budget
from r5_env import make_bank, suite
from r5_eval import controllers, run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='r5_seed0')
    ap.add_argument('--max-minutes', type=float, default=.5)
    ap.add_argument('--max-steps', type=int, default=600)
    a = ap.parse_args()
    budget = Budget(a.max_minutes)
    receipt = Path(f'results/r5/{a.tag}_test.json')
    evaluation = json.loads(receipt.read_text())
    if not evaluation['complete'] or evaluation['config']['max_steps'] != a.max_steps:
        raise ValueError('finish evaluation first, and use its exact --max-steps')
    for source, digest in evaluation['config']['source_sha256'].items():
        if sha256(Path(source).read_bytes()).hexdigest() != digest:
            raise ValueError('evaluation source changed: preserve old receipt and evaluate a fresh tag')
    n = evaluation['config']['n']
    bank = make_bank('test', n)
    all_ids, all_modes, all_labels, seed = suite(bank, 'test')
    chosen = []
    rows = evaluation['methods']['rnn']['episodes']
    for g in dict.fromkeys(all_labels):
        candidates = np.flatnonzero(all_labels == g)
        failures = [i for i in candidates if rows is not None and rows[i]['crashed']]
        chosen.append(failures[0] if failures else candidates[0])
    ids, modes, labels = all_ids[chosen], all_modes[chosen], all_labels[chosen]
    selected_tracks = list(dict.fromkeys(ids.tolist()))
    data = dict(groups=labels.tolist(), tracks=[dict(center=bank.tracks[i].center.tolist(),
                width=bank.tracks[i].half_width, seed=bank.tracks[i].seed) for i in selected_tracks],
                ids=[selected_tracks.index(int(i)) for i in ids], methods={}, dt=.1, default_group=0,
                note='Per group: first recorded recurrent crash, otherwise first test episode. Selected checkpoints; exact keyed test surfaces. Paint is independent of grip. Grip HUD is evaluator-only.')
    for name, factory, path in controllers(a.tag):
        if sha256(Path(path).read_bytes()).hexdigest() != evaluation['config']['checkpoints'][name]:
            raise ValueError('selected checkpoint changed since evaluation')
        # Keep the headline batch shape too: float32 BLAS kernels can vary by size.
        r = run(factory(), bank, all_ids, all_modes, seed, a.max_steps, record=True, deadline=budget)
        original = evaluation['methods'][name]['episodes']
        assert all(bool(r['finished'][i]) == original[i]['finished'] and
                   bool(r['crashed'][i]) == original[i]['crashed'] for i in chosen)
        # Keep failures and stopped cars; do not select only successful trajectories.
        data['methods'][name] = dict(checkpoint=path, frames=np.round(r['trajectory'][:, chosen], 4).tolist(),
                                    finished=r['finished'][chosen].tolist(), crashed=r['crashed'][chosen].tolist(),
                                    events=[r['events'].rows(i) for i in chosen])
        if name == 'rnn' and r['crashed'][chosen].any():
            data['default_group'] = int(np.flatnonzero(r['crashed'][chosen])[0])
    if a.tag.startswith('r5_smoke'):
        data['note'] = 'SMOKE ONLY; no round-5 learning conclusions. ' + data['note']
    html = '''<!doctype html><html><meta charset="utf-8"><title>R5: Discover the grip</title>
<style>body{font:16px system-ui;background:#111827;color:#e5e7eb;max-width:1000px;margin:25px auto}canvas{width:100%;background:#172334;border-radius:12px}select,button,input{font:inherit;margin:6px;padding:5px}#info{white-space:pre-wrap}</style>
<h1>Discover the road's grip</h1><p id="note"></p>
<label>Scene <select id="scene"></select></label><label>Driver <select id="driver"></select></label>
<button id="play">Pause</button><input id="seek" type="range" min="0" value="0" step="1"><span id="clock"></span>
<canvas width="1000" height="650" id="race"></canvas><p id="info"></p>
<script>const D=DATA;
const scene=document.getElementById('scene'),driver=document.getElementById('driver'),canvas=document.getElementById('race'),ctx=canvas.getContext('2d'),seek=document.getElementById('seek');
D.groups.forEach((v,i)=>scene.add(new Option(v,i)));Object.keys(D.methods).forEach(v=>driver.add(new Option(v,v)));scene.value=D.default_group;document.getElementById('note').textContent=D.note;
let playing=true,t=0;document.getElementById('play').onclick=()=>{playing=!playing;document.getElementById('play').textContent=playing?'Pause':'Play'};seek.oninput=()=>{t=+seek.value;playing=false;draw()};scene.onchange=driver.onchange=()=>{t=0;draw()};
function draw(){let g=+scene.value,m=D.methods[driver.value],tr=D.tracks[D.ids[g]],f=m.frames[Math.min(t,m.frames.length-1)][g];seek.max=m.frames.length-1;seek.value=t;
let xs=tr.center.map(p=>p[0]),ys=tr.center.map(p=>p[1]),xmin=Math.min(...xs)-12,xmax=Math.max(...xs)+12,ymin=Math.min(...ys)-12,ymax=Math.max(...ys)+12,s=Math.min(940/(xmax-xmin),590/(ymax-ymin));let X=x=>30+(x-xmin)*s,Y=y=>620-(y-ymin)*s;
ctx.clearRect(0,0,1000,650);ctx.beginPath();tr.center.forEach((p,i)=>i?ctx.lineTo(X(p[0]),Y(p[1])):ctx.moveTo(X(p[0]),Y(p[1])));ctx.closePath();ctx.lineJoin='round';ctx.lineWidth=2*tr.width*s;ctx.strokeStyle=D.ids[g]%2?'#495263':'#46566b';ctx.stroke();ctx.lineWidth=1;ctx.setLineDash([5,9]);ctx.strokeStyle='#8996a8';ctx.stroke();ctx.setLineDash([]);
ctx.beginPath();m.frames.slice(0,t+1).forEach((a,i)=>i?ctx.lineTo(X(a[g][0]),Y(a[g][1])):ctx.moveTo(X(a[g][0]),Y(a[g][1])));ctx.strokeStyle='#5eead4';ctx.lineWidth=2;ctx.stroke();
ctx.save();ctx.translate(X(f[0]),Y(f[1]));ctx.rotate(-f[2]);ctx.fillStyle='#fbbf24';ctx.fillRect(-2*s,-s,4*s,2*s);ctx.restore();
let status=m.finished[g]?'FINISH':m.crashed[g]?'CRASH / DNF':'TIME LIMIT / DNF';document.getElementById('clock').textContent=(t*.1).toFixed(1)+' s';document.getElementById('info').textContent='Saved outcome: '+status+' | speed '+f[3].toFixed(1)+' m/s | hidden grip (diagnostic) '+f[6].toFixed(2)+' | lateral fraction '+f[7].toFixed(2)+'\\nCheckpoint: '+m.checkpoint+'\\nEvents: '+JSON.stringify(m.events[g]);}
setInterval(()=>{if(playing){t=(t+1)%D.methods[driver.value].frames.length;draw()}},100);draw();</script></html>'''
    html = html.replace('DATA', json.dumps(data, separators=(',', ':'), allow_nan=False), 1)
    path = Path(f'media/{a.tag}_demo.html')
    path.parent.mkdir(exist_ok=True)
    path.write_text(html)
    print(json.dumps(dict(demo=str(path), seconds=budget.elapsed, methods=list(data['methods']))))


if __name__ == '__main__':
    main()
