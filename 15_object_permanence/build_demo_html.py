"""Build media/demo.html: a self-contained 'shell game' replay (no network needed).

  ../.venv/bin/python build_demo_html.py      # needs results/demo_data_{base,hard}.json from demo.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Where Did It Go?</title>
<style>
:root{--bg:#141418;--panel:#1d1d23;--ink:#e8e8ec;--mute:#9a9aa6;--acc:#f2c14e;--gru:#ff6b6b;--kf:#3ec1d3}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1100px;margin:0 auto;padding:18px 16px 40px}
h1{font-size:24px;margin:4px 0 2px}
p.lead{color:var(--mute);margin:0 0 14px;max-width:760px}
.row{display:flex;flex-wrap:wrap;gap:14px;align-items:flex-start}
.card{background:var(--panel);border-radius:10px;padding:10px}
.card h2{font-size:14px;margin:0 0 6px;color:var(--mute);font-weight:600;letter-spacing:.02em}
canvas{display:block;width:100%;max-width:384px;aspect-ratio:1;image-rendering:pixelated;border-radius:6px;cursor:crosshair}
.ctrl{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:12px 0}
button,select{background:#2a2a33;color:var(--ink);border:1px solid #3a3a46;border-radius:7px;padding:6px 12px;font:inherit;cursor:pointer}
button.primary{background:var(--acc);color:#1a1a1a;border-color:var(--acc);font-weight:600}
input[type=range]{flex:1;min-width:160px}
label{color:var(--mute);user-select:none}
.score{font-variant-numeric:tabular-nums}
.score td{padding:2px 10px 2px 0}
.msg{min-height:1.5em;color:var(--acc);font-weight:600}
.legend span{display:inline-block;margin-right:14px;color:var(--mute)}
.sw{display:inline-block;width:11px;height:11px;border-radius:50%;vertical-align:-1px;margin-right:4px}
</style></head>
<body><main>
<h1>Where did it go?</h1>
<p class="lead">Balls roll behind grey walls. A CNN+GRU tracker (left) and a Kalman filter with a colour detector
(right) watch the same frames and report a position and an uncertainty ellipse (1σ solid, 2σ faint) for every ball,
including the ones they can't see. The dotted ring is the true hidden position. In <b>shell-game mode</b> the replay
stops while a ball is hidden. Click where you think it is, then compare your guess with the two trackers.</p>
<div class="ctrl">
 <select id="set"></select><select id="seq"></select>
 <button id="play" class="primary">Play</button>
 <button id="prev">◀</button><button id="next">▶</button>
 <input id="t" type="range" min="0" max="39" value="0"><span id="tl" class="score">t=0</span>
</div>
<div class="ctrl">
 <label><input type="checkbox" id="truth" checked> show hidden truth</label>
 <label><input type="checkbox" id="shell"> shell-game mode</label>
 <label>speed <select id="speed"><option value="250">slow</option><option value="140" selected>normal</option><option value="70">fast</option></select></label>
</div>
<div class="msg" id="msg"></div>
<div class="row">
 <div class="card"><h2>CNN+GRU tracker (learned)</h2><canvas id="c0" width="384" height="384"></canvas><div id="e0" class="score"></div></div>
 <div class="card"><h2>Kalman filter + bounce (classical)</h2><canvas id="c1" width="384" height="384"></canvas><div id="e1" class="score"></div></div>
 <div class="card" style="min-width:220px"><h2>Shell-game scoreboard</h2>
  <table class="score" id="board"></table>
  <p style="color:var(--mute);font-size:13px">Mean distance (px, frame is 64 px wide) between each guess and the true hidden
  position, over every round you've played.</p>
  <div class="legend"><span><i class="sw" style="background:#fff"></i>you</span><span><i class="sw" style="background:var(--gru)"></i>GRU</span><span><i class="sw" style="background:var(--kf)"></i>Kalman</span></div>
 </div>
</div>
<p class="lead" style="margin-top:18px"><b>base</b>: three distinct colours, constant velocity, wall bounces, with walls wider than in training.
<b>hard</b>: identical yellow balls (identity follows motion only; slot colours here are just labels), ball–ball collisions, and a 30% chance that a ball
changes velocity once it is fully hidden. Sequences come from the held-out long-occlusion test splits.</p>
</main>
<script>
const DATA = __DATA__;
const S = 6, R = 3.5;
const PAL = {base: [[255,90,80],[90,230,110],[110,150,255]], hard: [[255,120,200],[80,230,255],[255,200,60]]};
const $ = id => document.getElementById(id);
let setName = Object.keys(DATA)[0], seq = null, t = 0, timer = null, guess = null, stopT = -1, stopK = -1, solved = false;
const scores = {you: [], gru: [], kalman: []};

function fillSets(){ $('set').innerHTML = Object.keys(DATA).map(k=>`<option>${k}</option>`).join(''); }
function fillSeqs(){ $('seq').innerHTML = DATA[setName].map((s,i)=>`<option value="${i}">${setName} #${s.index}${i==0?' (unselected)':''}</option>`).join(''); }
function load(i){ seq = DATA[setName][i]; t = 0; guess = null; solved = false; pickStop(); draw(); }

function pickStop(){ // middle of the longest completed full occlusion
  stopT = -1; stopK = -1; let best = 0;
  for (let k = 0; k < 3; k++){ if (!seq.present[k]) continue;
    let s = -1;
    for (let tt = 0; tt < seq.vis.length; tt++){
      const h = seq.vis[tt][k] === 0;
      if (h && s < 0) s = tt;
      if (!h && s >= 0){ const L = tt - s; if (s > 0 && L > best){ best = L; stopT = s + Math.floor(L*0.7); stopK = k; } s = -1; }
    }}
}

function renderFrame(ctx){
  const im = ctx.createImageData(64,64), d = im.data, c = seq.colors, bg=[.05,.05,.07];
  for (let y=0;y<64;y++) for (let x=0;x<64;x++){
    let r=bg[0],g=bg[1],b=bg[2];
    for (let k=0;k<3;k++){ if(!seq.present[k]) continue;
      const dx=x+.5-seq.pos[t][k][0], dy=y+.5-seq.pos[t][k][1];
      const a=Math.min(1,Math.max(0,R+.5-Math.hypot(dx,dy)));
      r=r*(1-a)+a*c[k][0]; g=g*(1-a)+a*c[k][1]; b=b*(1-a)+a*c[k][2]; }
    for (const o of seq.occ){ if(o[0]<0) continue;
      const cov=Math.min(1,Math.max(0,Math.min(x+1,o[1])-Math.max(x,o[0])));
      r=r*(1-cov)+cov*.45; g=g*(1-cov)+cov*.45; b=b*(1-cov)+cov*.45; }
    const i=4*(y*64+x); d[i]=r*255; d[i+1]=g*255; d[i+2]=b*255; d[i+3]=255; }
  const off = document.createElement('canvas'); off.width=off.height=64; off.getContext('2d').putImageData(im,0,0);
  ctx.imageSmoothingEnabled=false; ctx.drawImage(off,0,0,384,384);
}

function ellipse(ctx, m, C, k, col, fillA, lineA){
  const a=C[0][0], b=C[0][1], d=C[1][1], tr=(a+d)/2, det=Math.sqrt(Math.max(0,((a-d)/2)**2+b*b));
  const l1=tr+det, l2=Math.max(tr-det,1e-9), theta=0.5*Math.atan2(2*b, a-d);
  if (l1 > 4000) return;
  ctx.beginPath(); ctx.ellipse(m[0]*S, m[1]*S, Math.sqrt(l1)*k*S, Math.sqrt(l2)*k*S, theta, 0, 2*Math.PI);
  ctx.fillStyle=`rgba(${col},${fillA})`; ctx.fill(); ctx.strokeStyle=`rgba(${col},${lineA})`; ctx.lineWidth=2; ctx.stroke();
}
function cross(ctx, m, col, s=6){ ctx.strokeStyle=col; ctx.lineWidth=2; ctx.beginPath();
  ctx.moveTo(m[0]*S-s,m[1]*S); ctx.lineTo(m[0]*S+s,m[1]*S); ctx.moveTo(m[0]*S,m[1]*S-s); ctx.lineTo(m[0]*S,m[1]*S+s); ctx.stroke(); }

function draw(){
  $('t').value=t; $('tl').textContent=`t=${t}`;
  const hideAnswer = $('shell').checked && t===stopT && !solved;
  ['gru','kalman'].forEach((name,ci)=>{
    const ctx=$('c'+ci).getContext('2d'); renderFrame(ctx);
    const P = PAL[setName] || PAL.base; let errs=[];
    for (let k=0;k<3;k++){ if(!seq.present[k]) continue;
      const col=P[k].join(','), m=seq[name].mean[t][k], C=seq[name].cov[t][k], g=seq.pos[t][k];
      const hidden = seq.vis[t][k] < 0.999;
      if (hideAnswer && k===stopK) continue;
      if (hidden && $('truth').checked){ ctx.setLineDash([4,5]); ctx.strokeStyle=`rgba(${col},.75)`; ctx.lineWidth=2;
        ctx.beginPath(); ctx.arc(g[0]*S,g[1]*S,R*S,0,2*Math.PI); ctx.stroke(); ctx.setLineDash([]); }
      ellipse(ctx,m,C,2,col,.08,.45); ellipse(ctx,m,C,1,col,.16,.95); cross(ctx,m,`rgb(${col})`);
      errs.push(Math.hypot(m[0]-g[0],m[1]-g[1]));
    }
    if (guess && (t===stopT)){ cross(ctx,guess,'#fff',9); }
    $('e'+ci).textContent = hideAnswer ? 'hidden ball: ???' : `mean error this frame: ${(errs.reduce((a,b)=>a+b,0)/Math.max(errs.length,1)).toFixed(2)} px`;
  });
  if (hideAnswer){ const P=PAL[setName]; $('msg').innerHTML=`Where is the <span style="color:rgb(${P[stopK].join(',')})">ball #${stopK+1}</span> now? Click on either panel.`; }
  else if (!solved || t!==stopT) $('msg').textContent = $('shell').checked && stopT>=0 && t<stopT ? 'Watch closely…' : '';
}

function step(){ if (t >= seq.pos.length-1){ stop(); return; }
  t++; if ($('shell').checked && t===stopT && !solved){ stop(); } draw(); }
function stop(){ clearInterval(timer); timer=null; $('play').textContent='Play'; }
function play(){ if (timer){ stop(); return; } if (t>=seq.pos.length-1){ t=0; solved=false; guess=null; }
  timer=setInterval(step, +$('speed').value); $('play').textContent='Pause'; }

function onClick(ev){
  if (!($('shell').checked && t===stopT && !solved)) return;
  const r=ev.target.getBoundingClientRect(); guess=[(ev.clientX-r.left)/r.width*64,(ev.clientY-r.top)/r.height*64];
  const g=seq.pos[t][stopK], e=m=>Math.hypot(m[0]-g[0],m[1]-g[1]);
  const you=e(guess), gr=e(seq.gru.mean[t][stopK]), kf=e(seq.kalman.mean[t][stopK]);
  scores.you.push(you); scores.gru.push(gr); scores.kalman.push(kf); solved=true;
  $('msg').textContent=`You were ${you.toFixed(1)} px off · GRU ${gr.toFixed(1)} px · Kalman ${kf.toFixed(1)} px. Press Play to watch it come back out.`;
  board(); draw(); $('msg').textContent=`You were ${you.toFixed(1)} px off · GRU ${gr.toFixed(1)} px · Kalman ${kf.toFixed(1)} px. Press Play to watch it come back out.`;
}
function board(){ const mean=a=>a.length?(a.reduce((x,y)=>x+y,0)/a.length).toFixed(2):'–';
  $('board').innerHTML=`<tr><td>rounds</td><td>${scores.you.length}</td></tr><tr><td>you</td><td>${mean(scores.you)}</td></tr><tr><td style="color:var(--gru)">GRU</td><td>${mean(scores.gru)}</td></tr><tr><td style="color:var(--kf)">Kalman</td><td>${mean(scores.kalman)}</td></tr>`; }

fillSets(); fillSeqs(); load(0); board();
$('set').onchange=e=>{ setName=e.target.value; fillSeqs(); load(0); };
$('seq').onchange=e=>load(+e.target.value);
$('play').onclick=play; $('prev').onclick=()=>{ if(t>0){t--; draw();} }; $('next').onclick=()=>{ if(t<seq.pos.length-1){t++; draw();} };
$('t').oninput=e=>{ t=+e.target.value; draw(); };
$('truth').onchange=draw; $('shell').onchange=()=>{ solved=false; guess=null; t=0; draw(); };
$('speed').onchange=()=>{ if(timer){ stop(); play(); } };
$('c0').onclick=onClick; $('c1').onclick=onClick;
</script></body></html>
"""


def main():
    data = {}
    for v in ["base", "hard"]:
        p = os.path.join(HERE, "results", f"demo_data_{v}.json")
        if os.path.exists(p):
            with open(p) as f:
                data[v] = json.load(f)
    html = TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    os.makedirs(os.path.join(HERE, "media"), exist_ok=True)
    out = os.path.join(HERE, "media", "demo.html")
    with open(out, "w") as f:
        f.write(html)
    print("wrote", out, f"{len(html)/1e6:.2f} MB", {k: len(v) for k, v in data.items()})


if __name__ == "__main__":
    main()
