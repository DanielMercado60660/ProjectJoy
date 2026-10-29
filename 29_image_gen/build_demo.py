"""Build media/demo.html: a self-contained 'Monster Factory' gallery with baked samples (no network needed).

    ../.venv/bin/python build_demo.py
Reads runs/*/samples_final.npz, runs/flow_cond/combos.npz (optional), results/main_eval.json and media/*.png|gif.
"""
import base64, io, json, os

import numpy as np
from PIL import Image

import data as D

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
NICE = {"vae": "VAE", "gan": "DCGAN", "ddpm": "DDPM", "flow": "Flow matching", "flow_cond": "Conditional flow",
        "flow_x2": "Flow (2x budget)", "ddpm_x2": "DDPM (2x)", "gan_x2": "DCGAN (2x)", "vae_x2": "VAE (2x)",
        "pca": "PCA-Gaussian"}
TILE, PER_ROW = 32, 32


def b64png(im):
    buf = io.BytesIO(); im.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def b64file(p):
    mime = {"gif": "image/gif", "mp4": "video/mp4"}.get(p.rsplit(".", 1)[-1], "image/png")
    return f"data:{mime};base64," + base64.b64encode(open(p, "rb").read()).decode()


def sprite(x):
    n = len(x); rows = int(np.ceil(n / PER_ROW))
    g = np.full((rows * TILE, PER_ROW * TILE, 3), 255, np.uint8)
    for i in range(n):
        r, c = divmod(i, PER_ROW)
        g[r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE] = x[i]
    return Image.fromarray(g)


def main(n_per=256):
    models = {}
    for m in ["flow", "ddpm", "gan", "vae", "flow_x2", "ddpm_x2", "gan_x2", "vae_x2"]:
        p = f"runs/{m}/samples_final.npz"
        if os.path.exists(p):
            z = np.load(p)
            models[m] = {"sprite": b64png(sprite(z["x"][:n_per])), "n": int(min(n_per, len(z["x"]))),
                         "pred": z["pred"][:n_per].tolist(), "valid": z["valid"][:n_per].astype(int).tolist()}
    if os.path.exists("results/samples_pca.npz"):
        z = np.load("results/samples_pca.npz")
        models["pca"] = {"sprite": b64png(sprite(z["x"][:n_per])), "n": n_per, "pred": [], "valid": []}
    combos = None
    if os.path.exists("runs/flow_cond/combos.npz"):
        z = np.load("runs/flow_cond/combos.npz")  # x: (1296*k, 32,32,3) ordered by combo id, pred, k
        k = int(z["k"])  # demo keeps 1 sample per combo to stay small
        combos = {"sprite": b64png(sprite(z["x"][::k])), "k": 1, "pred": z["pred"][::k].tolist()}
    real = D.load(32)[0][:n_per]
    models_real = {"sprite": b64png(sprite(real)), "n": len(real)}
    ev = json.load(open("results/main_eval.json")) if os.path.exists("results/main_eval.json") else {}
    if os.path.exists("results/main_eval_x2.json"):
        ev.update({k: v for k, v in json.load(open("results/main_eval_x2.json")).items() if k.endswith("_x2")})
    figs = {k: b64file(f"media/{f}") for k, f in [("table", "comparison_table.png"), ("ql", "quality_latency.png"),
                                                   ("train", "training_progress.mp4"), ("traj", "trajectory.mp4"),
                                                   ("stab", "stability.png"), ("scale", "budget_scaling.png"), ("cond", "cond_attribute_sweep.png"),
                                                   ("mem", "memorization_pairs.png")]
            if os.path.exists(f"media/{f}") and os.path.getsize(f"media/{f}") < 6e6}
    payload = {"models": models, "real": models_real, "combos": combos, "attrs": D.ATTRS, "eval": ev,
               "nice": NICE, "tile": TILE, "per_row": PER_ROW}
    html = TEMPLATE.replace("__PAYLOAD__", json.dumps(payload)).replace("__FIGS__", json.dumps(figs))
    open("media/demo.html", "w").write(html)
    print("media/demo.html", f"{os.path.getsize('media/demo.html') / 1e6:.1f} MB")


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Monster Factory</title>
<style>
:root{--bg:#fbf8f3;--fg:#1d1b19;--muted:#6b645c;--card:#fff;--line:#e6dfd5;--acc:#2f7d4f;--acc2:#b8452f}
@media (prefers-color-scheme:dark){:root{--bg:#1a1917;--fg:#eee9e2;--muted:#a59d93;--card:#24221f;--line:#3a3631;--acc:#5cc08a;--acc2:#e07a62}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,system-ui,Segoe UI,Helvetica,Arial,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:30px;margin:0 0 4px}h2{font-size:20px;margin:36px 0 8px}.sub{color:var(--muted);margin:0 0 18px}
.tabs{display:flex;flex-wrap:wrap;gap:6px;margin:12px 0}
button{font:inherit;border:1px solid var(--line);background:var(--card);color:var(--fg);padding:7px 12px;border-radius:8px;cursor:pointer}
button.on{background:var(--acc);border-color:var(--acc);color:#fff}button:hover{border-color:var(--acc)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(80px,1fr));gap:6px}
.t{aspect-ratio:1;border-radius:6px;image-rendering:pixelated;background-repeat:no-repeat;position:relative;border:1px solid var(--line)}
.t.bad{outline:2px dashed var(--acc2);outline-offset:-3px}
.t:hover::after{content:attr(data-l);position:absolute;left:0;right:0;bottom:-2px;transform:translateY(100%);background:var(--fg);color:var(--bg);font-size:11px;padding:3px 5px;border-radius:4px;z-index:5;white-space:pre}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px;margin:10px 0}
.stats{display:flex;flex-wrap:wrap;gap:14px;color:var(--muted);font-size:13px;margin-top:8px}.stats b{color:var(--fg)}
select{font:inherit;padding:6px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--fg)}
.pick{display:flex;flex-wrap:wrap;gap:10px;align-items:end}.pick label{display:flex;flex-direction:column;font-size:12px;color:var(--muted)}
img.fig,video.fig{max-width:100%;display:block;border-radius:8px;border:1px solid var(--line);background:#fff}
.big{display:flex;gap:16px;flex-wrap:wrap;align-items:center}.big .t{width:160px}
</style></head><body><main>
<h1>Monster Factory</h1>
<p class="sub">Four generators trained from scratch in MLX on 40,000 procedurally drawn 32&times;32 monsters, each under the same two caps (2.5 PFLOP of training compute or 25 training minutes, whichever came first). DDPM and flow matching share one U-Net and hit the FLOP cap at identical steps; the cheaper VAE and GAN hit the wall-clock cap first (0.8 and 1.2 PFLOP), so only DDPM vs flow is compute-matched. Every tile was baked offline by the trained model, so this page needs no network. Hover a tile to see the attributes our evaluator classifier reads off it. A dashed tile means the classifier was unsure about at least one attribute. Tabs marked 2x kept training to double the budget.</p>
<div class="tabs" id="tabs"></div>
<div class="card"><div class="grid" id="grid"></div><div class="stats" id="stats"></div>
<div style="margin-top:10px"><button id="shuffle">Generate another batch &#8635;</button></div></div>
<div id="condsec"></div>
<h2>How they compare</h2><div id="figs"></div>
</main><script>
const P=__PAYLOAD__, F=__FIGS__;
const A=P.attrs, AN=Object.keys(A);
function tile(src,i,label,bad){const d=document.createElement('div');d.className='t'+(bad?' bad':'');
 const r=Math.floor(i/P.per_row),c=i%P.per_row;d.style.backgroundImage=`url(${src})`;
 d.style.backgroundSize=`${P.per_row*100}% auto`;d.style.backgroundPosition=`${c/(P.per_row-1)*100}% ${r/(Math.ceil(nOf(src)/P.per_row)-1||1)*100}%`;
 if(label)d.dataset.l=label;return d}
const counts=new Map();function nOf(src){return counts.get(src)}
function lab(p){return p&&p.length?AN.map((a,j)=>A[a][p[j]]).join(' · '):''}
let cur='flow';
function show(m){cur=m;document.querySelectorAll('#tabs button').forEach(b=>b.classList.toggle('on',b.dataset.m===m));
 const M=m==='real'?P.real:P.models[m];counts.set(M.sprite,M.n);const g=document.getElementById('grid');g.innerHTML='';
 const idx=[...Array(M.n).keys()].sort(()=>Math.random()-.5).slice(0,48);
 idx.forEach(i=>g.appendChild(tile(M.sprite,i,M.pred?lab(M.pred[i]):'',M.valid&&M.valid.length&&!M.valid[i])));
 const e=P.eval[m==='real'?'real_reference':m];const s=document.getElementById('stats');
 s.innerHTML=e?`<span title="all 6 classifier heads &gt;0.9 confident: a crispness proxy, not human recognizability (a blurry PCA baseline scores 14.6%)">classifier-valid <b>${(e.valid*100).toFixed(1)}%</b></span><span>attribute combos covered <b>${e.coverage}</b>/1296</span><span title="Frechet distance in our attribute classifier's feature space (5000 samples); not FID, not comparable to published FIDs">CFD <b>${e.cfd.toFixed(2)}</b></span><span>precision <b>${e.precision.toFixed(2)}</b> / recall <b>${e.recall.toFixed(2)}</b></span>`+(e.mem_rate!=null?`<span>memorized <b>${(e.mem_rate*100).toFixed(2)}%</b></span>`:'')+(e.sample_sec_per_1k?`<span>${e.sample_sec_per_1k.toFixed(2)} s / 1000 samples</span>`:''):''}
const tabs=document.getElementById('tabs');
[...Object.keys(P.models),'real'].forEach(m=>{const b=document.createElement('button');b.textContent=m==='real'?'Real data':(P.nice[m]||m);b.dataset.m=m;b.onclick=()=>show(m);tabs.appendChild(b)});
document.getElementById('shuffle').onclick=()=>show(cur);show(P.models.flow?'flow':Object.keys(P.models)[0]);
if(P.combos){const C=P.combos;counts.set(C.sprite,1296*C.k);const sec=document.getElementById('condsec');
 sec.innerHTML='<h2>Order a monster (conditional flow matching)</h2><p class="sub">Pick all six attributes. The tiles are what the conditional model drew for that exact recipe, with classifier-free guidance. A ✓ means our evaluator classifier read that attribute back as requested.</p><div class="card"><div class="pick" id="pick"></div><div class="big" id="out" style="margin-top:12px"></div></div>';
 const pick=document.getElementById('pick');AN.forEach(a=>{const l=document.createElement('label');l.textContent=a;const s=document.createElement('select');
  A[a].forEach((v,i)=>{const o=document.createElement('option');o.value=i;o.textContent=v;s.appendChild(o)});s.onchange=order;l.appendChild(s);pick.appendChild(l)});
 const rb=document.createElement('button');rb.textContent='Random recipe';rb.onclick=()=>{pick.querySelectorAll('select').forEach(s=>s.selectedIndex=Math.floor(Math.random()*s.options.length));order()};pick.appendChild(rb);
 function order(){const v=[...pick.querySelectorAll('select')].map(s=>+s.value);let id=0;AN.forEach((a,j)=>id=id*A[a].length+v[j]);
  const out=document.getElementById('out');out.innerHTML='';for(let k=0;k<C.k;k++){const i=id*C.k+k;const p=C.pred[i];
   const w=document.createElement('div');w.appendChild(tile(C.sprite,i,lab(p),false));
   const ok=AN.map((a,j)=>(p[j]===v[j]?'✓':'✗')+' '+A[a][p[j]]).join('<br>');const t=document.createElement('div');t.style.fontSize='12px';t.innerHTML=ok;w.appendChild(t);out.appendChild(w)}}
 order()}
const FD=document.getElementById('figs');const cap={table:'Comparison table',ql:'Quality vs latency',train:'Training progress at equal fractions of each model budget cap',traj:'Noise to monster: DDPM (DDIM) vs flow (Euler)',stab:'Quality over training',mem:'Memorization check: the most train-like samples next to their nearest training image',scale:'Doubling the training budget',cond:'Conditional flow: change one attribute at a time (rows: shape, color, eyes, horns, mouth, pattern)'};
Object.entries(F).forEach(([k,src])=>{const c=document.createElement('div');c.className='card';c.innerHTML=`<b>${cap[k]||k}</b><br>`+(src.startsWith('data:video')?`<video class="fig" src="${src}" autoplay loop muted playsinline controls></video>`:`<img class="fig" src="${src}" alt="${cap[k]||k}">`);FD.appendChild(c)});
</script></body></html>
"""

if __name__ == "__main__":
    main()
