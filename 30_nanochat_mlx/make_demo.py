"""
Build media/demo.html (self-contained, no network): loss curves, generations at several
pretraining checkpoints, benchmark tables, sandbox results, and step-by-step replays of the
SFT model playing held-out Grid Courier episodes. Also writes media/episode_transcripts.md.

    ../.venv/bin/python make_demo.py --ckpt checkpoints/sft/last
"""
import argparse
import base64
import glob
import html
import json
import os

import mlx.core as mx

import checkpoint as ckpt
from eval_sandbox import LMPolicy
from sandbox import TEST3_SEED0, TEST_SEED0, TESTSEEN_SEED0, make_env, parse_action
from tokenizer import ChatTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))


def snapshot(env):
    return {"agent": list(env.agent), "holding": env.holding,
            "items": {k: (list(v) if isinstance(v, tuple) else v) for k, v in env.item_loc.items()},
            "conts": {k: list(v) for k, v in env.cont_loc.items()}}


def record(policy, env):
    frames = [{"state": snapshot(env), "obs": env.observation(), "action": None, "valid": None}]
    while not env.over():
        raw = policy(env)
        act = parse_action(raw)
        if act is None:
            env.steps += 1
            ok = False
        else:
            ok = env.step(act)
        frames.append({"state": snapshot(env), "obs": env.observation(), "action": raw, "valid": ok})
    return {"task": env.task_text(), "success": env.success(), "frames": frames}


def b64(path):
    p = os.path.join(HERE, path)
    return base64.b64encode(open(p, "rb").read()).decode() if os.path.exists(p) else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/sft/last")
    ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args()
    model, _ = ckpt.load_model(a.ckpt)
    tok = ChatTokenizer.load()
    pol = LMPolicy(model, tok)
    episodes = []
    for split, s0 in [("test", TEST_SEED0), ("test3", TEST3_SEED0), ("test_seen", TESTSEEN_SEED0)]:
        for i in range(a.n):
            ep = record(pol, make_env(s0 + i, split))
            ep["split"] = split
            ep["seed"] = s0 + i
            episodes.append(ep)
    # transcripts (markdown)
    with open(os.path.join(HERE, "media", "episode_transcripts.md"), "w") as f:
        f.write(f"# Grid Courier episodes played by `{a.ckpt}` (greedy decoding, KV cache)\n\n")
        for ep in episodes:
            f.write(f"## {ep['split']} seed {ep['seed']}: {'SUCCESS' if ep['success'] else 'FAIL'} in {len(ep['frames'])-1} actions\n\n```\n")
            f.write(ep["frames"][0]["obs"] + "\n")
            for fr in ep["frames"][1:]:
                f.write(f"> {fr['action']}" + ("" if fr["valid"] else "    [INVALID]") + "\n")
            f.write("```\n\n")
    samples = []
    for d in sorted(glob.glob(os.path.join(HERE, "checkpoints/pilot/step_*"))) + [os.path.join(HERE, "checkpoints/pilot/last")]:
        p = os.path.join(d, "samples.json")
        if os.path.exists(p):
            m = ckpt.load_meta(d)
            samples.append({"step": m["step"], "tokens": m["step"] * m["args"]["total_batch"], "samples": json.load(open(p))})
    data = {k: (json.load(open(os.path.join(HERE, p))) if os.path.exists(os.path.join(HERE, p)) else None)
            for k, p in [("bench", "media/bench.json"), ("infer", "media/inference_bench.json"), ("sandbox", "media/sandbox_eval.json")]}
    extra = os.path.join(HERE, "media/sandbox_eval_extra.json")
    if os.path.exists(extra) and data["sandbox"] is not None:
        for k, v in json.load(open(extra)).items():
            data["sandbox"].setdefault(k, {}).update(v)
    page = TEMPLATE.replace("__DATA__", json.dumps({"episodes": episodes, "samples": samples, **data})) \
        .replace("__LOSS__", b64("media/loss_curve.png")).replace("__SFT__", b64("media/sft_curve.png"))
    open(os.path.join(HERE, "media", "demo.html"), "w").write(page)
    print("wrote media/demo.html and media/episode_transcripts.md;",
          sum(e["success"] for e in episodes), "/", len(episodes), "episodes solved")


TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Nanochat on MLX</title>
<style>
:root{--bg:#fbfaf7;--fg:#1d1d1f;--mut:#6b6b70;--card:#fff;--line:#e4e2dc;--acc:#1f5fa8;--bad:#c0392b;--good:#2a8a4a}
@media (prefers-color-scheme:dark){:root{--bg:#141416;--fg:#ececef;--mut:#9a9aa2;--card:#1d1d21;--line:#303036;--acc:#6ea8ff;--bad:#ff7b6b;--good:#5fd38a}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,system-ui,sans-serif;margin:0;padding:24px 16px}
main{max-width:1040px;margin:0 auto}h1{font-size:26px;margin:0 0 4px}h2{font-size:18px;margin:32px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px}
.mut{color:var(--mut)}img{max-width:100%;background:#fff;border-radius:6px}
table{border-collapse:collapse;font-size:13px;margin:8px 0;width:100%;overflow-x:auto;display:block}td,th{border-bottom:1px solid var(--line);padding:4px 8px;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;margin:8px 0}
.sample b{color:var(--acc)}pre{white-space:pre-wrap;font:12.5px/1.45 ui-monospace,Menlo,monospace;margin:0}
.ep{display:flex;gap:16px;flex-wrap:wrap}.grid{display:grid;grid-template-columns:repeat(5,52px);grid-template-rows:repeat(5,52px);gap:3px}
.cell{background:var(--bg);border:1px solid var(--line);border-radius:4px;font-size:10px;position:relative;display:flex;flex-direction:column;align-items:center;justify-content:center;line-height:1.1}
.agent{outline:3px solid var(--acc)}.cont{font-weight:700}.ctrl button{margin-right:6px;padding:4px 10px;border:1px solid var(--line);border-radius:5px;background:var(--card);color:var(--fg);cursor:pointer}
.bad{color:var(--bad)}.good{color:var(--good)}select{padding:4px;background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:5px;max-width:100%}
</style></head><body><main>
<h1>Nanochat on MLX: a pilot</h1>
<div class="mut">A pure-MLX port of karpathy/nanochat (tokenizer, GPT, Muon, pretraining, SFT, KV-cache inference), a pilot pretraining run on one shared M5 Max, then a tiny job: deliver items in a 5x5 text grid world.</div>
<h2>1. Pretraining loss</h2><img src="data:image/png;base64,__LOSS__" alt="loss curve">
<h2>2. What the model says along the way</h2><div class="mut">Greedy continuations of fixed prompts at pretraining checkpoints.</div><div id="samples"></div>
<h2>3. Grid Courier episodes (SFT model, greedy decoding)</h2>
<div class="mut">"test" episodes use delivery pairs never seen in training; "test3" episodes have three deliveries (training tasks have at most two). Blue outline = agent; bold = container.</div>
<div class="ctrl" style="margin:8px 0"><select id="epsel"></select> <button id="prev">&#9664; prev</button><button id="next">next &#9654;</button><button id="play">play</button> <span id="stepinfo" class="mut"></span></div>
<div class="ep"><div class="grid" id="grid"></div><div class="card" style="flex:1;min-width:260px"><pre id="obs"></pre><div id="act" style="margin-top:8px"></div></div></div>
<h2>4. Sandbox results</h2><div class="mut">Review caveat: the held-out action <code>put I in C</code> is a verbatim copy of words in the task line, and a from-scratch LM also scores 100%. The MLP's flat 28-way head never saw the 5 held-out <code>put</code> classes as training targets, so its 0% is by construction.</div><div id="sandbox"></div><img src="data:image/png;base64,__SFT__" alt="sft curve" style="max-width:560px">
<h2>5. Throughput</h2><div class="mut">Review caveat: measured on a heavily shared GPU. The MFU column divides by a contended 12.3 TFLOP/s matmul "peak". On an idle GPU the bf16 matmul peak measured 60.3 TFLOP/s, d6 trained at ~214k tok/s and decoded at ~1.3k tok/s with a 0.9&ndash;1.6 ms first token. See RESULTS.md &sect;2 and &sect;6.</div><div id="bench"></div><div id="infer"></div>
</main><script>
const D=__DATA__;
const esc=s=>String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
document.getElementById('samples').innerHTML=D.samples.map(s=>`<div class="card sample"><div class="mut">step ${s.step} &middot; ${(s.tokens/1e6).toFixed(0)}M tokens</div>`+Object.entries(s.samples).map(([p,c])=>`<div><b>${esc(p)}</b>${esc(c)}</div>`).join('')+'</div>').join('');
const sel=document.getElementById('epsel');D.episodes.forEach((e,i)=>{const o=document.createElement('option');o.value=i;o.textContent=`${e.split} #${e.seed%1000}: ${e.success?'success':'fail'} (${e.frames.length-1} actions) - ${e.task}`;sel.appendChild(o)});
let ep=0,fr=0,timer=null;
function draw(){const e=D.episodes[ep],f=e.frames[fr],s=f.state;const g=document.getElementById('grid');g.innerHTML='';
for(let y=4;y>=0;y--)for(let x=0;x<5;x++){const c=document.createElement('div');c.className='cell';let t=[];
for(const [k,v] of Object.entries(s.conts))if(v[0]==x&&v[1]==y){c.classList.add('cont');const inside=Object.entries(s.items).filter(([i,l])=>l===k).map(([i])=>i);t.push(k+(inside.length?'['+inside.join(',')+']':''))}
for(const [k,v] of Object.entries(s.items))if(Array.isArray(v)&&v[0]==x&&v[1]==y)t.push(k);
if(s.agent[0]==x&&s.agent[1]==y){c.classList.add('agent');t.push(s.holding?'you+'+s.holding:'you')}
c.innerHTML=t.map(esc).join('<br>')||`<span class="mut">${x},${y}</span>`;g.appendChild(c)}
document.getElementById('obs').textContent=f.obs;
document.getElementById('act').innerHTML=f.action===null?'<span class="mut">start</span>':`action: <b>${esc(f.action)}</b> `+(f.valid?'<span class="good">valid</span>':'<span class="bad">INVALID</span>');
document.getElementById('stepinfo').textContent=`step ${fr}/${e.frames.length-1}`+(fr==e.frames.length-1?(e.success?' - task done':' - out of steps'):'')}
sel.onchange=()=>{ep=+sel.value;fr=0;draw()};document.getElementById('next').onclick=()=>{fr=Math.min(fr+1,D.episodes[ep].frames.length-1);draw()};
document.getElementById('prev').onclick=()=>{fr=Math.max(fr-1,0);draw()};
document.getElementById('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;return}fr=0;timer=setInterval(()=>{if(fr>=D.episodes[ep].frames.length-1){clearInterval(timer);timer=null}else{fr++;draw()}},450);draw()};
draw();
function table(rows,cols){return '<table><tr>'+cols.map(c=>`<th>${c[0]}</th>`).join('')+'</tr>'+rows.map(r=>'<tr>'+cols.map(c=>`<td>${c[1](r)}</td>`).join('')+'</tr>').join('')+'</table>'}
const pct=x=>x==null?'-':(100*x).toFixed(1)+'%';
if(D.sandbox){const rows=[];for(const [k,v] of Object.entries(D.sandbox))rows.push({k,...v});
document.getElementById('sandbox').innerHTML=table(rows,[['policy',r=>r.k],['success (seen pairs)',r=>pct(r.test_seen&&r.test_seen.success_rate)],['success (held-out pairs)',r=>pct(r.test&&r.test.success_rate)],['success (3 deliveries)',r=>pct(r.test3&&r.test3.success_rate)],['validity (held-out)',r=>pct(r.test&&r.test.action_validity)],['validity (3 deliveries)',r=>pct(r.test3&&r.test3.action_validity)],['ms/decision',r=>{const s=r.test||r.test3;return s?s.decision_latency_ms_median.toFixed(1):'-'}]])}
if(D.bench)document.getElementById('bench').innerHTML=`<div class="mut">Measured matmul peak: bf16 ${D.bench.matmul_peak_tflops.bf16.toFixed(1)} TFLOP/s, fp32 ${D.bench.matmul_peak_tflops.fp32.toFixed(1)} TFLOP/s (shared GPU)</div>`+table(D.bench.rows,[['depth',r=>'d'+r.depth],['params',r=>r.params_M.toFixed(1)+'M'],['dtype',r=>r.dtype],['compile',r=>r.compile?'yes':'no'],['B x T',r=>r.B+'x'+r.T],['tok/s',r=>Math.round(r.tok_per_s).toLocaleString()],['TFLOP/s',r=>r.tflops.toFixed(2)],['MFU vs measured peak',r=>pct(r.mfu_vs_measured_peak)],['peak mem',r=>r.peak_mem_gb.toFixed(1)+' GB']]);
if(D.infer)document.getElementById('infer').innerHTML='<div class="mut" style="margin-top:12px">Inference, batch 1, pilot model</div>'+table(D.infer,[['prompt tokens',r=>r.prompt_len],['first-token latency',r=>r.first_token_ms.toFixed(1)+' ms'],['decode tok/s (KV cache)',r=>r.decode_tok_per_s.toFixed(0)],['no-cache tok/s',r=>r.naive_no_cache_tok_per_s.toFixed(0)]]);
</script></body></html>"""

if __name__ == "__main__":
    main()
