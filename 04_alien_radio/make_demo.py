"""Build media/demo.html: a self-contained page (no network) where you pick a target, read the
alien message, edit its symbols, swap partners, and watch the receiver choose.

Senders are exported as their deterministic (argmax) codebooks: 128 objects -> 3 symbols.
Receivers are exported as weights and run in a ~40-line JS forward pass. The page re-checks its
JS forward pass against choices computed in Python (shown in the footer).

  ../.venv/bin/python make_demo.py
"""
from __future__ import annotations

import base64
import json
import os

import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten

from alien import VALUE_NAMES, make_episodes
from train import load_model, sender_table, receiver_choices

HERE = os.path.dirname(os.path.abspath(__file__))
CK = os.environ.get("ALIEN_CKPT", os.path.join(HERE, "checkpoints"))
MEDIA = os.environ.get("ALIEN_MEDIA", os.path.join(HERE, "media"))
RES = os.environ.get("ALIEN_RES", os.path.join(HERE, "results"))
LABEL = {"gumbel": "Gumbel", "reinforce": "REINFORCE", "compositional": "Hand-code-trained"}


def b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def receiver_blob(model, p):
    """Concatenate member p's receiver weights (in,out layout) as float32."""
    parts, shapes = [], {}
    params = dict(tree_flatten(model.receiver.parameters()))
    off = 0
    for name in ["m1", "m2", "c1", "c2"]:
        W = np.array(params[f"{name}.weight"][p], dtype=np.float32)          # (in, out)
        b = np.array(params[f"{name}.bias"][p, 0], dtype=np.float32)          # (out,)
        shapes[name] = {"W": off, "b": off + W.size, "nin": W.shape[0], "nout": W.shape[1]}
        off += W.size + b.size
        parts += [W.ravel(), b]
    return np.concatenate(parts), shapes


def main():
    metrics = json.load(open(os.path.join(RES, "metrics.json")))
    loaded = {}
    for m in ["gumbel", "reinforce", "compositional"]:
        p = os.path.join(CK, f"full_{m}_s0")
        if os.path.exists(p + ".json"):
            loaded[m] = load_model(p)
    world = loaded["gumbel"][1]
    cfg = loaded["gumbel"][2]
    V, L, K = cfg["V"], cfg["L"], cfg["K"]

    senders, receivers, blobs = [], [], []
    off = 0
    shapes = None
    for m, (model, w, c, meta) in loaded.items():
        tab = sender_table(model, world, c)
        for p in range(c["P"]):
            if m != "compositional":
                senders.append({"id": f"{m}{p}", "label": f"{LABEL[m]} seed {p}", "table": tab[p].tolist()})
            elif p == 0:
                senders.append({"id": "handcode", "label": "Hand-designed code (colour, shape, location)",
                                "table": tab[0].tolist()})
            if m == "compositional" and p > 0:
                continue
            blob, shp = receiver_blob(model, p)
            shapes = shp
            receivers.append({"id": f"{m}{p}", "label": f"{LABEL[m]} seed {p}" if m != "compositional"
                              else "Trained on hand-designed code", "offset": off})
            blobs.append(blob)
            off += blob.size
    weights = base64.b64encode(np.concatenate(blobs).astype("<f4").tobytes()).decode()

    # Python reference choices for a JS self-check
    checks = []
    ep_c, ep_t = make_episodes(world, 60, K, "test", seed=4242)
    for m in loaded:
        model, _, c, _ = loaded[m]
        tab = sender_table(model, world, c)
        tgt = ep_c[np.arange(len(ep_c)), ep_t]
        msgs = tab[:, tgt]
        ch = receiver_choices(model.receiver, world, msgs, ep_c, V)
        for i in range(0, len(ep_c), 3):
            checks.append({"r": f"{m}0", "msg": msgs[0, i].tolist(), "cands": ep_c[i].tolist(),
                           "choice": int(ch[0, i])})

    gm = metrics["methods"]
    head = {gm[m]["label"]: {"train": float(np.mean(gm[m]["train_acc"])), "heldout": float(np.mean(gm[m]["heldout_acc"]))}
            for m in gm}
    data = {
        "dims": world.dims, "attrs": world.attr_names, "names": VALUE_NAMES,
        "objects": world.objects.tolist(), "heldout": world.test_ids.tolist(),
        "train": world.train_ids.tolist(), "L": L, "V": V, "K": K,
        "senders": senders, "receivers": receivers, "shapes": shapes, "weights": weights,
        "checks": checks, "head": head, "P": cfg["P"], "demoMember": metrics.get("demo_member", 0),
        "images": {k: b64(os.path.join(MEDIA, f)) for k, f in
                   [("curves", "training_curves.png"), ("swap", "partner_swap.png"),
                    ("dict", "dictionary_gumbel.png"), ("summary", "summary_bars.png")]
                   if os.path.exists(os.path.join(MEDIA, f))},
    }
    html = TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    out = os.path.join(MEDIA, "demo.html")
    with open(out, "w") as f:
        f.write(html)
    print(f"wrote {out} ({len(html) / 1e6:.2f} MB), {len(senders)} senders, {len(receivers)} receivers")


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Alien Radio</title>
<style>
:root{--bg:#f6f5f1;--panel:#ffffff;--ink:#1c1c1a;--muted:#6b6a64;--line:#dddbd3;--accent:#2a6fdb;--good:#2e9e5b;--bad:#d9534f;--tile:#14171c;--glyph:#7cf2c1;--chip:#efeee9}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#121316;--panel:#1b1d21;--ink:#ecebe6;--muted:#9a9890;--line:#2e3036;--accent:#6aa0ff;--good:#4cc184;--bad:#ef6f6b;--tile:#07090c;--glyph:#7cf2c1;--chip:#24262b}}
:root[data-theme="dark"]{--bg:#121316;--panel:#1b1d21;--ink:#ecebe6;--muted:#9a9890;--line:#2e3036;--accent:#6aa0ff;--good:#4cc184;--bad:#ef6f6b;--tile:#07090c;--glyph:#7cf2c1;--chip:#24262b}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
main{max-width:1080px;margin:0 auto;padding:24px 16px 60px}
h1{font-size:30px;margin:0 0 4px;letter-spacing:-.5px}
h2{font-size:18px;margin:0 0 10px}
.sub{color:var(--muted);margin:0 0 18px;max-width:760px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px;margin:14px 0}
.row{display:flex;flex-wrap:wrap;gap:10px;align-items:center}
button,select{font:inherit;border:1px solid var(--line);background:var(--chip);color:var(--ink);border-radius:8px;padding:6px 12px;cursor:pointer}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff}
.chip{padding:4px 10px;border-radius:999px;font-size:13px}
.chip.on{outline:2px solid var(--accent)}
.stage{display:grid;grid-template-columns:200px 1fr;gap:18px;align-items:start}
@media (max-width:760px){.stage{grid-template-columns:1fr}}
.label{font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-bottom:6px}
.msg{display:flex;gap:8px}
.glyph{width:64px;height:72px;background:var(--tile);border-radius:10px;display:flex;flex-direction:column;align-items:center;justify-content:center;cursor:pointer;border:2px solid transparent;position:relative}
.glyph.edited{border-color:#f5a623}
.glyph small{color:#8a96a8;font-size:10px}
.picker{display:flex;gap:4px;margin-top:8px;flex-wrap:wrap}
.picker .glyph{width:38px;height:42px}
.cands{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px}
@media (max-width:520px){.cands{grid-template-columns:repeat(3,minmax(0,1fr))}}
.cand{border:2px solid var(--line);border-radius:10px;padding:6px;text-align:center;position:relative}
.cand.pick{border-color:var(--accent)}
.cand.target{box-shadow:0 0 0 3px rgba(245,166,35,.55)}
.bar{height:6px;background:var(--line);border-radius:3px;overflow:hidden;margin-top:4px}
.bar>i{display:block;height:100%;background:var(--accent)}
.cap{font-size:11px;color:var(--muted);line-height:1.25}
.badge{display:inline-block;padding:2px 10px;border-radius:999px;font-weight:600;font-size:13px}
.badge.ok{background:var(--good);color:#fff}.badge.no{background:var(--bad);color:#fff}
.novel{position:absolute;top:2px;right:4px;font-size:10px;color:#f5a623;font-weight:700}
table{border-collapse:collapse;width:100%;font-size:13px}
td,th{border-bottom:1px solid var(--line);padding:4px 6px;text-align:left}
.dictgrid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}
@media (max-width:760px){.dictgrid{grid-template-columns:1fr}}
img.fig{width:100%;height:auto;border-radius:8px;background:#fff}
.stat{font-size:26px;font-weight:700}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.mono{font-family:ui-monospace,Menlo,monospace}
</style></head>
<body><main>
<h1>Alien Radio</h1>
<p class="sub">Two neural agents invented this language from nothing. The <b>sender</b> sees one object and may
transmit only <b>3 symbols from an alphabet of 8</b>. The <b>receiver</b> sees five candidates in shuffled order and
must point at the right one. Nothing else passes between them. 20% of colour × shape × location combinations
(marked <span style="color:#f5a623;font-weight:700">NEW</span>) were never seen in training.</p>

<div class="panel">
  <div class="row" style="margin-bottom:10px">
    <button class="primary" id="btnSeen">New round (seen objects)</button>
    <button id="btnNew">New round (never-seen combinations)</button>
  </div>
  <div class="row">
    <label>Sender <select id="selS"></select></label>
    <label>Receiver <select id="selR"></select></label>
    <span class="cap" id="pairNote"></span>
  </div>
</div>

<div class="panel">
  <div class="label">Or build the target yourself</div>
  <div class="row" id="builder"></div>
</div>

<div class="panel">
  <div class="stage">
    <div>
      <div class="label">Sender sees</div>
      <div id="targetView"></div>
      <div class="cap" id="targetName"></div>
    </div>
    <div>
      <div class="label">Message on the channel <span class="cap">(click a symbol to change it: an intervention)</span></div>
      <div class="msg" id="msg"></div>
      <div id="picker"></div>
      <div class="row" style="margin-top:8px"><button id="btnReset">Restore sender's message</button><span class="cap" id="reading"></span></div>
    </div>
  </div>
  <div class="label" style="margin-top:16px">Receiver sees (shuffled) and chooses <span id="verdict"></span></div>
  <div class="cands" id="cands"></div>
</div>

<div class="panel">
  <h2>This pair, measured live in your browser</h2>
  <div class="stats" id="live"></div>
  <p class="cap">400 random rounds each. Swap the receiver for one from another seed: the senders' languages were
  invented independently, so a stranger mostly can't understand them.</p>
</div>

<div class="panel">
  <h2>The receiver's dictionary</h2>
  <p class="cap">Built by intervention: for every object, overwrite one message position with each of the 8 symbols and
  record what the receiver picks among all 128 objects. Each position is labelled with the attribute its edits change
  most. Each symbol is labelled with the one attribute value the receiver most reliably reads from it in that slot,
  whatever the other two symbols say (100% = a perfectly context-free word).</p>
  <div class="dictgrid" id="dict"></div>
</div>

<div class="panel" id="figs"></div>
<p class="cap" id="selfcheck"></p>
</main>
<script>
const D = __DATA__;
const COLORS = {red:"#e5484d",orange:"#f76b15",yellow:"#f2c200",green:"#30a46c",cyan:"#05a2c2",blue:"#3e63dd",purple:"#8e4ec6",pink:"#e93d82"};
const GLYPHS = [
 '<path d="M20 5V35M9 13H31"/>',
 '<circle cx="20" cy="20" r="12"/><circle cx="20" cy="20" r="2.5" fill="currentColor"/>',
 '<path d="M7 33L20 7L33 33"/>',
 '<path d="M8 9H32L8 31H32"/>',
 '<path d="M9 32Q20 -4 31 32M13 22H27"/>',
 '<path d="M20 5L35 20L20 35L5 20Z"/>',
 '<path d="M20 35V12M9 8V17Q20 27 31 17V8"/>',
 '<path d="M20 20m-4 0a4 4 0 1 1 8 0a8 8 0 1 1-16 0a12 12 0 1 1 24 0"/>'
];
const glyphSVG = (v, s=34) => `<svg width="${s}" height="${s}" viewBox="0 0 40 40" fill="none" stroke="var(--glyph)" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" style="color:var(--glyph)">${GLYPHS[v % GLYPHS.length]}</svg>`;
const heldSet = new Set(D.heldout);
const objIndex = new Map(D.objects.map((o, i) => [o.join(","), i]));
const NAMES = D.attrs.map(a => D.names[a]);
const describe = i => D.objects[i].map((v, a) => NAMES[a][v]).join(" ");
function objSVG(i, s=64){
  const [c, sh, loc] = D.objects[i];
  const col = COLORS[D.names.color[c]];
  const cx = (loc % 2 ? 0.75 : 0.25) * s, cy = (loc >= 2 ? 0.75 : 0.25) * s, r = s * 0.17;
  let shape = "";
  if (sh === 0) shape = `<circle cx="${cx}" cy="${cy}" r="${r}" fill="${col}"/>`;
  else if (sh === 1) shape = `<rect x="${cx-r}" y="${cy-r}" width="${2*r}" height="${2*r}" fill="${col}"/>`;
  else if (sh === 2) shape = `<path d="M${cx} ${cy-r*1.1}L${cx+r*1.1} ${cy+r*0.9}L${cx-r*1.1} ${cy+r*0.9}Z" fill="${col}"/>`;
  else { let p = ""; for (let k = 0; k < 10; k++){ const a = -Math.PI/2 + k*Math.PI/5, rr = k%2 ? r*0.45 : r*1.15; p += (k?"L":"M") + (cx+rr*Math.cos(a)).toFixed(1) + " " + (cy+rr*Math.sin(a)).toFixed(1);} shape = `<path d="${p}Z" fill="${col}"/>`; }
  return `<svg width="${s}" height="${s}" viewBox="0 0 ${s} ${s}"><rect x="1" y="1" width="${s-2}" height="${s-2}" rx="6" fill="none" stroke="var(--line)"/><path d="M${s/2} 4V${s-4}M4 ${s/2}H${s-4}" stroke="var(--line)" stroke-dasharray="2 3"/>${shape}</svg>`;
}

// ---------------------------------------------------------------- receiver forward pass
const raw = Uint8Array.from(atob(D.weights), c => c.charCodeAt(0));
const W = new Float32Array(raw.buffer);
const FEAT = D.dims.reduce((a, b) => a + b, 0);
function feat(i){ const f = new Float32Array(FEAT); let off = 0; D.objects[i].forEach((v, a) => { f[off + v] = 1; off += D.dims[a]; }); return f; }
const FEATS = D.objects.map((_, i) => feat(i));
function linear(rec, name, x, relu){
  const s = D.shapes[name], o = rec.offset, nout = s.nout, y = new Float32Array(nout);
  for (let j = 0; j < nout; j++) y[j] = W[o + s.b + j];
  for (let i = 0; i < s.nin; i++){ const xi = x[i]; if (xi === 0) continue; const base = o + s.W + i * nout; for (let j = 0; j < nout; j++) y[j] += xi * W[base + j]; }
  if (relu) for (let j = 0; j < nout; j++) if (y[j] < 0) y[j] = 0;
  return y;
}
const cache = new Map();
function candEmb(rec, i){ const k = rec.id + ":" + i; if (!cache.has(k)) cache.set(k, linear(rec, "c2", linear(rec, "c1", FEATS[i], true), false)); return cache.get(k); }
function msgEmb(rec, msg){ const k = rec.id + ":m" + msg.join(""); if (!cache.has(k)){ const x = new Float32Array(D.L * D.V); msg.forEach((v, p) => x[p * D.V + v] = 1); cache.set(k, linear(rec, "m2", linear(rec, "m1", x, true), false)); } return cache.get(k); }
function scores(rec, msg, cands){ const m = msgEmb(rec, msg); return cands.map(i => { const c = candEmb(rec, i); let s = 0; for (let j = 0; j < m.length; j++) s += m[j] * c[j]; return s; }); }
const softmax = s => { const mx = Math.max(...s), e = s.map(x => Math.exp(x - mx)), z = e.reduce((a, b) => a + b, 0); return e.map(x => x / z); };
const argmax = a => a.indexOf(Math.max(...a));

// ---------------------------------------------------------------- episodes (same recipe as training)
const ALL = D.objects.map((_, i) => i);
function episode(pool, forcedTarget){
  const set = new Set(pool);
  for (;;){
    let ids;
    if (forcedTarget !== undefined || Math.random() < 0.5){
      const c = D.objects[forcedTarget !== undefined ? forcedTarget : pool[Math.floor(Math.random() * pool.length)]];
      const s = new Set(forcedTarget !== undefined ? [forcedTarget] : []);
      for (let t = 0; t < 300 && s.size < D.K; t++){
        const o = c.map((v, a) => Math.random() < 0.5 ? v : Math.floor(Math.random() * D.dims[a]));
        const i = objIndex.get(o.join(","));
        if (set.has(i)) s.add(i);
      }
      if (s.size < D.K) continue;
      ids = [...s];
    } else { ids = [...pool].sort(() => Math.random() - 0.5).slice(0, D.K); }
    const target = forcedTarget !== undefined ? forcedTarget : ids[Math.floor(Math.random() * ids.length)];
    ids.sort(() => Math.random() - 0.5);
    return {cands: ids, target};
  }
}

// ---------------------------------------------------------------- UI state
const S = {sender: null, receiver: null, ep: null, msg: null, edit: -1};
const selS = document.getElementById("selS"), selR = document.getElementById("selR");
D.senders.forEach(s => selS.add(new Option(s.label, s.id)));
D.receivers.forEach(r => selR.add(new Option(r.label, r.id)));
selS.value = "gumbel" + D.demoMember; selR.value = "gumbel" + D.demoMember;
const sender = () => D.senders.find(s => s.id === selS.value);
const receiver = () => D.receivers.find(r => r.id === selR.value);

function newRound(pool, forced){ S.ep = episode(pool, forced); S.msg = sender().table[S.ep.target].slice(); S.edit = -1; render(); }
function render(){
  const {cands, target} = S.ep, rec = receiver(), orig = sender().table[target];
  document.getElementById("targetView").innerHTML = objSVG(target, 150);
  document.getElementById("targetName").innerHTML = describe(target) + (heldSet.has(target) ? ' · <b style="color:#f5a623">never seen in training</b>' : "");
  document.getElementById("msg").innerHTML = S.msg.map((v, p) => `<div class="glyph ${v !== orig[p] ? "edited" : ""}" data-p="${p}">${glyphSVG(v)}<small>pos ${p} · s${v}</small></div>`).join("");
  document.querySelectorAll("#msg .glyph").forEach(g => g.onclick = () => { S.edit = S.edit === +g.dataset.p ? -1 : +g.dataset.p; render(); });
  const pk = document.getElementById("picker");
  pk.innerHTML = S.edit < 0 ? "" : `<div class="cap">Replace position ${S.edit} with:</div><div class="picker">` + [...Array(D.V).keys()].map(v => `<div class="glyph" data-v="${v}">${glyphSVG(v, 22)}<small>s${v}</small></div>`).join("") + "</div>";
  pk.querySelectorAll(".glyph").forEach(g => g.onclick = () => { S.msg[S.edit] = +g.dataset.v; render(); });
  const pr = softmax(scores(rec, S.msg, cands)), pick = argmax(pr);
  document.getElementById("cands").innerHTML = cands.map((c, k) => `<div class="cand ${k === pick ? "pick" : ""} ${c === target ? "target" : ""}">${heldSet.has(c) ? '<span class="novel">NEW</span>' : ""}${objSVG(c, 72)}<div class="cap">${describe(c)}</div><div class="bar"><i style="width:${(pr[k] * 100).toFixed(1)}%"></i></div><div class="cap">${(pr[k] * 100).toFixed(1)}%${k === pick ? " ← pick" : ""}</div></div>`).join("");
  const ok = cands[pick] === target;
  document.getElementById("verdict").innerHTML = ok ? '<span class="badge ok">correct</span>' : '<span class="badge no">wrong</span>';
  const all = softmax(scores(rec, S.msg, ALL)), best = argmax(all);
  document.getElementById("reading").innerHTML = `Out of all 128 objects, the receiver reads this message as <b>${describe(best)}</b> (${(all[best] * 100).toFixed(0)}%)`;
}

function liveStats(){
  const snd = sender(), rec = receiver();
  const acc = pool => { let k = 0; for (let n = 0; n < 400; n++){ const e = episode(pool); const sc = scores(rec, snd.table[e.target], e.cands); k += e.cands[argmax(sc)] === e.target; } return k / 400; };
  const a1 = acc(D.train), a2 = acc(D.heldout);
  const same = snd.id === rec.id || (snd.id === "handcode" && rec.id.startsWith("compositional"));
  document.getElementById("pairNote").textContent = same ? "original partners" : "swapped partners: they never trained together";
  document.getElementById("live").innerHTML = [["seen combinations", a1], ["never-seen combinations", a2], ["chance", 1 / D.K]].map(([n, v]) => `<div><div class="stat">${(v * 100).toFixed(0)}%</div><div class="cap">${n}</div></div>`).join("");
}

function dictionary(){
  const snd = sender(), rec = receiver(), A = D.dims.length;
  const allEmb = ALL.map(i => candEmb(rec, i));
  const readMsg = msg => { const m = msgEmb(rec, msg); let b = -1, bs = -Infinity; for (let i = 0; i < allEmb.length; i++){ let s = 0; const c = allEmb[i]; for (let j = 0; j < m.length; j++) s += m[j] * c[j]; if (s > bs){ bs = s; b = i; } } return b; };
  let html = "";
  for (let p = 0; p < D.L; p++){
    const change = new Array(A).fill(0); let n = 0;
    const tally = [...Array(D.V)].map(() => [...Array(A)].map((_, a) => new Array(D.dims[a]).fill(0)));
    for (const o of ALL){
      const base = snd.table[o], b0 = readMsg(base);
      for (let v = 0; v < D.V; v++){
        const m = base.slice(); m[p] = v; const c = v === base[p] ? b0 : readMsg(m);
        D.objects[c].forEach((val, a) => tally[v][a][val]++);
        if (v !== base[p]){ n++; D.objects[c].forEach((val, a) => change[a] += val !== D.objects[b0][a]); }
      }
    }
    const a = argmax(change);
    html += `<div><div class="label">position ${p} → mostly ${D.attrs[a]}</div><div class="cap">edits change: ${D.attrs.map((nm, k) => nm + " " + Math.round(100 * change[k] / n) + "%").join(", ")}</div><table><tr><th>symbol</th><th>receiver reads</th><th>how reliably</th></tr>`;
    for (let v = 0; v < D.V; v++){
      // the attribute value this symbol most reliably evokes, scored against chance (1 / #values)
      let bestA = 0, bestV = 0, bestC = 0, bestS = -1;
      for (let b = 0; b < A; b++){ const t = tally[v][b], tot = t.reduce((x, y) => x + y, 0), j = argmax(t), c = t[j] / tot; if (c - 1 / D.dims[b] > bestS){ bestS = c - 1 / D.dims[b]; bestA = b; bestV = j; bestC = c; } }
      const val = NAMES[bestA][bestV];
      const sw = D.attrs[bestA] === "color" ? `<span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:${COLORS[val]};margin-right:4px"></span>` : "";
      html += `<tr><td>${glyphSVG(v, 18)} s${v}</td><td>${sw}${D.attrs[bestA]}: ${val}</td><td>${Math.round(100 * bestC)}% <span class="cap">(chance ${Math.round(100 / D.dims[bestA])}%)</span></td></tr>`;
    }
    html += "</table></div>";
  }
  document.getElementById("dict").innerHTML = html;
}

// builder chips
const B = [0, 0, 0];
function builder(){
  document.getElementById("builder").innerHTML = D.attrs.map((a, k) => `<div class="row" style="gap:4px">` + NAMES[k].map((v, j) => `<button class="chip ${B[k] === j ? "on" : ""}" data-a="${k}" data-v="${j}">${a === "color" ? `<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${COLORS[v]};margin-right:4px"></span>` : ""}${v}</button>`).join("") + "</div>").join("");
  document.querySelectorAll("#builder .chip").forEach(c => c.onclick = () => { B[+c.dataset.a] = +c.dataset.v; builder(); const t = objIndex.get(B.join(",")); newRound(heldSet.has(t) ? D.heldout : D.train, t); });
}

function refreshPair(){ liveStats(); dictionary(); if (S.ep){ S.msg = sender().table[S.ep.target].slice(); S.edit = -1; render(); } }
document.getElementById("btnSeen").onclick = () => newRound(D.train);
document.getElementById("btnNew").onclick = () => newRound(D.heldout);
document.getElementById("btnReset").onclick = () => { S.msg = sender().table[S.ep.target].slice(); S.edit = -1; render(); };
selS.onchange = refreshPair; selR.onchange = refreshPair;

// figures + headline numbers from the Python evaluation
const H = D.head, pct = x => (x * 100).toFixed(1) + "%";
let figs = `<h2>Offline evaluation (Python, 20,000 rounds per split, mean over ${D.P} seeds)</h2><table><tr><th>method</th><th>seen</th><th>never-seen</th></tr>` +
  Object.entries(H).map(([m, v]) => `<tr><td>${m}</td><td>${pct(v.train)}</td><td>${pct(v.heldout)}</td></tr>`).join("") + "</table>";
for (const [k, cap] of [["summary", "Accuracy by method"], ["curves", "Training curves"], ["dict", "Dictionary figure (sender side + interventions)"], ["swap", "Partner swapping"]])
  if (D.images[k]) figs += `<div class="label" style="margin-top:14px">${cap}</div><img class="fig" alt="${cap}" src="data:image/png;base64,${D.images[k]}">`;
document.getElementById("figs").innerHTML = figs;

// self-check: JS receiver forward pass vs Python choices
let match = 0;
for (const c of D.checks){ const rec = D.receivers.find(r => r.id === c.r); match += argmax(scores(rec, c.msg, c.cands)) === c.choice; }
document.getElementById("selfcheck").textContent = `Self-check: JS forward pass reproduces the Python receiver's choice on ${match}/${D.checks.length} reference rounds. Senders are shown as their deterministic codebooks (argmax symbol per position).`;
window.__selfcheck = {match, total: D.checks.length};

builder(); refreshPair(); newRound(D.train);
</script>
</body></html>
"""

if __name__ == "__main__":
    main()
