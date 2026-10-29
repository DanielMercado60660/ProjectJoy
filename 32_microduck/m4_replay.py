"""M4: build media/demo.html, a self-contained replay of recorded episodes (no network needed).

Per episode: the head camera (every 2nd tick, JPEG), a top-down view drawn from the room layout with
the duck's true path (replay only), the bot's remembered objects (from its odometry frame, placed in
the room through its start pose), and per tick: the goal, the bot's activity and skill, the executed
and proposed commands, the skill's expected vs observed outcome, and memory contents.

    python m4_replay.py <episode.npz> [<episode.npz> ...]
"""
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
SPLITS = {"train": ("train.json", "configs")}


def load_cfg(cid):
    for f, part in (("train.json", "configs"), ("instances.json", "dev"), ("arrangements.json", "dev"),
                    ("appearances.json", "dev"), ("distractor.json", "dev"), ("instances.json", "test"),
                    ("arrangements.json", "test"), ("appearances.json", "test"), ("distractor.json", "test")):
        d = json.loads((HERE / "splits" / f).read_text())
        for c in d[part]:
            if c["id"] == cid:
                return c
    raise KeyError(cid)


def jpeg(a):
    b = io.BytesIO()
    Image.fromarray(a).resize((192, 192), Image.Resampling.NEAREST).save(b, "JPEG", quality=72)
    return base64.b64encode(b.getvalue()).decode()


def episode_payload(path, every=2):
    z = np.load(path)
    H = json.loads(str(z["header"]))
    cfg = load_cfg(H["config_id"])
    from tasks import make_episode
    ep = make_episode(cfg, H["task"], H["k"])
    start = ep["start"] or cfg["layout"]["start"]
    lay, app = cfg["layout"], cfg["appearance"]
    to255 = lambda c: [int(255 * v) for v in c]  # noqa: E731
    mem = json.loads(str(z["memory"]))
    T = len(z["frames"])
    idx = list(range(0, T, every)) + ([T - 1] if (T - 1) % every else [])
    # the skill's report, rebuilt from the header's skill log: a skill commands ticks start..end-1 and
    # reports at tick end (the stored `observed` arrays of the M4 recordings are one tick off at every
    # skill boundary, REVIEW_M4.md); shown with the tick it was reported at, since it is a hindsight label
    observed = ["-"] * T
    for e in H["skill_log"]:
        for t in range(max(0, e["start"]), min(T, e["end"])):
            if observed[t] == "-":
                observed[t] = f"{e['skill']}: {'ok' if e['ok'] else 'not ok'} ({e['reason']}) · reported at tick {e['end']}"
    ticks = []
    for t in idx:
        tr = z["truth"][t]
        ticks.append({
            "t": t, "time": round(float(z["sim_time"][t]), 1), "img": jpeg(z["frames"][t]),
            "duck": [round(float(tr[0]), 3), round(float(tr[1]), 3), round(float(tr[2]), 3)],
            "ball": [round(float(tr[3]), 3), round(float(tr[4]), 3)], "lamp": bool(tr[5]), "presses": int(tr[6]),
            "skill": H["skill_vocab"][int(z["skill"][t])], "activity": H["activity_vocab"][int(z["activity"][t])],
            "expected": H["expected_by_skill"][int(z["skill"][t])], "observed": observed[t],
            "exec": [round(float(v), 2) for v in z["cmd_exec"][t]], "bot": [round(float(v), 2) for v in z["cmd_bot"][t]],
            "mem": mem[t]})
    room = {"size": lay["room"]["size"], "home": lay["home"], "button": {"pos": lay["button"]["pos"], "size": lay["button"]["size"]},
            "pads": [{"pos": p["pos"], "size": p["size"], "rgb": to255(app["pad_rgb"][p["color"]]), "color": p["color"]} for p in lay["pads"]],
            "partition": lay["partition"], "occluder": lay["occluder"], "lamp": lay["lamp"],
            "floor": to255(app["floor"]["rgb1"]), "wall": to255(app["wall_rgb"])}
    ctl = H["controller"]
    if H.get("dagger"):
        ctl = f"learned policy drives ({Path(H['dagger']['policy']).parent.name}, beta {H['dagger']['beta']}); the bot runs in shadow"
    return {"name": Path(path).stem, "task": H["task"], "instruction": H["instruction"], "controller": ctl,
            "perturbation": H["perturbation"], "result": {k: H["result"].get(k) for k in ("success", "ticks", "sim_time")},
            "start": start, "room": room, "ticks": ticks, "events": H["events"]}


HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#f6f5f2;--panel:#fff;--ink:#1d1d1f;--mute:#6b6b70;--line:#e2e0da;--accent:#c2410c;--ok:#15803d;--bad:#b91c1c}
@media (prefers-color-scheme: dark){:root{--bg:#141416;--panel:#1d1d20;--ink:#ececee;--mute:#9a9aa2;--line:#2c2c31;--accent:#fb923c;--ok:#4ade80;--bad:#f87171}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
header{padding:16px 20px 8px}h1{font-size:18px;margin:0 0 4px}header p{margin:0;color:var(--mute);font-size:13px}
.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;padding:8px 20px}
select,button{font:inherit;padding:6px 10px;border:1px solid var(--line);border-radius:8px;background:var(--panel);color:var(--ink)}
button{cursor:pointer}input[type=range]{flex:1;min-width:160px}
main{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr) minmax(260px,1.1fr);gap:12px;padding:8px 20px 20px}
@media (max-width:900px){main{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:10px}
.card h2{font-size:12px;letter-spacing:.04em;text-transform:uppercase;color:var(--mute);margin:0 0 8px;font-weight:600}
canvas,img{width:100%;height:auto;display:block;border-radius:8px;image-rendering:pixelated;background:#000}
dl{display:grid;grid-template-columns:max-content 1fr;gap:4px 10px;margin:0;font-size:13px}dt{color:var(--mute)}dd{margin:0;word-break:break-word}
.ok{color:var(--ok)}.bad{color:var(--bad)}.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}
#mem{font-size:12px;margin-top:8px;max-height:220px;overflow:auto}#mem div{padding:2px 0;border-bottom:1px solid var(--line)}
.legend{font-size:12px;color:var(--mute);margin-top:6px}
</style></head><body>
<header><h1>__TITLE__</h1>
<p>__SUB__</p></header>
<div class="bar"><select id="ep"></select><button id="play">Play</button>
<input id="slider" type="range" min="0" max="0" value="0"><span id="clock" class="mono"></span>
<select id="speed"><option value="1">1×</option><option value="2" selected>2×</option><option value="4">4×</option></select></div>
<main>
<div class="card"><h2>Head camera</h2><img id="cam" alt="head camera frame"></div>
<div class="card"><h2>Room from above</h2><canvas id="top" width="420" height="420"></canvas>
<div class="legend">blue: duck (true) · rings: bot memory · magenta: ball (true) · square lamp: bright when on</div></div>
<div class="card"><h2>What the bot is doing</h2><dl id="info"></dl><div id="mem"></div></div>
</main>
<script>
const EPISODES = __DATA__;
const $ = id => document.getElementById(id);
let ep = 0, i = 0, timer = null;
EPISODES.forEach((e, k) => { const o = document.createElement('option'); o.value = k;
  o.textContent = `${e.task} · ${JSON.stringify(e.instruction.target)} · ${e.name}${e.perturbation.type ? ' · ' + e.perturbation.type : ''} · ${e.result.success ? 'success' : 'failure'}`; $('ep').appendChild(o); });
const COL = {pad_red:'#ef4444',pad_yellow:'#eab308',pad_green:'#22c55e',pad_blue:'#3b82f6',button:'#06b6d4',ball:'#d946ef',mat:'#f5f5f5',lamp_on:'#fde68a',lamp_off:'#525252'};
function odoToWorld(e, p){ const s = e.start, c = Math.cos(s.yaw), n = Math.sin(s.yaw);
  return [s.pos[0] + c*p[0] - n*p[1], s.pos[1] + n*p[0] + c*p[1]]; }
function drawTop(){
  const e = EPISODES[ep], T = e.ticks[i], cv = $('top'), g = cv.getContext('2d'), W = cv.width;
  const R = e.room, hx = R.size[0]/2, sc = W/(R.size[0]+0.2), X = x => (x+hx+0.1)*sc, Y = y => W-(y+hx+0.1)*sc;
  g.fillStyle = `rgb(${R.wall})`; g.fillRect(0,0,W,W);
  g.fillStyle = `rgb(${R.floor})`; g.fillRect(X(-hx),Y(hx),2*hx*sc,2*hx*sc);
  const sq = (p,s,col) => { g.fillStyle = col; g.fillRect(X(p[0]-s/2),Y(p[1]+s/2),s*sc,s*sc); };
  sq(R.home.pos, R.home.size, '#f2f2f2');
  R.pads.forEach(p => sq(p.pos, p.size, `rgb(${p.rgb})`));
  sq(R.button.pos, R.button.size+0.05, '#111'); sq(R.button.pos, R.button.size, '#22d3ee');
  const bar = (o,col) => { if(!o) return; g.save(); g.translate(X(o.pos[0]),Y(o.pos[1])); g.rotate(-o.yaw);
    g.fillStyle = col; g.fillRect(-o.length/2*sc,-o.thickness/2*sc,o.length*sc,Math.max(3,o.thickness*sc)); g.restore(); };
  bar(R.partition,'#555'); bar(R.occluder,'#333');
  const L = R.lamp, off = L.offset, lp = L.wall==='N'?[off,hx]:L.wall==='S'?[off,-hx]:L.wall==='E'?[hx,off]:[-hx,off];
  g.fillStyle = T.lamp ? '#fde047' : '#3f3f46'; g.fillRect(X(lp[0])-7,Y(lp[1])-7,14,14);
  g.strokeStyle = '#2563eb'; g.lineWidth = 2; g.beginPath();
  e.ticks.slice(0,i+1).forEach((k,j) => { const x = X(k.duck[0]), y = Y(k.duck[1]); j ? g.lineTo(x,y) : g.moveTo(x,y); }); g.stroke();
  g.fillStyle = '#d946ef'; g.beginPath(); g.arc(X(T.ball[0]),Y(T.ball[1]),5,0,7); g.fill();
  (T.mem.tracks||[]).forEach(m => { const w = odoToWorld(e,[m[1],m[2]]); g.strokeStyle = COL[m[0]]||'#fff'; g.lineWidth = 2;
    g.beginPath(); g.arc(X(w[0]),Y(w[1]),8,0,7); g.stroke(); });
  const d = T.duck; g.fillStyle = '#1d4ed8'; g.beginPath(); g.arc(X(d[0]),Y(d[1]),7,0,7); g.fill();
  g.strokeStyle = '#1d4ed8'; g.lineWidth = 3; g.beginPath(); g.moveTo(X(d[0]),Y(d[1]));
  g.lineTo(X(d[0]+0.18*Math.cos(d[2])),Y(d[1]+0.18*Math.sin(d[2]))); g.stroke();
}
function cmdText(c){ if (c[0] === null || Number.isNaN(c[0])) return '— (the bot has finished)'; return `vx ${c[0].toFixed(2)} wz ${c[2].toFixed(2)} · head p ${c[4].toFixed(2)} y ${c[5].toFixed(2)}${c[7] ? ' · DONE' : ''}`; }
function show(){
  const e = EPISODES[ep], T = e.ticks[i];
  $('cam').src = 'data:image/jpeg;base64,' + T.img; $('slider').max = e.ticks.length-1; $('slider').value = i;
  $('clock').textContent = `t = ${T.time.toFixed(1)} s`;
  const lastEv = e.events.filter(v => v.tick <= T.t).slice(-1)[0];
  const rows = [['goal', `${e.task} → ${JSON.stringify(e.instruction.target)}`],
    ['controller', e.controller + (e.perturbation.type ? ` · perturbation: ${e.perturbation.type} at tick ${e.perturbation.tick}` : '')],
    ['activity', T.activity.replaceAll('>', ' › ')], ['skill', T.skill], ['expected', T.expected], ['observed', T.observed + (T.observed !== '-' ? ` (now tick ${T.t})` : '')],
    ['command', cmdText(T.exec)], ['bot asked', cmdText(T.bot)],
    ['last event', lastEv ? `${lastEv.event} @${lastEv.tick}` : '—'],
    ['truth', `lamp ${T.lamp ? 'on' : 'off'} · presses ${T.presses}`],
    ...(e.note ? [['what happened', e.note]] : []),
    ['outcome', `<span class="${e.result.success ? 'ok' : 'bad'}">${e.result.success ? 'success' : 'failure'}</span> in ${e.result.sim_time} s`]];
  $('info').innerHTML = rows.map(([k,v]) => `<dt>${k}</dt><dd${k==='command'||k==='bot asked'?' class="mono"':''}>${v}</dd>`).join('');
  const mem = (T.mem.tracks||[]).map(m => `<div><b style="color:${COL[m[0]]||'inherit'}">${m[0]}</b> at (${m[1].toFixed(2)}, ${m[2].toFixed(2)}) · seen ${m[3]}× · last tick ${m[4]}</div>`).join('');
  $('mem').innerHTML = `<b>memory</b> (bot's own frame)${T.mem.lamp ? ` · lamp ${T.mem.lamp[0]} @${T.mem.lamp[1]}` : ''}${mem || '<div>—</div>'}`;
  drawTop();
}
function play(){ if(timer){ clearInterval(timer); timer = null; $('play').textContent = 'Play'; return; }
  $('play').textContent = 'Pause'; timer = setInterval(() => { const n = EPISODES[ep].ticks.length;
    if(i >= n-1){ play(); return; } i = Math.min(n-1, i + 1); show(); }, 200/Number($('speed').value)); }
$('ep').onchange = e => { ep = Number(e.target.value); i = 0; show(); };
$('slider').oninput = e => { i = Number(e.target.value); show(); };
$('play').onclick = play; $('speed').onchange = () => { if(timer){ play(); play(); } };
show();
</script></body></html>"""


TITLE = "Microduck replay — reference bot demonstrations"
SUB = ("Head camera as recorded (96×96), the room from above (true path in blue; the bot's remembered objects as "
       "rings, placed from its own odometry frame), and what the bot was doing each tick. Truth is shown for "
       "replay only; the bot never sees it.")


def main(paths, out=HERE / "media" / "demo.html", title=TITLE, sub=SUB, notes=None, every=2):
    """notes: one explanation per episode (a list aligned with paths), shown as 'what happened'."""
    sys.path.insert(0, str(HERE))
    data = [episode_payload(p, every) for p in paths]
    for d, n in zip(data, notes or [None] * len(data)):
        d["note"] = n
    html = HTML.replace("__TITLE__", title).replace("__SUB__", sub).replace("__DATA__", json.dumps(data, separators=(",", ":")))
    out.write_text(html)
    print(out, round(len(html) / 1e6, 2), "MB", [d["name"] for d in data])


if __name__ == "__main__":
    main(sys.argv[1:])
