// Terra demo UI: painting, live replanning, D* Lite incremental repair, walking agent.
(() => {
const T = TERRA;
const canvas = document.getElementById("view");
const ctx = canvas.getContext("2d");
const PAL = [[0.62, 0.76, 0.42], [0.22, 0.45, 0.25], [0.60, 0.56, 0.52], [0.20, 0.42, 0.78], [0.80, 0.52, 0.22]];
const PLANNERS = [
  { id: "dijkstra", name: "Dijkstra", color: "#ffffff", width: 6 },
  { id: "astar", name: "A* (octile)", color: "#ffd400", width: 4.5 },
  { id: "dstar", name: "D* Lite", color: "#ff3b30", width: 2.5 },
  { id: "wastar", name: "Weighted A*", color: "#00e5ff", width: 2.5 },
  { id: "learned", name: "Learned-h A*", color: "#c77dff", width: 2.5 },
];
const TOOLS = [
  { id: "start", label: "Start", sw: "#ffffff" }, { id: "goal", label: "Goal", sw: "#ff2bd6" },
  { id: "water", label: "Water", kind: T.WATER }, { id: "bridge", label: "Bridge", kind: T.BRIDGE },
  { id: "grass", label: "Grass", kind: T.GRASS }, { id: "forest", label: "Forest", kind: T.FOREST },
  { id: "rock", label: "Rock", kind: T.ROCK }, { id: "raise", label: "Raise ▲", sw: "#d9c7a0" },
  { id: "lower", label: "Lower ▼", sw: "#6f5d45" },
];

const S = {
  map: null, G: null, D: null, start: 0, goal: 0, agent: 0, tool: "water", brush: 1,
  show: { dijkstra: true, astar: true, dstar: true, wastar: true, learned: true }, explored: "dstar",
  learnedH: null, learnedMs: 0, learnedTimer: null,
  results: {}, walking: false, timer: null, travel: 0, steps: 0, trail: [], contours: false,
  lastDstarTrace: [], weight: 2, randomSeed: 1, dirty: true,
};

// ---------------------------------------------------------------- map loading
function b64ToBytes(s) { const bin = atob(s); const out = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i); return out; }
function loadBaked(i) {
  const m = BAKED[i];
  const elev = new Float32Array(b64ToBytes(m.elev).buffer);
  const kind = new Int8Array(b64ToBytes(m.kind).buffer);
  return { h: m.h, w: m.w, elev, kind, start: m.start, goal: m.goal, name: m.name };
}
function setMap(m) {
  S.map = m;
  S.G = new T.Graph(m.h, m.w, m.elev, m.kind);
  S.start = m.start; S.goal = m.goal;
  ensurePassable(S.start); ensurePassable(S.goal);
  resetAgent();
}
function ensurePassable(u) {
  if (T.MULT[S.G.kind[u]] === Infinity) S.G.setCells([u], T.GRASS, null);
}
function newDStar() {
  S.D = new T.DStarLite(S.G, S.agent, S.goal);
  const r = S.D.compute();
  S.lastDstarTrace = [];  // overlay shows incremental repairs only
  S.results.dstar = { expanded: r.expanded, ms: r.ms, note: "initial search" };
}
function resetAgent() {
  stopWalk();
  S.agent = S.start; S.travel = 0; S.steps = 0; S.trail = [S.start];
  newDStar();
  scheduleLearned();
  replanScratch();
  logEdit(`New search from scratch. D* Lite initial search expanded <b>${S.results.dstar.expanded}</b> nodes.`);
  S.dirty = true;
}

// ---------------------------------------------------------------- planning
function replanScratch() {
  const G = S.G, h = T.octileField(G, S.goal);
  S.results.dijkstra = T.astar(G, S.agent, S.goal, null);
  S.results.astar = T.astar(G, S.agent, S.goal, h);
  S.results.wastar = T.astar(G, S.agent, S.goal, h, S.weight);
  if (S.learnedH) S.results.learned = T.astar(G, S.agent, S.goal, S.learnedH, 1.0);
  else S.results.learned = null;
  const p = S.D.path();
  S.results.dstar = Object.assign(S.results.dstar || {}, { path: p, cost: S.D.costToGo() });
}
// cells changed -> D* Lite repairs incrementally; the others replan from scratch
function applyChange(affected, what) {
  S.D.trace = [];
  const t0 = performance.now();
  S.D.updateVertices(affected);
  const r = S.D.compute();
  const ms = performance.now() - t0;
  S.lastDstarTrace = S.D.trace; S.D.trace = null;
  S.results.dstar = { expanded: r.expanded, ms };
  if (what !== "Start moved") scheduleLearned();
  replanScratch();
  const a = S.results.astar;
  const ratio = r.expanded > 0 ? (a.expanded / r.expanded) : Infinity;
  const verdict = r.expanded < a.expanded
    ? `<span class="ok">${ratio === Infinity ? "nothing to repair" : ratio.toFixed(1) + "× fewer nodes"}</span>`
    : `<span class="bad">more nodes than A* this time</span>`;
  logEdit(`${what}: D* Lite repaired <b>${r.expanded}</b> nodes in ${ms.toFixed(1)} ms · A* from scratch expanded <b>${a.expanded}</b> (${a.ms.toFixed(1)} ms) · ${verdict}` +
    (S.D.costToGo() === Infinity ? ` · <span class="bad">goal unreachable!</span>` : ""));
  S.dirty = true;
}
// learned heuristic: one U-Net forward pass per (map, goal), run in a Web Worker so painting
// stays smooth; the result is cached while the agent walks (h does not depend on the start).
let netWorker = null, netReq = 0;
function getWorker() {
  if (netWorker !== null) return netWorker;
  try {
    const src = document.getElementById("terra-core").textContent + "\n" + document.getElementById("hnet").textContent +
      "\nonmessage = e => { const d = e.data; if (d.spec) { HNET.load(d.spec); return; }" +
      " const G = new TERRA.Graph(d.h, d.w, d.elev, d.kind); const t0 = performance.now();" +
      " const out = HNET.heuristic(G, d.goal); postMessage({ id: d.id, h: out.h, ms: performance.now() - t0 }, [out.h.buffer]); };";
    netWorker = new Worker(URL.createObjectURL(new Blob([src], { type: "text/javascript" })));
    netWorker.postMessage({ spec: HNET_SPEC });
    netWorker.onmessage = e => { if (e.data.id === netReq) learnedReady(e.data.h, e.data.ms); };
    netWorker.onerror = err => {  // e.g. workers blocked for this origin: fall back to the main thread
      console.warn("learned-h worker failed, using main thread", err && err.message);
      netWorker = false;
      const t0 = performance.now(); learnedReady(HNET.heuristic(S.G, S.goal).h, performance.now() - t0);
    };
  } catch (err) { netWorker = false; }
  return netWorker;
}
function learnedReady(h, ms) {
  S.learnedH = h; S.learnedMs = ms;
  S.results.learned = T.astar(S.G, S.agent, S.goal, S.learnedH, 1.0);
  setNetNote(`U-Net "${HNET_SPEC.tag}" (base ${HNET_SPEC.base}${HNET_SPEC.tag === "hnet16" ? "; a browser-sized sibling of the evaluated base-32 model, see RESULTS.md appendix" : ""}) forward pass in JS (web worker): ${ms.toFixed(0)} ms per map+goal. A* then uses h = octile·e<sup>r</sup>, ` +
    `which is not admissible, so the route can be slightly suboptimal.`);
  S.dirty = true;
}
function scheduleLearned() {
  S.learnedH = null; S.results.learned = null;
  if (typeof HNET === "undefined" || !HNET.loaded || !S.show.learned) return;
  clearTimeout(S.learnedTimer);
  setNetNote("computing learned heuristic…");
  const id = ++netReq;
  S.learnedTimer = setTimeout(() => {
    const w = getWorker();
    if (w) w.postMessage({ id, h: S.G.h, w: S.G.w, elev: S.G.elev, kind: S.G.kind, goal: S.goal });
    else { const t0 = performance.now(); const h = HNET.heuristic(S.G, S.goal).h; learnedReady(h, performance.now() - t0); }
  }, 150);
}
function setNetNote(html) { const el = document.getElementById("netnote"); if (el) el.innerHTML = html; }
function logEdit(html) { document.getElementById("editlog").innerHTML = html; }
function toast(msg) {
  const t = document.getElementById("toast"); t.textContent = msg; t.style.opacity = 1;
  clearTimeout(toast._t); toast._t = setTimeout(() => (t.style.opacity = 0), 1600);
}

// ---------------------------------------------------------------- agent
function stepAgent() {
  const p = S.D.path();
  if (S.agent === S.goal) { stopWalk(); toast("Arrived!"); return; }
  if (p.length < 2) { stopWalk(); toast("No route to the goal"); return; }
  const nxt = p[1];
  S.travel += S.G.edgeCost(S.agent, nxt);
  S.agent = nxt; S.steps++; S.trail.push(nxt);
  S.D.moveTo(nxt);
  replanScratch();
  S.dirty = true;
}
function startWalk() {
  if (S.walking) return stopWalk();
  if (S.agent === S.goal) resetAgent();
  S.walking = true; document.getElementById("walk").textContent = "❚❚ Pause";
  S.timer = setInterval(stepAgent, 90);
}
function stopWalk() {
  S.walking = false; clearInterval(S.timer);
  const b = document.getElementById("walk"); if (b) b.textContent = "▶ Walk";
}
function collapseNextBridge() {
  const p = S.D.path(), G = S.G;
  const first = p.find(u => G.kind[u] === T.BRIDGE);
  if (first === undefined) { toast("No bridge on the current route"); return; }
  // flood fill the connected bridge cells
  const seen = new Set([first]), stack = [first];
  while (stack.length) {
    const u = stack.pop(), r = (u / G.w) | 0, c = u - r * G.w;
    for (const [dr, dc] of T.DIRS.slice(0, 4)) {
      const rr = r + dr, cc = c + dc, v = rr * G.w + cc;
      if (rr >= 0 && rr < G.h && cc >= 0 && cc < G.w && !seen.has(v) && G.kind[v] === T.BRIDGE) { seen.add(v); stack.push(v); }
    }
  }
  const cells = [...seen].filter(u => u !== S.agent && u !== S.goal);
  const aff = G.setCells(cells, T.WATER, null);
  applyChange(aff, `Bridge collapsed (${cells.length} cells)`);
  toast("Bridge collapsed!");
}

// ---------------------------------------------------------------- painting
function cellAt(ev) {
  const rect = canvas.getBoundingClientRect();
  const x = (ev.clientX - rect.left) / rect.width * S.G.w, y = (ev.clientY - rect.top) / rect.height * S.G.h;
  return { r: Math.max(0, Math.min(S.G.h - 1, Math.floor(y))), c: Math.max(0, Math.min(S.G.w - 1, Math.floor(x))) };
}
let painting = false, pending = new Set(), lastCell = -1, strokeLabel = "";
function paintAt(ev) {
  const { r, c } = cellAt(ev), G = S.G, u = r * G.w + c;
  const tool = TOOLS.find(t => t.id === S.tool);
  if (tool.id === "start" || tool.id === "goal") {
    if (T.MULT[G.kind[u]] === Infinity) { toast("Pick a dry cell"); return; }
    if (u === lastCell) return; lastCell = u;
    stopWalk();
    if (tool.id === "start") { S.start = u; S.agent = u; S.travel = 0; S.steps = 0; S.trail = [u];
      S.D.moveTo(u); applyChange([], "Start moved"); }
    else { S.goal = u; newDStar(); scheduleLearned(); replanScratch(); logEdit(`Goal moved: D* Lite must restart (its search is rooted at the goal). Initial search: <b>${S.results.dstar.expanded}</b> nodes.`); S.dirty = true; }
    return;
  }
  const rad = S.brush, changed = [];
  for (let dr = -rad; dr <= rad; dr++) for (let dc = -rad; dc <= rad; dc++) {
    const rr = r + dr, cc = c + dc;
    if (rr < 0 || rr >= G.h || cc < 0 || cc >= G.w || dr * dr + dc * dc > rad * rad + rad) continue;
    const v = rr * G.w + cc;
    if (tool.kind !== undefined) {
      if ((v === S.agent || v === S.goal) && T.MULT[tool.kind] === Infinity) continue;
      if (G.kind[v] !== tool.kind) { G.kind[v] = tool.kind; changed.push(v); }
    } else {
      const fall = Math.exp(-(dr * dr + dc * dc) / (2 * Math.max(1, rad * rad / 2)));
      const e = Math.min(1, Math.max(0, G.elev[v] + (tool.id === "raise" ? 1 : -1) * 0.03 * fall));
      if (e !== G.elev[v]) { G.elev[v] = e; changed.push(v); }
    }
  }
  if (changed.length) {
    for (const v of G.touch(changed)) pending.add(v);
    strokeLabel = `Painted ${tool.label.replace(/ [▲▼]/, "").toLowerCase()}`;
  }
}
function flushPaint() {
  if (pending.size) { applyChange(pending, strokeLabel); pending = new Set(); }
}
canvas.addEventListener("pointerdown", ev => { painting = true; lastCell = -1; canvas.setPointerCapture(ev.pointerId); paintAt(ev); });
canvas.addEventListener("pointermove", ev => { if (painting) paintAt(ev); });
const endPaint = () => { if (painting) { painting = false; flushPaint(); } };
canvas.addEventListener("pointerup", endPaint);
canvas.addEventListener("pointercancel", endPaint);

// ---------------------------------------------------------------- rendering
const off = document.createElement("canvas"); const offx = off.getContext("2d");
function terrainImage() {
  const G = S.G, h = G.h, w = G.w;
  off.width = w; off.height = h;
  const img = offx.createImageData(w, h);
  const z = i => G.elev[i] * 24;
  for (let r = 0; r < h; r++) for (let c = 0; c < w; c++) {
    const i = r * w + c;
    const gx = (z(r * w + Math.min(w - 1, c + 1)) - z(r * w + Math.max(0, c - 1))) / ((c > 0 && c < w - 1) ? 2 : 1);
    const gy = (z(Math.min(h - 1, r + 1) * w + c) - z(Math.max(0, r - 1) * w + c)) / ((r > 0 && r < h - 1) ? 2 : 1);
    const slope = Math.PI / 2 - Math.atan(Math.hypot(gx, gy)), aspect = Math.atan2(-gx, gy);
    const az = 315 * Math.PI / 180, alt = 45 * Math.PI / 180;
    let hs = Math.sin(alt) * Math.sin(slope) + Math.cos(alt) * Math.cos(slope) * Math.cos(az - aspect);
    hs = Math.max(0, Math.min(1, hs));
    const k = G.kind[i], base = PAL[k];
    let f = (0.55 + 0.45 * hs) * (0.85 + 0.25 * G.elev[i]);
    if (k === T.WATER) f = 0.9 + 0.1 * hs;
    for (let ch = 0; ch < 3; ch++) img.data[4 * i + ch] = Math.min(255, base[ch] * f * 255);
    img.data[4 * i + 3] = 255;
  }
  offx.putImageData(img, 0, 0);
}
function draw() {
  if (!S.dirty) return requestAnimationFrame(draw);
  S.dirty = false;
  const G = S.G, W = canvas.width, sc = W / G.w;
  terrainImage();
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, 0, 0, W, W);
  if (S.contours) drawContours(sc);
  // explored overlay
  let nodes = null;
  if (S.explored === "dstar") nodes = S.lastDstarTrace;
  else if (S.explored && S.results[S.explored] && S.results[S.explored].closed) {
    const cl = S.results[S.explored].closed; nodes = [];
    for (let i = 0; i < cl.length; i++) if (cl[i]) nodes.push(i);
  }
  if (nodes) {
    ctx.fillStyle = S.explored === "dstar" ? "rgba(255,40,40,0.45)" : "rgba(255,255,255,0.30)";
    for (const u of nodes) { const r = (u / G.w) | 0, c = u - r * G.w; ctx.fillRect(c * sc, r * sc, sc, sc); }
  }
  // bridges outline
  ctx.strokeStyle = "rgba(255,200,120,0.9)"; ctx.lineWidth = 1;
  for (let u = 0; u < G.n; u++) if (G.kind[u] === T.BRIDGE) { const r = (u / G.w) | 0, c = u - r * G.w; ctx.strokeRect(c * sc + .5, r * sc + .5, sc - 1, sc - 1); }
  // trail
  drawPath(S.trail, "rgba(255,255,255,0.85)", 2, sc, [4, 4]);
  for (const P of PLANNERS) {
    const res = S.results[P.id];
    if (S.show[P.id] && res && res.path && res.path.length) drawPath(res.path, P.color, P.width, sc);
  }
  marker(S.goal, "#ff2bd6", sc, true);
  marker(S.agent, "#ffffff", sc, false);
  updateStats();
  requestAnimationFrame(draw);
}
function drawPath(path, color, width, sc, dash) {
  if (!path || path.length < 2) return;
  const w = S.G.w;
  ctx.save(); ctx.lineJoin = "round"; ctx.lineCap = "round";
  ctx.beginPath();
  path.forEach((u, i) => { const r = (u / w) | 0, c = u - r * w; const x = (c + .5) * sc, y = (r + .5) * sc; i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
  if (!dash) { ctx.strokeStyle = "rgba(0,0,0,0.55)"; ctx.lineWidth = width + 2.5; ctx.stroke(); }
  ctx.setLineDash(dash || []); ctx.strokeStyle = color; ctx.lineWidth = width; ctx.stroke();
  ctx.restore();
}
function marker(u, color, sc, star) {
  const w = S.G.w, r = (u / w) | 0, c = u - r * w, x = (c + .5) * sc, y = (r + .5) * sc, R = Math.max(6, sc * 1.4);
  ctx.save(); ctx.fillStyle = color; ctx.strokeStyle = "#000"; ctx.lineWidth = 2;
  ctx.beginPath();
  if (star) { for (let i = 0; i < 10; i++) { const a = -Math.PI / 2 + i * Math.PI / 5, rr = i % 2 ? R * 0.45 : R * 1.15; ctx.lineTo(x + rr * Math.cos(a), y + rr * Math.sin(a)); } ctx.closePath(); }
  else ctx.arc(x, y, R * 0.8, 0, 2 * Math.PI);
  ctx.fill(); ctx.stroke(); ctx.restore();
}
function drawContours(sc) {
  const G = S.G, w = G.w, step = 0.08;
  ctx.fillStyle = "rgba(0,0,0,0.22)";
  for (let r = 0; r < G.h - 1; r++) for (let c = 0; c < w - 1; c++) {
    const a = Math.floor(G.elev[r * w + c] / step), b = Math.floor(G.elev[r * w + c + 1] / step), d = Math.floor(G.elev[(r + 1) * w + c] / step);
    if (a !== b || a !== d) ctx.fillRect(c * sc + sc * 0.35, r * sc + sc * 0.35, sc * 0.3, sc * 0.3);
  }
}
function fmt(x, d = 1) { return x === undefined ? "–" : x === Infinity ? "∞" : x.toFixed(d); }
function updateStats() {
  const tb = document.getElementById("stats");
  if (!tb.children.length) {
    for (const P of PLANNERS) {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td><label><input type="checkbox" ${S.show[P.id] ? "checked" : ""} data-p="${P.id}"><span class="chip" style="background:${P.color}"></span>${P.name}</label></td>
        <td data-k="cost"></td><td data-k="exp"></td><td data-k="ms"></td><td><input type="radio" name="expl" value="${P.id}" ${S.explored === P.id ? "checked" : ""}></td>`;
      tb.appendChild(tr);
    }
    const tr = document.createElement("tr");
    tr.innerHTML = `<td class="muted">no overlay</td><td></td><td></td><td></td><td><input type="radio" name="expl" value=""></td>`;
    tb.appendChild(tr);
    tb.addEventListener("change", ev => {
      const t = ev.target;
      if (t.dataset.p) { S.show[t.dataset.p] = t.checked; if (t.dataset.p === "learned" && t.checked && !S.learnedH) scheduleLearned(); }
      if (t.name === "expl") S.explored = t.value;
      S.dirty = true;
    });
  }
  const opt = S.results.dijkstra ? S.results.dijkstra.cost : NaN;
  PLANNERS.forEach((P, i) => {
    const tr = tb.children[i], res = S.results[P.id] || {};
    let c = fmt(res.cost);
    if ((P.id === "wastar" || P.id === "learned") && isFinite(res.cost) && isFinite(opt)) c += ` <span class="muted">(+${(100 * (res.cost / opt - 1)).toFixed(1)}%)</span>`;
    tr.querySelector('[data-k=cost]').innerHTML = c;
    tr.querySelector('[data-k=exp]').textContent = res.expanded === undefined ? "–" : res.expanded + (P.id === "dstar" ? "*" : "");
    tr.querySelector('[data-k=ms]').textContent = fmt(res.ms, 1);
  });
  const remain = S.D.costToGo();
  document.getElementById("agentinfo").innerHTML =
    `walked ${S.steps} steps, travel cost <b>${S.travel.toFixed(1)}</b> · remaining (D* Lite) <b>${fmt(remain)}</b>` +
    `<br><span class="muted">* D* Lite count = nodes touched by its <i>last</i> repair, not a full search. Explored overlay for D* Lite shows that repair.</span>`;
}

// ---------------------------------------------------------------- controls
function buildControls() {
  const sel = document.getElementById("mapsel");
  BAKED.forEach((m, i) => { const o = document.createElement("option"); o.value = i; o.textContent = m.name; sel.appendChild(o); });
  // option values: "0".."3" = baked maps, "r0", "r1", ... = generated maps (kept pristine for reset)
  S.randomMaps = [];
  const mapFor = v => (v[0] === "r" ? S.randomMaps[+v.slice(1)] : loadBaked(+v));
  sel.addEventListener("change", () => setMap(mapFor(sel.value)));
  document.getElementById("newmap").addEventListener("click", () => {
    const g = T.generate(S.randomSeed++, 96);
    // start/goal: far-apart dry cells
    const dry = []; for (let u = 0; u < g.kind.length; u++) if (T.MULT[g.kind[u]] !== Infinity) dry.push(u);
    const near = (r, c) => dry.reduce((b, u) => { const d = Math.hypot(((u / 96) | 0) - r, (u % 96) - c); return d < b[0] ? [d, u] : b; }, [1e9, -1])[1];
    const m = { ...g, start: near(6, 6), goal: near(89, 89), name: `random #${S.randomSeed - 1}` };
    S.randomMaps.push(m);
    const o = document.createElement("option"); o.value = "r" + (S.randomMaps.length - 1); o.textContent = m.name;
    sel.appendChild(o); sel.value = o.value;
    setMap(m);  // Graph copies elev/kind, so m stays pristine for "Reset map"
  });
  document.getElementById("resetmap").addEventListener("click", () => setMap(mapFor(sel.value)));
  const tools = document.getElementById("tools");
  for (const t of TOOLS) {
    const b = document.createElement("button"); b.className = "tool"; b.dataset.t = t.id;
    const col = t.kind !== undefined ? `rgb(${PAL[t.kind].map(x => Math.round(x * 255)).join(",")})` : t.sw;
    b.innerHTML = `<span class="sw" style="background:${col}"></span>${t.label}`;
    b.addEventListener("click", () => { S.tool = t.id; [...tools.children].forEach(x => x.classList.toggle("on", x.dataset.t === t.id)); });
    tools.appendChild(b);
  }
  [...tools.children].forEach(x => x.classList.toggle("on", x.dataset.t === S.tool));
  const br = document.getElementById("brush");
  br.addEventListener("input", () => { S.brush = +br.value; document.getElementById("brushv").textContent = br.value; });
  const ws = document.getElementById("wslider");
  ws.addEventListener("input", () => { S.weight = +ws.value; document.getElementById("wv").textContent = S.weight.toFixed(2); replanScratch(); S.dirty = true; });
  document.getElementById("contours").addEventListener("change", e => { S.contours = e.target.checked; S.dirty = true; });
  document.getElementById("walk").addEventListener("click", startWalk);
  document.getElementById("collapse").addEventListener("click", collapseNextBridge);
  document.getElementById("resetagent").addEventListener("click", resetAgent);
}
function fit() {
  const avail = Math.min(window.innerWidth - 32, window.innerHeight - 120, 768);
  const px = Math.max(288, Math.floor(avail / 96) * 96);
  canvas.width = canvas.height = px * (window.devicePixelRatio > 1 ? 1 : 1);
  canvas.style.width = canvas.style.height = px + "px";
  S.dirty = true;
}
window.addEventListener("resize", fit);
buildControls();
fit();
setMap(loadBaked(0));
requestAnimationFrame(draw);
window.TERRA_STATE = S;  // for debugging / tests
window.TERRA_DEMO = { selectMap: v => { const sel = document.getElementById("mapsel"); sel.value = v; sel.dispatchEvent(new Event("change")); }, stepAgent, collapseNextBridge, resetAgent, paintAt, flushPaint, setTool: t => { S.tool = t; } };
// demo.html#selftest: scripted interaction; checks D* Lite == Dijkstra after every change
if (location.hash === "#selftest") {
  let bad = 0, checks = 0;
  const check = () => { checks++; const a = S.D.costToGo(), b = S.results.dijkstra.cost;
    if (!((a === Infinity && b === Infinity) || Math.abs(a - b) < 1e-6)) bad++; };
  for (let i = 0; i < 6; i++) stepAgent(); check();
  collapseNextBridge(); check();
  for (let i = 0; i < 25; i++) stepAgent(); check();
  const G = S.G;
  for (let k = 0; k < 15; k++) {   // random paint strokes with every tool
    const tool = TOOLS[2 + (k % 7)]; S.tool = tool.id; S.brush = k % 3;
    const u = (k * 2654435761 >>> 0) % G.n, r = (u / G.w) | 0, c = u - r * G.w, rect = canvas.getBoundingClientRect();
    paintAt({ clientX: rect.left + (c + .5) / G.w * rect.width, clientY: rect.top + (r + .5) / G.h * rect.height });
    flushPaint(); check();
  }
  S.explored = "dstar"; S.tool = "water";
  let learnedMsg = "learned: not baked";
  if (typeof HNET !== "undefined" && HNET.loaded) {
    const t0 = performance.now(), L = HNET.heuristic(S.G, S.goal), ms = performance.now() - t0;
    const lr = T.astar(S.G, S.agent, S.goal, L.h, 1.0), opt = S.results.dijkstra.cost;
    if (!(lr.cost >= opt - 1e-9)) bad++;
    learnedMsg = `learned: cost ${(lr.cost / opt).toFixed(3)}x opt, expanded ${lr.expanded} vs A* ${S.results.astar.expanded}, net ${ms.toFixed(0)} ms`;
  }
  document.getElementById("editlog").innerHTML += `<div id="selftest">SELFTEST ${bad === 0 ? "PASS" : "FAIL"} checks=${checks} bad=${bad} | ${learnedMsg}</div>`;
}
})();
