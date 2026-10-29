// Terra core: terrain kinds, THE cost definition, planners (Dijkstra, A*, weighted A*,
// D* Lite) and a procedural generator. A line-by-line port of terra/cost.py and
// terra/planners.py; tests/test_js_parity.py runs it in node and checks it against Python.
const TERRA = (() => {
  const GRASS = 0, FOREST = 1, ROCK = 2, WATER = 3, BRIDGE = 4;
  const MULT = [1.0, 2.0, 3.0, Infinity, 1.0];
  const HEIGHT_SCALE = 24.0, SLOPE_K = 1.0, MIN_UNIT_COST = 1.0, SQRT2 = Math.SQRT2;
  const DIRS = [[-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [-1, 1], [1, -1], [1, 1]];

  // ------------------------------------------------------------------ cost graph
  class Graph {
    constructor(h, w, elev, kind) {
      this.h = h; this.w = w; this.n = h * w;
      this.elev = Float32Array.from(elev);
      this.kind = Int8Array.from(kind);
      this.nbr = new Int32Array(this.n * 8).fill(-1);
      this.cst = new Float64Array(this.n * 8).fill(Infinity);
      for (let u = 0; u < this.n; u++) this.rebuildNode(u);
    }
    passable(r, c) { return MULT[this.kind[r * this.w + c]] !== Infinity; }
    // c(a,b) = d * m_bar * (1 + SLOPE_K * |z_b - z_a| / d), no corner cutting, water impassable
    rebuildNode(u) {
      const w = this.w, h = this.h, r = (u / w) | 0, c = u - r * w;
      const ka = this.kind[u], ma = MULT[ka];
      for (let k = 0; k < 8; k++) {
        const dr = DIRS[k][0], dc = DIRS[k][1], rr = r + dr, cc = c + dc, i = u * 8 + k;
        this.nbr[i] = -1; this.cst[i] = Infinity;
        if (rr < 0 || rr >= h || cc < 0 || cc >= w) continue;
        const v = rr * w + cc, mb = MULT[this.kind[v]];
        if (ma === Infinity || mb === Infinity) continue;
        let d = 1.0;
        if (dr && dc) {
          if (!this.passable(rr, c) || !this.passable(r, cc)) continue;
          d = SQRT2;
        }
        const za = this.elev[u] * HEIGHT_SCALE, zb = this.elev[v] * HEIGHT_SCALE;
        const mbar = 0.5 * (ma + mb);
        this.nbr[i] = v;
        this.cst[i] = d * mbar * (1.0 + SLOPE_K * Math.abs(zb - za) / d);
      }
    }
    // cells: array of node ids. kind/elev: value or null. Returns affected node ids.
    setCells(cells, kind, elev) {
      for (const u of cells) {
        if (kind !== null && kind !== undefined) this.kind[u] = kind;
        if (elev !== null && elev !== undefined) this.elev[u] = elev;
      }
      return this.touch(cells);
    }
    touch(cells) {
      const aff = new Set(), w = this.w;
      for (const u of cells) {
        const r = (u / w) | 0, c = u - r * w;
        for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) {
          const rr = r + dr, cc = c + dc;
          if (rr >= 0 && rr < this.h && cc >= 0 && cc < w) aff.add(rr * w + cc);
        }
      }
      for (const u of aff) this.rebuildNode(u);
      return aff;
    }
    edgeCost(a, b) {
      for (let k = 0; k < 8; k++) if (this.nbr[a * 8 + k] === b) return this.cst[a * 8 + k];
      return Infinity;
    }
    octile(a, b) {
      const w = this.w, ar = (a / w) | 0, ac = a - ar * w, br = (b / w) | 0, bc = b - br * w;
      const dr = Math.abs(ar - br), dc = Math.abs(ac - bc);
      return MIN_UNIT_COST * (Math.max(dr, dc) + (SQRT2 - 1.0) * Math.min(dr, dc));
    }
  }

  function pathCost(G, path) {
    let t = 0;
    for (let i = 0; i + 1 < path.length; i++) t += G.edgeCost(path[i], path[i + 1]);
    return t;
  }

  // ------------------------------------------------------------------ binary heap on (k1,k2,v)
  class Heap {
    constructor() { this.a = []; this.b = []; this.v = []; }
    get size() { return this.v.length; }
    less(i, j) {
      const a = this.a, b = this.b;
      if (a[i] !== a[j]) return a[i] < a[j];
      if (b[i] !== b[j]) return b[i] < b[j];
      return this.v[i] < this.v[j];
    }
    swap(i, j) {
      let t = this.a[i]; this.a[i] = this.a[j]; this.a[j] = t;
      t = this.b[i]; this.b[i] = this.b[j]; this.b[j] = t;
      t = this.v[i]; this.v[i] = this.v[j]; this.v[j] = t;
    }
    push(k1, k2, v) {
      this.a.push(k1); this.b.push(k2); this.v.push(v);
      let i = this.v.length - 1;
      while (i > 0) { const p = (i - 1) >> 1; if (this.less(i, p)) { this.swap(i, p); i = p; } else break; }
    }
    pop() { // removes top; read top via a[0], b[0], v[0] first
      const n = this.v.length - 1;
      this.swap(0, n); this.a.pop(); this.b.pop(); this.v.pop();
      let i = 0;
      for (;;) {
        const l = 2 * i + 1, r = l + 1; let m = i;
        if (l < n && this.less(l, m)) m = l;
        if (r < n && this.less(r, m)) m = r;
        if (m === i) break;
        this.swap(i, m); i = m;
      }
    }
  }

  // ------------------------------------------------------------------ A* / Dijkstra
  // h: Float64Array of heuristic values to goal (or null => Dijkstra). weight: WA* inflation.
  function astar(G, s, g, h, weight = 1.0) {
    const t0 = performance.now();
    const n = G.n, gs = new Float64Array(n).fill(Infinity), parent = new Int32Array(n).fill(-1);
    const closed = new Uint8Array(n);
    const pq = new Heap();
    gs[s] = 0;
    pq.push(h ? weight * h[s] : 0, -0, s);
    let expanded = 0, found = false;
    while (pq.size) {
      const u = pq.v[0]; pq.pop();
      if (closed[u]) continue;
      closed[u] = 1; expanded++;
      if (u === g) { found = true; break; }
      const gu = gs[u];
      for (let k = 0; k < 8; k++) {
        const v = G.nbr[u * 8 + k];
        if (v < 0) continue;
        const nv = gu + G.cst[u * 8 + k];
        if (nv < gs[v]) {
          gs[v] = nv; parent[v] = u;
          pq.push(h ? nv + weight * h[v] : nv, -nv, v);
        }
      }
    }
    const ms = performance.now() - t0;
    if (!found) return { path: [], cost: Infinity, expanded, ms, closed };
    const path = [g];
    while (path[path.length - 1] !== s) path.push(parent[path[path.length - 1]]);
    path.reverse();
    return { path, cost: gs[g], expanded, ms, closed };
  }

  function octileField(G, g) {
    const h = new Float64Array(G.n);
    for (let u = 0; u < G.n; u++) h[u] = G.octile(u, g);
    return h;
  }

  // ------------------------------------------------------------------ D* Lite (Koenig & Likhachev 2002)
  class DStarLite {
    constructor(G, start, goal) {
      this.G = G; this.sStart = start; this.sLast = start; this.goal = goal; this.km = 0;
      const n = G.n;
      this.g = new Float64Array(n).fill(Infinity);
      this.rhs = new Float64Array(n).fill(Infinity);
      this.k1 = new Float64Array(n).fill(NaN);  // key currently in queue (NaN = not queued)
      this.k2 = new Float64Array(n).fill(NaN);
      this.U = new Heap();
      this.rhs[goal] = 0;
      this.trace = null;
      this._insert(goal);
    }
    _h(s) { return this.G.octile(s, this.sStart); }
    _insert(s) {
      const m = Math.min(this.g[s], this.rhs[s]);
      const a = m + this._h(s) + this.km, b = m;
      if (this.k1[s] === a && this.k2[s] === b) return;
      this.k1[s] = a; this.k2[s] = b; this.U.push(a, b, s);
    }
    _top() {
      const U = this.U;
      while (U.size) {
        const s = U.v[0];
        if (U.a[0] === this.k1[s] && U.b[0] === this.k2[s]) return s;
        U.pop();
      }
      return -1;
    }
    _update(u) {
      if (this.g[u] !== this.rhs[u]) this._insert(u);
      else { this.k1[u] = NaN; this.k2[u] = NaN; }
    }
    _minSucc(u) {
      let best = Infinity; const G = this.G;
      for (let k = 0; k < 8; k++) {
        const v = G.nbr[u * 8 + k];
        if (v < 0) continue;
        const x = G.cst[u * 8 + k] + this.g[v];
        if (x < best) best = x;
      }
      return best;
    }
    compute() {
      const t0 = performance.now();
      const G = this.G, g = this.g, rhs = this.rhs, s0 = this.sStart, goal = this.goal;
      let expanded = 0;
      for (;;) {
        const u = this._top();
        const ka = u < 0 ? Infinity : this.U.a[0], kb = u < 0 ? Infinity : this.U.b[0];
        const ms = Math.min(g[s0], rhs[s0]);
        const sa = ms + this._h(s0) + this.km, sb = ms;
        const less = ka < sa || (ka === sa && kb < sb);
        if (!(less || rhs[s0] > g[s0])) break;
        if (u < 0) break;
        expanded++;
        if (this.trace) this.trace.push(u);
        const m = Math.min(g[u], rhs[u]);
        const na = m + this._h(u) + this.km, nb = m;
        if (ka < na || (ka === na && kb < nb)) {
          this._insert(u);
        } else if (g[u] > rhs[u]) {
          g[u] = rhs[u]; this.k1[u] = NaN; this.k2[u] = NaN;
          for (let k = 0; k < 8; k++) {
            const s = G.nbr[u * 8 + k];
            if (s < 0) continue;
            if (s !== goal) { const x = G.cst[u * 8 + k] + g[u]; if (x < rhs[s]) rhs[s] = x; }
            this._update(s);
          }
        } else {
          const gOld = g[u]; g[u] = Infinity;
          for (let k = 0; k < 8; k++) {
            const s = G.nbr[u * 8 + k];
            if (s < 0) continue;
            if (s !== goal && rhs[s] === G.cst[u * 8 + k] + gOld) rhs[s] = this._minSucc(s);
            this._update(s);
          }
          if (u !== goal && rhs[u] === gOld) rhs[u] = this._minSucc(u);
          this._update(u);
        }
      }
      return { expanded, ms: performance.now() - t0 };
    }
    moveTo(s) { this.sStart = s; }
    updateVertices(nodes) {
      this.km += this.G.octile(this.sLast, this.sStart);
      this.sLast = this.sStart;
      for (const u of nodes) {
        if (u !== this.goal) this.rhs[u] = this._minSucc(u);
        this._update(u);
      }
    }
    costToGo() { return Math.min(this.g[this.sStart], this.rhs[this.sStart]); }
    path() {
      if (this.costToGo() === Infinity) return [];
      const G = this.G; let s = this.sStart; const out = [s];
      while (s !== this.goal && out.length <= G.n) {
        let best = Infinity, bv = -1;
        for (let k = 0; k < 8; k++) {
          const v = G.nbr[s * 8 + k];
          if (v < 0) continue;
          const x = G.cst[s * 8 + k] + this.g[v];
          if (x < best) { best = x; bv = v; }
        }
        if (bv < 0) return [];
        s = bv; out.push(s);
      }
      return out;
    }
  }

  // ------------------------------------------------------------------ procedural terrain (JS)
  // Same recipe as terra/terrain.py (fBm + tilt, priority-flood, D8 flow accumulation,
  // rivers + lakes, forest/rock noise, short bridges). Not bit-identical to Python's (different
  // noise interpolation), just the same kind of world.
  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function fbm(rand, h, w, octaves, base) {
    const out = new Float64Array(h * w);
    let amp = 1, total = 0;
    for (let o = 0; o < octaves; o++) {
      const n = base * (1 << o), gw = n + 2;
      const grid = new Float64Array(gw * gw);
      for (let i = 0; i < grid.length; i++) grid[i] = rand();
      for (let r = 0; r < h; r++) for (let c = 0; c < w; c++) {
        const y = r / h * n, x = c / w * n, iy = Math.floor(y), ix = Math.floor(x);
        let fy = y - iy, fx = x - ix;
        fy = fy * fy * (3 - 2 * fy); fx = fx * fx * (3 - 2 * fx);
        const a = grid[iy * gw + ix], b = grid[iy * gw + ix + 1], cc = grid[(iy + 1) * gw + ix], d = grid[(iy + 1) * gw + ix + 1];
        out[r * w + c] += amp * ((a * (1 - fx) + b * fx) * (1 - fy) + (cc * (1 - fx) + d * fx) * fy);
      }
      total += amp; amp *= 0.5;
    }
    let lo = Infinity, hi = -Infinity;
    for (const v of out) { if (v < lo) lo = v; if (v > hi) hi = v; }
    for (let i = 0; i < out.length; i++) out[i] = (out[i] - lo) / (hi - lo + 1e-12);
    return out;
  }
  function generate(seed, size = 96) {
    const rand = mulberry32(seed * 9973 + 17), h = size, w = size, n = h * w;
    let elev = fbm(rand, h, w, 5, 3);
    const ang = rand() * 2 * Math.PI;
    for (let r = 0; r < h; r++) for (let c = 0; c < w; c++) {
      const i = r * w + c;
      elev[i] = 0.75 * elev[i] + 0.25 * (Math.cos(ang) * (c / size - 0.5) + Math.sin(ang) * (r / size - 0.5) + 0.5);
    }
    let lo = Infinity, hi = -Infinity;
    for (const v of elev) { if (v < lo) lo = v; if (v > hi) hi = v; }
    for (let i = 0; i < n; i++) elev[i] = Math.pow((elev[i] - lo) / (hi - lo), 1.3);
    // priority flood
    const filled = Float64Array.from(elev), closed = new Uint8Array(n), pq = new Heap();
    for (let r = 0; r < h; r++) for (let c = 0; c < w; c++) {
      if (r === 0 || c === 0 || r === h - 1 || c === w - 1) { closed[r * w + c] = 1; pq.push(filled[r * w + c], 0, r * w + c); }
    }
    while (pq.size) {
      const z = pq.a[0], u = pq.v[0]; pq.pop();
      const r = (u / w) | 0, c = u - r * w;
      for (const [dr, dc] of DIRS) {
        const rr = r + dr, cc = c + dc;
        if (rr < 0 || rr >= h || cc < 0 || cc >= w) continue;
        const v = rr * w + cc;
        if (closed[v]) continue;
        closed[v] = 1;
        if (filled[v] <= z) filled[v] = z + 1e-6;
        pq.push(filled[v], 0, v);
      }
    }
    // D8 flow accumulation
    const recv = new Int32Array(n).fill(-1);
    for (let r = 0; r < h; r++) for (let c = 0; c < w; c++) {
      let best = 0, bv = -1;
      for (const [dr, dc] of DIRS) {
        const rr = r + dr, cc = c + dc;
        if (rr < 0 || rr >= h || cc < 0 || cc >= w) continue;
        const drop = (filled[r * w + c] - filled[rr * w + cc]) / (dr && dc ? 1.4142 : 1);
        if (drop > best) { best = drop; bv = rr * w + cc; }
      }
      recv[r * w + c] = bv;
    }
    const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => filled[b] - filled[a]);
    const acc = new Float64Array(n).fill(1);
    for (const u of order) if (recv[u] >= 0) acc[recv[u]] += acc[u];
    const thr = 0.02 * n, water = new Uint8Array(n);
    for (let u = 0; u < n; u++) if (acc[u] >= thr) {
      water[u] = 1;
      const v = recv[u];
      if (v >= 0) { const r = (u / w) | 0, vr = (v / w) | 0, vc = v - vr * w; if (vr !== r && vc !== u - r * w) water[r * w + vc] = 1; }
    }
    for (let u = 0; u < n; u++) if (acc[u] >= thr * 6) {  // widen big rivers
      const r = (u / w) | 0, c = u - r * w;
      for (const [dr, dc] of DIRS) { const rr = r + dr, cc = c + dc; if (rr >= 0 && rr < h && cc >= 0 && cc < w) water[rr * w + cc] = 1; }
    }
    for (let u = 0; u < n; u++) if (filled[u] - elev[u] > 0.045) water[u] = 1;  // lakes
    const veg = fbm(rand, h, w, 4, 5), rocky = fbm(rand, h, w, 3, 6);
    const kind = new Int8Array(n);
    for (let u = 0; u < n; u++) {
      let k = GRASS;
      if (veg[u] > 0.55 && elev[u] < 0.6) k = FOREST;
      if (elev[u] > 0.7 || (elev[u] > 0.55 && rocky[u] > 0.72)) k = ROCK;
      if (water[u]) k = WATER;
      kind[u] = k;
    }
    // bridges: short straight spans with land at both ends, kept apart
    const cands = [];
    for (let r = 2; r < h - 2; r++) for (let c = 2; c < w - 2; c++) {
      if (kind[r * w + c] !== WATER) continue;
      for (const [dr, dc] of [[0, 1], [1, 0]]) {
        if (kind[(r - dr) * w + c - dc] === WATER) continue;
        const cells = []; let rr = r, cc = c;
        while (rr < h && cc < w && kind[rr * w + cc] === WATER && cells.length <= 7) { cells.push(rr * w + cc); rr += dr; cc += dc; }
        if (rr >= h - 1 || cc >= w - 1 || kind[rr * w + cc] === WATER || cells.length > 7) continue;
        if (kind[(rr + dr) * w + cc + dc] === WATER) continue;
        cands.push(cells);
      }
    }
    for (let i = cands.length - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [cands[i], cands[j]] = [cands[j], cands[i]]; }
    const chosen = [];
    for (const cells of cands) {
      const m = cells[cells.length >> 1], mr = (m / w) | 0, mc = m - mr * w;
      if (chosen.every(b => { const q = b[b.length >> 1], qr = (q / w) | 0, qc = q - qr * w; return Math.abs(mr - qr) + Math.abs(mc - qc) > h / 4; })) chosen.push(cells);
      if (chosen.length >= 3) break;
    }
    for (const b of chosen) for (const u of b) kind[u] = BRIDGE;
    return { h, w, elev: Float32Array.from(elev), kind };
  }

  return { GRASS, FOREST, ROCK, WATER, BRIDGE, MULT, HEIGHT_SCALE, SLOPE_K, DIRS,
           Graph, Heap, astar, octileField, DStarLite, pathCost, generate };
})();
if (typeof module !== "undefined") module.exports = TERRA;
