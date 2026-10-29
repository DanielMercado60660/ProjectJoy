// Core dungeon logic for demo.html (also loaded by tests/test_web_parity.py through node).
// Mirrors dungeon/mazes.py (BFS), dungeon/env.py (observation) and dungeon/policy.py (GRU policy).
const S = 13, VIEW = 5, RAD = 2, T_MAX = 200;
const DR = [-1, 0, 1, 0], DC = [0, 1, 0, -1];

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function bfsDist(walls, r0, c0) {
  const d = new Int32Array(S * S).fill(-1);
  if (walls[r0 * S + c0]) return d;
  const q = [r0 * S + c0]; d[q[0]] = 0;
  for (let h = 0; h < q.length; h++) {
    const r = Math.floor(q[h] / S), c = q[h] % S;
    for (let k = 0; k < 4; k++) {
      const nr = r + DR[k], nc = c + DC[k];
      if (nr < 0 || nc < 0 || nr >= S || nc >= S) continue;
      const j = nr * S + nc;
      if (!walls[j] && d[j] < 0) { d[j] = d[q[h]] + 1; q.push(j); }
    }
  }
  return d;
}

function bfsPath(walls, s, g) {
  const d = bfsDist(walls, g[0], g[1]);
  if (d[s[0] * S + s[1]] < 0) return null;
  const p = [[s[0], s[1]]]; let r = s[0], c = s[1];
  while (!(r === g[0] && c === g[1])) {
    for (let k = 0; k < 4; k++) {
      const nr = r + DR[k], nc = c + DC[k];
      if (nr >= 0 && nc >= 0 && nr < S && nc < S && !walls[nr * S + nc] && d[nr * S + nc] === d[r * S + c] - 1) {
        r = nr; c = nc; break;
      }
    }
    p.push([r, c]);
  }
  return p;
}

// ------------------------------------------------------------------ learned policy
function decodeF32(b64) {
  let bin;
  if (typeof atob === 'function') bin = atob(b64); else bin = Buffer.from(b64, 'base64').toString('binary');
  const u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Float32Array(u8.buffer);
}

function makePolicy(spec) {
  const W = {};
  for (const [k, v] of Object.entries(spec.params)) W[k] = { shape: v.shape, data: decodeF32(v.b64) };
  const H = spec.hidden;
  function linear(name, x) { // y = x W^T + b, W (out, in)
    const w = W[name + '.weight'], b = W[name + '.bias'];
    const [o, i] = w.shape, y = new Float32Array(o);
    for (let r = 0; r < o; r++) {
      let s = b.data[r]; const off = r * i;
      for (let j = 0; j < i; j++) s += w.data[off + j] * x[j];
      y[r] = s;
    }
    return y;
  }
  const relu = (x) => x.map((v) => (v > 0 ? v : 0));
  const sig = (v) => 1 / (1 + Math.exp(-v));
  function step(obs, h) {
    if (spec.memoryless) h = new Float32Array(H); // ablation: no recurrent state
    const x = relu(linear('enc2', relu(linear('enc1', obs))));
    const gx = linear('gru.x2h', x), gh = linear('gru.h2h', h);
    const hn = new Float32Array(H);
    for (let j = 0; j < H; j++) {
      const r = sig(gx[j] + gh[j]), z = sig(gx[H + j] + gh[H + j]);
      const n = Math.tanh(gx[2 * H + j] + r * gh[2 * H + j]);
      hn[j] = (1 - z) * n + z * h[j];
    }
    const logits = linear('pi2', linear('pi1', hn).map(Math.tanh));
    return { logits, h: hn };
  }
  return { step, hidden: H };
}

function observe(walls, pos, goal, prevA, bumped) {
  const o = new Float32Array(2 * VIEW * VIEW + 5);
  for (let i = 0; i < VIEW; i++) for (let j = 0; j < VIEW; j++) {
    const r = pos[0] + i - RAD, c = pos[1] + j - RAD;
    o[i * VIEW + j] = (r < 0 || c < 0 || r >= S || c >= S) ? 1 : walls[r * S + c];
  }
  const gr = goal[0] - pos[0] + RAD, gc = goal[1] - pos[1] + RAD;
  if (gr >= 0 && gc >= 0 && gr < VIEW && gc < VIEW) o[VIEW * VIEW + gr * VIEW + gc] = 1;
  if (prevA >= 0) o[2 * VIEW * VIEW + prevA] = 1;
  o[2 * VIEW * VIEW + 4] = bumped ? 1 : 0;
  return o;
}

function runPolicy(policy, walls, s, g, seed, greedy) {
  const rnd = mulberry32(seed);
  let pos = [s[0], s[1]], prevA = -1, bumped = false, h = new Float32Array(policy.hidden);
  const traj = [[pos[0], pos[1]]];
  for (let t = 1; t <= T_MAX; t++) {
    const out = policy.step(observe(walls, pos, g, prevA, bumped), h); h = out.h;
    let a = 0;
    if (greedy) { for (let k = 1; k < 4; k++) if (out.logits[k] > out.logits[a]) a = k; }
    else {
      const m = Math.max(...out.logits), p = Array.from(out.logits, (v) => Math.exp(v - m));
      const z = p.reduce((x, y) => x + y, 0); let u = rnd() * z;
      for (a = 0; a < 3; a++) { u -= p[a]; if (u <= 0) break; }
    }
    const nr = pos[0] + DR[a], nc = pos[1] + DC[a];
    bumped = !!walls[nr * S + nc];
    if (!bumped) pos = [nr, nc];
    prevA = a; traj.push([pos[0], pos[1]]);
    if (pos[0] === g[0] && pos[1] === g[1]) return { success: true, steps: t, traj };
  }
  return { success: false, steps: T_MAX, traj };
}

// ------------------------------------------------------------------ scripted solvers (mirror scripted.py)
function runScripted(algo, walls, s, g, seed) {
  const rnd = mulberry32(seed), ri = (n) => Math.floor(rnd() * n);
  const known = new Uint8Array(S * S), visited = new Uint8Array(S * S), stack = [];
  let r = s[0], c = s[1], heading = ri(4), goalSeen = false;
  const traj = [[r, c]]; visited[r * S + c] = 1;
  const reveal = () => {
    for (let i = r - RAD; i <= r + RAD; i++) for (let j = c - RAD; j <= c + RAD; j++)
      if (i >= 0 && j >= 0 && i < S && j < S) known[i * S + j] = 1 + walls[i * S + j];
  };
  const towards = () => {
    const kw = new Uint8Array(S * S).fill(1);
    for (let i = 0; i < S * S; i++) if (known[i] === 1) kw[i] = 0;
    const d = bfsDist(kw, g[0], g[1]);
    if (d[r * S + c] < 0) return -1;
    for (let k = 0; k < 4; k++) {
      const nr = r + DR[k], nc = c + DC[k];
      if (!kw[nr * S + nc] && d[nr * S + nc] === d[r * S + c] - 1) return k;
    }
    return -1;
  };
  for (let t = 1; t <= T_MAX; t++) {
    reveal();
    if (Math.abs(g[0] - r) <= RAD && Math.abs(g[1] - c) <= RAD) goalSeen = true;
    let a = goalSeen ? towards() : -1;
    if (a < 0) {
      if (algo === 'random_walk') {
        const legal = [0, 1, 2, 3].filter((k) => !walls[(r + DR[k]) * S + c + DC[k]]);
        a = legal.length ? legal[ri(legal.length)] : ri(4);
      } else if (algo === 'wall_follow') {
        for (const turn of [3, 0, 1, 2]) {
          const k = (heading + turn) % 4;
          if (!walls[(r + DR[k]) * S + c + DC[k]]) { a = k; break; }
        }
        if (a < 0) a = heading;
      } else { // tremaux
        const order = [0, 1, 2, 3];
        for (let i = 3; i > 0; i--) { const j = ri(i + 1); [order[i], order[j]] = [order[j], order[i]]; }
        for (const k of order) {
          const nr = r + DR[k], nc = c + DC[k];
          if (!walls[nr * S + nc] && !visited[nr * S + nc]) { a = k; stack.push([r, c]); break; }
        }
        if (a < 0) {
          if (stack.length) {
            const [pr, pc] = stack.pop();
            for (let k = 0; k < 4; k++) if (r + DR[k] === pr && c + DC[k] === pc) a = k;
          } else a = ri(4);
        }
      }
    }
    const nr = r + DR[a], nc = c + DC[a];
    if (!walls[nr * S + nc]) { r = nr; c = nc; heading = a; }
    visited[r * S + c] = 1; traj.push([r, c]);
    if (r === g[0] && c === g[1]) return { success: true, steps: t, traj };
  }
  return { success: false, steps: T_MAX, traj };
}

if (typeof module !== 'undefined') module.exports = { S, bfsDist, bfsPath, makePolicy, runPolicy, runScripted, observe, mulberry32 };
