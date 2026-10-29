// Tiny AlphaZero, JS side: Connect Four rules, the policy/value net forward pass
// (BatchNorm folded into the convs at export time) and single-game PUCT MCTS.
// Used inline by media/demo.html and by web/check_js.js (node) for verification.
"use strict";
const AZ = (() => {
  const W = 7, H = 6;

  // ---------------------------------------------------------------- rules
  class Game {
    constructor() { this.cells = new Int8Array(W * H); this.h = new Int8Array(W); this.n = 0; this.winner = 0; this.over = false; this.moves = []; }
    clone() { const g = new Game(); g.cells.set(this.cells); g.h.set(this.h); g.n = this.n; g.winner = this.winner; g.over = this.over; g.moves = this.moves.slice(); return g; }
    toMove() { return (this.n % 2) + 1; }            // 1 = first player, 2 = second
    legal(c) { return !this.over && this.h[c] < H; }
    legalMoves() { const m = []; for (let c = 0; c < W; c++) if (this.legal(c)) m.push(c); return m; }
    play(c) {                                          // returns true if this move wins
      const p = this.toMove(), r = this.h[c];
      this.cells[r * W + c] = p; this.h[c]++; this.n++; this.moves.push(c);
      if (this.winsAt(r, c, p)) { this.winner = p; this.over = true; return true; }
      if (this.n === W * H) this.over = true;
      return false;
    }
    winsAt(r, c, p) {
      const dirs = [[0, 1], [1, 0], [1, 1], [1, -1]];
      for (const [dr, dc] of dirs) {
        let k = 1;
        for (let s = 1; s < 4; s++) { const rr = r + dr * s, cc = c + dc * s; if (rr < 0 || rr >= H || cc < 0 || cc >= W || this.cells[rr * W + cc] !== p) break; k++; }
        for (let s = 1; s < 4; s++) { const rr = r - dr * s, cc = c - dc * s; if (rr < 0 || rr >= H || cc < 0 || cc >= W || this.cells[rr * W + cc] !== p) break; k++; }
        if (k >= 4) return true;
      }
      return false;
    }
    winningCells() {
      if (!this.winner) return [];
      const p = this.winner, dirs = [[0, 1], [1, 0], [1, 1], [1, -1]];
      for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) for (const [dr, dc] of dirs) {
        const cells = [];
        for (let s = 0; s < 4; s++) { const rr = r + dr * s, cc = c + dc * s; if (rr < 0 || rr >= H || cc < 0 || cc >= W || this.cells[rr * W + cc] !== p) break; cells.push([rr, cc]); }
        if (cells.length === 4) return cells;
      }
      return [];
    }
    features() {                                       // (H, W, 2), row 0 = bottom
      const x = new Float32Array(H * W * 2), me = this.toMove();
      for (let i = 0; i < H * W; i++) { const v = this.cells[i]; if (v) x[i * 2 + (v === me ? 0 : 1)] = 1; }
      return x;
    }
  }

  // ---------------------------------------------------------------- network
  function b64ToF32(s) {
    const bin = (typeof atob === "function") ? atob(s) : Buffer.from(s, "base64").toString("binary");
    const u8 = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
    return new Float32Array(u8.buffer);
  }

  // conv weights arrive as [out][kh][kw][in] (MLX layout); re-pack to [kh][kw][in][out]
  function packConv(t) {
    const [O, KH, KW, I] = t.shape, src = b64ToF32(t.data), dst = new Float32Array(src.length);
    for (let o = 0; o < O; o++) for (let a = 0; a < KH; a++) for (let b = 0; b < KW; b++) for (let i = 0; i < I; i++)
      dst[((a * KW + b) * I + i) * O + o] = src[((o * KH + a) * KW + b) * I + i];
    return { w: dst, b: b64ToF32(t.bias), O, K: KH, I };
  }
  function packLinear(t) { const [O, I] = t.shape; return { w: b64ToF32(t.data), b: b64ToF32(t.bias), O, I }; }

  function conv(x, L, relu, skip, out) {             // x: (H, W, I) -> (H, W, O), same padding
    const { w, b, O, K, I } = L, pad = (K - 1) >> 1, y = out || new Float32Array(H * W * O);
    for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) {
      const yo = (r * W + c) * O;
      for (let o = 0; o < O; o++) y[yo + o] = b[o];
      for (let a = 0; a < K; a++) {
        const rr = r + a - pad; if (rr < 0 || rr >= H) continue;
        for (let bb = 0; bb < K; bb++) {
          const cc = c + bb - pad; if (cc < 0 || cc >= W) continue;
          const xo = (rr * W + cc) * I, wo = (a * K + bb) * I * O;
          for (let i = 0; i < I; i++) {
            const xv = x[xo + i]; if (xv === 0) continue;
            const wi = wo + i * O;
            for (let o = 0; o < O; o++) y[yo + o] += xv * w[wi + o];
          }
        }
      }
    }
    if (skip) for (let i = 0; i < y.length; i++) y[i] += skip[i];
    if (relu) for (let i = 0; i < y.length; i++) if (y[i] < 0) y[i] = 0;
    return y;
  }
  function linear(x, L, relu) {
    const { w, b, O, I } = L, y = new Float32Array(O);
    for (let o = 0; o < O; o++) { let s = b[o]; const wo = o * I; for (let i = 0; i < I; i++) s += x[i] * w[wo + i]; y[o] = relu && s < 0 ? 0 : s; }
    return y;
  }

  class Net {
    constructor(spec) {
      this.stem = packConv(spec.stem);
      this.blocks = spec.blocks.map(b => [packConv(b[0]), packConv(b[1])]);
      this.pconv = packConv(spec.p_conv); this.pfc = packLinear(spec.p_fc);
      this.vconv = packConv(spec.v_conv); this.vfc1 = packLinear(spec.v_fc1); this.vfc2 = packLinear(spec.v_fc2);
    }
    forward(x) {                                       // -> {logits: Float32Array(7), value}
      if (!this.buf) { const n = H * W * this.stem.O; this.buf = [new Float32Array(n), new Float32Array(n), new Float32Array(n)]; }
      let [h, y, z] = this.buf;
      conv(x, this.stem, true, null, h);
      for (const [c1, c2] of this.blocks) { conv(h, c1, true, null, y); conv(y, c2, true, h, z); const t = h; h = z; z = t; }
      const p = conv(h, this.pconv, true);
      const logits = linear(p, this.pfc, false);
      const v = conv(h, this.vconv, true);
      const v1 = linear(v, this.vfc1, true);
      const value = Math.tanh(linear(v1, this.vfc2, false)[0]);
      return { logits, value };
    }
  }

  // ---------------------------------------------------------------- MCTS
  // Node stats are from the view of the player who moved INTO the node.
  function softmaxLegal(logits, game) {
    const out = new Float32Array(W); let mx = -Infinity, tot = 0;
    for (let c = 0; c < W; c++) if (game.legal(c) && logits[c] > mx) mx = logits[c];
    for (let c = 0; c < W; c++) if (game.legal(c)) { out[c] = Math.exp(logits[c] - mx); tot += out[c]; }
    for (let c = 0; c < W; c++) out[c] /= tot;
    return out;
  }
  function newNode(prior, move) { return { P: prior, N: 0, Wv: 0, move, children: null, term: 0 }; }

  class Search {                                     // incremental PUCT search (call run(k) repeatedly)
    constructor(net, rootGame, cPuct = 1.5) {
      this.net = net; this.game = rootGame.clone(); this.cPuct = cPuct;
      this.root = newNode(1, null);
      const r0 = this.expand(this.root, this.game);
      this.rootValue = r0.value;                       // raw value head for the player to move
      this.root.N = 1; this.root.Wv = -r0.value;       // the root expansion counts as one visit (as in mcts.py)
      this.sims = 0;
    }
    expand(node, game) {
      const { logits, value } = this.net.forward(game.features());
      const pri = softmaxLegal(logits, game);
      node.children = [];
      for (let c = 0; c < W; c++) if (game.legal(c)) node.children.push(newNode(pri[c], c));
      return { value, pri };
    }
    run(k) {
      for (let s = 0; s < k; s++) {
        const g = this.game.clone(); const path = [this.root]; let node = this.root;
        while (node.children && node.children.length) {
          const sq = Math.sqrt(node.N);
          let best = null, bs = -Infinity;
          for (const ch of node.children) {
            const q = ch.N > 0 ? ch.Wv / ch.N : 0;
            const u = q + this.cPuct * ch.P * sq / (1 + ch.N);
            if (u > bs) { bs = u; best = ch; }
          }
          node = best; path.push(node);
          if (node.term) break;
          const won = g.play(node.move);
          if (won) { node.term = 1; break; }
          if (g.over) { node.term = 2; break; }
        }
        let val;                                       // from the view of the mover into `node`
        if (node.term === 1) val = 1; else if (node.term === 2) val = 0;
        else val = -this.expand(node, g).value;
        for (let i = path.length - 1; i >= 0; i--) { path[i].N += 1; path[i].Wv += val; val = -val; }
        this.sims++;
      }
      return this.result();
    }
    result() {
      const visits = new Float32Array(W), q = new Float32Array(W).fill(NaN), prior = new Float32Array(W);
      for (const ch of this.root.children) { visits[ch.move] = ch.N; q[ch.move] = ch.N ? ch.Wv / ch.N : NaN; prior[ch.move] = ch.P; }
      return { visits, q, prior, rootValue: this.rootValue, rootQ: -this.root.Wv / this.root.N, sims: this.sims };
    }
  }
  function search(net, rootGame, sims, cPuct = 1.5) { return new Search(net, rootGame, cPuct).run(sims); }

  function bestMove(res) {
    let best = -1, bv = -1;
    for (let c = 0; c < W; c++) if (res.visits[c] > bv || (res.visits[c] === bv && res.prior[c] > res.prior[best])) { bv = res.visits[c]; best = c; }
    return best;
  }

  return { W, H, Game, Net, Search, search, bestMove, softmaxLegal };
})();
if (typeof module !== "undefined") module.exports = AZ;
