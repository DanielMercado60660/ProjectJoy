// HNet forward pass in plain JS: the MLX-trained U-Net from terra/learn.py, so the demo can
// run "A* with the learned heuristic" on whatever map you paint. Weights are baked by
// build_demo.py as float16. tests/test_js_parity.py compares this against MLX in node.
const HNET = (() => {
  let P = null, base = 32;

  function f16(u) {  // IEEE half -> float
    const s = (u & 0x8000) ? -1 : 1, e = (u >> 10) & 0x1f, f = u & 0x3ff;
    if (e === 0) return s * Math.pow(2, -14) * (f / 1024);
    if (e === 31) return f ? NaN : s * Infinity;
    return s * Math.pow(2, e - 15) * (1 + f / 1024);
  }
  function b64bytes(s) {
    if (typeof atob === "function") { const bin = atob(s); const o = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) o[i] = bin.charCodeAt(i); return o; }
    return new Uint8Array(Buffer.from(s, "base64"));
  }
  function load(spec) {
    const bytes = b64bytes(spec.data), u16 = new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.length / 2);
    P = {};
    for (const [name, shape, off] of spec.index) {
      const n = shape.reduce((a, b) => a * b, 1), arr = new Float32Array(n);
      for (let i = 0; i < n; i++) arr[i] = f16(u16[off + i]);
      P[name] = { shape, data: arr };
    }
    base = spec.base;
    // conv weights (out, kh, kw, in) -> (kh, kw, in, out) for a cache-friendly inner loop
    for (const name of Object.keys(P)) {
      const p = P[name];
      if (p.shape.length === 4) {
        const [co, kh, kw, ci] = p.shape, t = new Float32Array(p.data.length);
        for (let o = 0; o < co; o++) for (let y = 0; y < kh; y++) for (let x = 0; x < kw; x++) for (let i = 0; i < ci; i++)
          t[((y * kw + x) * ci + i) * co + o] = p.data[((o * kh + y) * kw + x) * ci + i];
        p.t = t;
      }
    }
  }
  function conv(x, H, W, Ci, name) {
    const p = P[name + ".weight"], b = P[name + ".bias"].data, [Co, K] = p.shape, t = p.t, pad = K >> 1;
    const y = new Float32Array(H * W * Co);
    for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) {
      const o = (r * W + c) * Co;
      for (let k = 0; k < Co; k++) y[o + k] = b[k];
      for (let ky = 0; ky < K; ky++) {
        const rr = r + ky - pad; if (rr < 0 || rr >= H) continue;
        for (let kx = 0; kx < K; kx++) {
          const cc = c + kx - pad; if (cc < 0 || cc >= W) continue;
          const xi = (rr * W + cc) * Ci, wb = (ky * K + kx) * Ci * Co;
          for (let i = 0; i < Ci; i++) {
            const v = x[xi + i]; if (v === 0) continue;
            const wo = wb + i * Co;
            for (let k = 0; k < Co; k++) y[o + k] += v * t[wo + k];
          }
        }
      }
    }
    return y;
  }
  function gnSilu(x, H, W, C, name, groups = 8) {
    const g = P[name + ".weight"].data, b = P[name + ".bias"].data, cg = C / groups, n = H * W;
    for (let gi = 0; gi < groups; gi++) {
      let s = 0, s2 = 0;
      for (let p = 0; p < n; p++) for (let k = gi * cg; k < (gi + 1) * cg; k++) { const v = x[p * C + k]; s += v; s2 += v * v; }
      const cnt = n * cg, mean = s / cnt, varr = Math.max(0, s2 / cnt - mean * mean), inv = 1 / Math.sqrt(varr + 1e-5);
      for (let p = 0; p < n; p++) for (let k = gi * cg; k < (gi + 1) * cg; k++) {
        const v = (x[p * C + k] - mean) * inv * g[k] + b[k];
        x[p * C + k] = v / (1 + Math.exp(-v));
      }
    }
    return x;
  }
  function block(x, H, W, Ci, Co, name) {
    let y = conv(x, H, W, Ci, name + ".c1"); gnSilu(y, H, W, Co, name + ".n1");
    y = conv(y, H, W, Co, name + ".c2"); gnSilu(y, H, W, Co, name + ".n2");
    return y;
  }
  function pool(x, H, W, C) {
    const h = H >> 1, w = W >> 1, y = new Float32Array(h * w * C);
    for (let r = 0; r < h; r++) for (let c = 0; c < w; c++) for (let k = 0; k < C; k++) {
      const a = x[((2 * r) * W + 2 * c) * C + k], b = x[((2 * r) * W + 2 * c + 1) * C + k],
            d = x[((2 * r + 1) * W + 2 * c) * C + k], e = x[((2 * r + 1) * W + 2 * c + 1) * C + k];
      y[(r * w + c) * C + k] = Math.max(a, b, d, e);
    }
    return y;
  }
  function upcat(x, H, W, C, skip, Cs) {  // nearest x2 upsample, then concat [x, skip]
    const H2 = H * 2, W2 = W * 2, Ct = C + Cs, y = new Float32Array(H2 * W2 * Ct);
    for (let r = 0; r < H2; r++) for (let c = 0; c < W2; c++) {
      const o = (r * W2 + c) * Ct, xi = ((r >> 1) * W + (c >> 1)) * C, si = (r * W2 + c) * Cs;
      for (let k = 0; k < C; k++) y[o + k] = x[xi + k];
      for (let k = 0; k < Cs; k++) y[o + C + k] = skip[si + k];
    }
    return y;
  }
  // x: Float32Array H*W*8 (NHWC) -> Float32Array H*W (log-ratio r)
  function forward(x, H, W, Cin) {
    const ch = [base, base * 2, base * 4, base * 4, base * 4];
    const skips = [], dims = [];
    let h = H, w = W, c = Cin;
    for (let i = 0; i < 5; i++) {
      if (i > 0) { x = pool(x, h, w, c); h >>= 1; w >>= 1; }
      x = block(x, h, w, c, ch[i], `enc.${i}`); c = ch[i];
      skips.push(x); dims.push([h, w]);
    }
    // global context
    const gw = P["glob.weight"].data, gb = P["glob.bias"].data, mean = new Float32Array(c);
    for (let p = 0; p < h * w; p++) for (let k = 0; k < c; k++) mean[k] += x[p * c + k] / (h * w);
    const gv = new Float32Array(c);
    for (let o = 0; o < c; o++) { let s = gb[o]; for (let k = 0; k < c; k++) s += gw[o * c + k] * mean[k]; gv[o] = s; }
    for (let p = 0; p < h * w; p++) for (let k = 0; k < c; k++) x[p * c + k] += gv[k];
    for (let i = 3; i >= 0; i--) {
      x = upcat(x, h, w, c, skips[i], ch[i]); h *= 2; w *= 2;
      x = block(x, h, w, c + ch[i], ch[i], `dec.${i}`); c = ch[i];
    }
    const ow = P["out.weight"].data, ob = P["out.bias"].data[0], out = new Float32Array(h * w);
    for (let p = 0; p < h * w; p++) { let s = ob; for (let k = 0; k < c; k++) s += ow[k] * x[p * c + k]; out[p] = s; }
    return out;
  }
  // same channels as terra/learn.py static_features + goal_features
  function features(G, goal) {
    const H = G.h, W = G.w, C = 8, x = new Float32Array(H * W * C), z = i => G.elev[i] * 24;
    const gr = (goal / W) | 0, gc = goal - gr * W;
    for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) {
      const i = r * W + c;
      const gx = c === 0 ? z(i + 1) - z(i) : c === W - 1 ? z(i) - z(i - 1) : (z(i + 1) - z(i - 1)) / 2;
      const gy = r === 0 ? z(i + W) - z(i) : r === H - 1 ? z(i) - z(i - W) : (z(i + W) - z(i - W)) / 2;
      const k = G.kind[i], m = TERRA.MULT[k];
      const dr = Math.abs(r - gr), dc = Math.abs(c - gc), oct = Math.max(dr, dc) + (Math.SQRT2 - 1) * Math.min(dr, dc);
      const o = i * C;
      x[o] = G.elev[i]; x[o + 1] = Math.tanh(Math.sqrt(gx * gx + gy * gy) / 2);
      x[o + 2] = m === Infinity ? 0 : (m - 1) / 2; x[o + 3] = k === TERRA.WATER ? 1 : 0;
      x[o + 4] = k === TERRA.BRIDGE ? 1 : 0; x[o + 5] = k === TERRA.FOREST ? 1 : 0;
      x[o + 6] = oct / H; x[o + 7] = Math.exp(-(oct * oct) / 8);
    }
    return x;
  }
  // learned heuristic field: octile * exp(max(r, 0))
  function heuristic(G, goal) {
    const r = forward(features(G, goal), G.h, G.w, 8), h = new Float64Array(G.n);
    for (let u = 0; u < G.n; u++) h[u] = G.octile(u, goal) * Math.exp(Math.max(0, r[u]));
    return { h, r };
  }
  return { load, forward, features, heuristic, get loaded() { return P !== null; } };
})();
if (typeof module !== "undefined") module.exports = HNET;
