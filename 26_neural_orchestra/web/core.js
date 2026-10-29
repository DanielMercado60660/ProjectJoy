// Neural Orchestra core: pure functions shared by demo.html and the node test (tests/test_js_decoder.mjs).
// - VAE decoder forward pass (weights baked by make_demo.py, int8 + per-row scale)
// - drum one-shot synthesis mirroring synth.py (sine sweeps, filtered noise, exponential envelopes)
const NO = (() => {
  const STEPS = 32, NI = 9, CELLS = STEPS * NI;

  function b64bytes(b64) {
    if (typeof atob === "function") {
      const s = atob(b64), out = new Uint8Array(s.length);
      for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i);
      return out;
    }
    return new Uint8Array(Buffer.from(b64, "base64"));
  }

  // W = {latent, n_styles, layers: [{n_in, n_out, q (b64 int8, row-major [out][in]), scale: [out], bias: [out]}]}
  function loadDecoder(W) {
    const layers = W.layers.map((L) => {
      const q = new Int8Array(b64bytes(L.q).buffer);
      const w = new Float32Array(L.n_in * L.n_out);
      for (let o = 0; o < L.n_out; o++) {
        const sc = L.scale[o];
        for (let i = 0; i < L.n_in; i++) w[o * L.n_in + i] = q[o * L.n_in + i] * sc;
      }
      return { n_in: L.n_in, n_out: L.n_out, w, b: Float32Array.from(L.bias) };
    });
    return { latent: W.latent, n_styles: W.n_styles, layers };
  }

  // erf via Abramowitz & Stegun 7.1.26 (|err| < 1.5e-7); MLX nn.gelu is the exact erf form.
  function erf(x) {
    const s = Math.sign(x); x = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * x);
    const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return s * y;
  }
  const gelu = (x) => 0.5 * x * (1 + erf(x / Math.SQRT2));

  function dense(L, x, act) {
    const y = new Float32Array(L.n_out);
    for (let o = 0; o < L.n_out; o++) {
      let acc = L.b[o];
      const row = o * L.n_in;
      for (let i = 0; i < L.n_in; i++) acc += L.w[row + i] * x[i];
      y[o] = act ? gelu(acc) : acc;
    }
    return y;
  }

  // z: latent array, style: soft style vector (length n_styles), T: temperature on hit logits.
  // Returns hit probabilities, velocity, offset (each 288).
  function decode(dec, z, style, T = 1) {
    let h = Float32Array.from([...z, ...style]);
    const n = dec.layers.length;
    dec.layers.forEach((L, k) => { h = dense(L, h, k < n - 1); });
    const p = new Float32Array(CELLS), v = new Float32Array(CELLS), off = new Float32Array(CELLS);
    for (let c = 0; c < CELLS; c++) {
      p[c] = 1 / (1 + Math.exp(-h[c] / T));
      v[c] = 1 / (1 + Math.exp(-h[CELLS + c]));
      off[c] = 0.5 * Math.tanh(h[2 * CELLS + c]);
    }
    return { p, v, off };
  }

  // Autoregressive GRU decoder (ARVAE): one 9-drum column per 16th step, fed back as the next input.
  // rand() is called once per cell in step-major order, like toPattern, so a frozen coin set gives smooth morphs.
  function loadAR(W) {
    const deq = (L) => loadDecoder({ latent: 0, n_styles: 0, layers: [L] }).layers[0];
    return { kind: "ar", latent: W.latent, n_styles: W.n_styles, H: W.rnn, init: deq(W.init_h), Wx: deq(W.Wx), Wh: deq(W.Wh),
             bhn: Float32Array.from(W.bhn), out: deq(W.out) };
  }
  const sig = (x) => 1 / (1 + Math.exp(-x));
  function generateAR(dec, z, style, mode, T, rand) {
    const H = dec.H, zc = [...z, ...style];
    let h = dense(dec.init, zc, false).map(Math.tanh);
    let prev = new Float32Array(3 * NI);
    const pat = { h: new Uint8Array(CELLS), v: new Float32Array(CELLS), o: new Float32Array(CELLS) }, P = new Float32Array(CELLS);
    for (let t = 0; t < STEPS; t++) {
      const xp = dense(dec.Wx, Float32Array.from([...prev, ...zc]), false); // includes b
      const hp = dense(dec.Wh, h, false);
      const hn = new Float32Array(H);
      for (let j = 0; j < H; j++) {
        const r = sig(xp[j] + hp[j]), u = sig(xp[H + j] + hp[H + j]);
        const n = Math.tanh(xp[2 * H + j] + r * (hp[2 * H + j] + dec.bhn[j]));
        hn[j] = (1 - u) * n + u * h[j];
      }
      h = hn;
      const o = dense(dec.out, h, false);
      prev = new Float32Array(3 * NI);
      for (let i = 0; i < NI; i++) {
        const c = t * NI + i, p = sig(o[i] / T);
        P[c] = p;
        const hit = mode === "sample" ? rand() < p : p > 0.5;
        if (mode !== "sample") rand();
        if (hit) {
          pat.h[c] = 1; pat.v[c] = sig(o[NI + i]); pat.o[c] = 0.5 * Math.tanh(o[2 * NI + i]);
          prev[i] = 1; prev[NI + i] = pat.v[c]; prev[2 * NI + i] = 2 * pat.o[c];
        }
      }
    }
    return { pat, p: P };
  }

  // Turn decoder output into a pattern {h, v, o} (Uint8/Float32, index = step*9 + inst).
  function toPattern(out, mode, rand) {
    const h = new Uint8Array(CELLS), v = new Float32Array(CELLS), o = new Float32Array(CELLS);
    for (let c = 0; c < CELLS; c++) {
      const hit = mode === "sample" ? (rand() < out.p[c]) : out.p[c] > 0.5;
      if (hit) { h[c] = 1; v[c] = out.v[c]; o[c] = out.off[c]; }
    }
    return { h, v, o };
  }

  // Baked patterns are lists of [step, inst, vel*100, off*100].
  function unpack(hits) {
    const p = { h: new Uint8Array(CELLS), v: new Float32Array(CELLS), o: new Float32Array(CELLS) };
    for (const [s, i, vv, oo] of hits) { const c = s * NI + i; p.h[c] = 1; p.v[c] = vv / 100; p.o[c] = oo / 100; }
    return p;
  }

  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gaussian(rand) {
    const u = Math.max(rand(), 1e-12), v = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  // ---- drum synthesis (mirrors synth.py; RBJ biquads stand in for scipy Butterworth) ----
  function biquad(x, sr, type, f0, Q = Math.SQRT1_2) {
    const w0 = 2 * Math.PI * f0 / sr, a = Math.sin(w0) / (2 * Q), c = Math.cos(w0);
    let b0, b1, b2;
    if (type === "high") { b0 = (1 + c) / 2; b1 = -(1 + c); b2 = (1 + c) / 2; }
    else { b0 = a; b1 = 0; b2 = -a; } // band-pass, 0 dB peak
    const a0 = 1 + a, a1 = -2 * c, a2 = 1 - a;
    const y = new Float32Array(x.length);
    let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
    for (let n = 0; n < x.length; n++) {
      const yn = (b0 * x[n] + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2) / a0;
      x2 = x1; x1 = x[n]; y2 = y1; y1 = yn; y[n] = yn;
    }
    return y;
  }
  const env = (n, sr, d) => Float32Array.from({ length: n }, (_, i) => Math.exp(-i / sr / d));
  function noise(n, rand) { return Float32Array.from({ length: n }, () => rand() * 2 - 1); }
  function sweep(n, sr, f0, f1, k) {
    const y = new Float32Array(n); let ph = 0;
    for (let i = 0; i < n; i++) { const f = f1 + (f0 - f1) * Math.exp(-i / sr * k); ph += 2 * Math.PI * f / sr; y[i] = Math.sin(ph); }
    return y;
  }
  const mul = (a, b, g = 1) => a.map((x, i) => x * b[i] * g);
  const add = (a, b) => a.map((x, i) => x + b[i]);

  function makeKit(sr) {
    const rand = mulberry32(7), kit = [];
    let n = Math.floor(0.45 * sr);
    const kick = mul(sweep(n, sr, 160, 45, 30), env(n, sr, 0.18));
    const click = mul(noise(200, rand), env(200, sr, 0.002), 0.3);
    for (let i = 0; i < 200; i++) kick[i] += click[i];
    kit.push(kick);
    n = Math.floor(0.25 * sr);
    kit.push(add(mul(biquad(noise(n, rand), sr, "high", 1500), env(n, sr, 0.07), 0.6), mul(sweep(n, sr, 260, 180, 40), env(n, sr, 0.05), 0.5)));
    n = Math.floor(0.3 * sr);
    const burst = new Float32Array(n);
    for (const d of [0, 0.010, 0.021]) { const i0 = Math.floor(d * sr), e = env(n - i0, sr, d < 0.02 ? 0.006 : 0.09); for (let i = 0; i < e.length; i++) burst[i0 + i] += e[i]; }
    kit.push(mul(biquad(noise(n, rand), sr, "band", 1400, 0.9), burst, 2.0));
    n = Math.floor(0.08 * sr);
    kit.push(mul(biquad(noise(n, rand), sr, "high", 7000), env(n, sr, 0.018)));
    n = Math.floor(0.5 * sr);
    kit.push(mul(biquad(noise(n, rand), sr, "high", 6000), env(n, sr, 0.16)));
    kit.push(mul(sweep(n, sr, 160, 95, 12), env(n, sr, 0.22)));
    kit.push(mul(sweep(n, sr, 240, 160, 12), env(n, sr, 0.18)));
    n = Math.floor(1.2 * sr);
    const metal = new Float32Array(n);
    for (const f of [523, 811, 1077, 1463, 1879]) for (let i = 0; i < n; i++) metal[i] += Math.sign(Math.sin(2 * Math.PI * f * i / sr)) / 5;
    kit.push(add(mul(biquad(metal, sr, "high", 3000), env(n, sr, 0.45), 0.35), mul(biquad(noise(n, rand), sr, "high", 8000), env(n, sr, 0.3), 0.1)));
    n = Math.floor(2.0 * sr);
    kit.push(mul(biquad(noise(n, rand), sr, "high", 4000), env(n, sr, 0.7), 0.6));
    return kit;
  }
  const GAIN = [1.0, 0.8, 0.7, 0.35, 0.3, 0.7, 0.7, 0.35, 0.35];

  return { STEPS, NI, CELLS, loadDecoder, loadAR, generateAR, decode, toPattern, unpack, mulberry32, gaussian, makeKit, GAIN, erf };
})();
if (typeof module !== "undefined") module.exports = NO;
