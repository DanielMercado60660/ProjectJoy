// Tiny Grand Prix simulator + controllers in plain JS (a port of racing/sim.py,
// racing/policy.py and racing/baselines.py RayFollower). Used by demo.html and by
// tests/test_js_parity.py (through node) to check it matches the Python sim.
"use strict";
const SIM = {
  DT: 0.1, WHEELBASE: 2.6, MAX_STEER: 0.45, A_MAX: 6.0, B_MAX: 12.0, G: 9.81,
  CAR_R: 1.0, RAY_LEN: 60.0, V_SCALE: 30.0,
  RAY_ANGLES: [-90, -60, -35, -18, -6, 6, 18, 35, 60, 90].map(d => d * Math.PI / 180),
};
SIM.DRAG = SIM.A_MAX / (40.0 * 40.0);

function prepTrack(t) {
  // t: {center: [[x,y]..], tangent: [[x,y]..], hw, length}
  const n = t.center.length;
  const left = [], right = [];
  for (let i = 0; i < n; i++) {
    const [cx, cy] = t.center[i], [tx, ty] = t.tangent[i];
    left.push([cx - ty * t.hw, cy + tx * t.hw]);
    right.push([cx + ty * t.hw, cy - tx * t.hw]);
  }
  const segs = [];
  for (const w of [left, right]) for (let i = 0; i < n; i++) segs.push([w[i][0], w[i][1], w[(i + 1) % n][0], w[(i + 1) % n][1]]);
  return Object.assign({}, t, { n, left, right, segs, seglen: t.length / n });
}

function castRay(track, px, py, ang) {
  const dx = Math.cos(ang), dy = Math.sin(ang), maxlen = SIM.RAY_LEN;
  let best = maxlen;
  const lim = (maxlen + 4.0) ** 2;
  for (const s of track.segs) {
    const ox = s[0] - px, oy = s[1] - py;
    if (ox * ox + oy * oy > lim) continue;
    const ex = s[2] - s[0], ey = s[3] - s[1];
    const den = dx * ey - dy * ex;
    if (Math.abs(den) < 1e-12) continue;
    const t = (ox * ey - oy * ex) / den, u = (ox * dy - oy * dx) / den;
    if (t >= 0 && u >= 0 && u <= 1 && t < best) best = t;
  }
  return best;
}

function newCar(track, grip) {
  const [x, y] = track.center[0], [tx, ty] = track.tangent[0];
  return { x, y, yaw: Math.atan2(ty, tx), v: 0, steerPrev: 0, pedalPrev: 0, idx: 0, sFrac: 0, prog: 0,
           lat: 0, done: false, crashed: false, finished: false, lapTime: null, t: 0, grip, collisions: 0 };
}

function observe(track, c) {
  const o = SIM.RAY_ANGLES.map(a => castRay(track, c.x, c.y, c.yaw + a) / SIM.RAY_LEN);
  o.push(c.v / SIM.V_SCALE, c.steerPrev, c.pedalPrev);
  return o;
}

function locate(track, px, py, idx0) {
  let bestD = 1e18, bi = idx0, bt = 0, blat = 0;
  const n = track.n, C = track.center;
  for (let k = -4; k <= 6; k++) {
    const i = ((idx0 + k) % n + n) % n, j = (i + 1) % n;
    const ax = C[i][0], ay = C[i][1], dx = C[j][0] - ax, dy = C[j][1] - ay;
    const l2 = dx * dx + dy * dy;
    let t = ((px - ax) * dx + (py - ay) * dy) / l2;
    t = Math.min(1, Math.max(0, t));
    const qx = ax + t * dx - px, qy = ay + t * dy - py, d = qx * qx + qy * qy;
    if (d < bestD) { bestD = d; bi = i; bt = t; blat = (dx * (py - ay) - dy * (px - ax)) / Math.sqrt(l2); }
  }
  return [bi, bt, blat];
}

function stepCar(track, c, steer, pedal) {
  if (c.done) return;
  const st = Math.min(1, Math.max(-1, steer)), pd = Math.min(1, Math.max(-1, pedal));
  c.steerPrev = st; c.pedalPrev = pd;
  const lim = c.grip * SIM.G;
  let a = pd > 0 ? pd * SIM.A_MAX : pd * SIM.B_MAX;
  a = Math.min(lim, Math.max(-lim, a));
  a -= SIM.DRAG * c.v * c.v;
  let vn = Math.max(0, c.v + a * SIM.DT);
  const vm = 0.5 * (c.v + vn);
  let om = vm * Math.tan(st * SIM.MAX_STEER) / SIM.WHEELBASE;
  const wl = lim / Math.max(vm, 1.0);
  om = Math.min(wl, Math.max(-wl, om));
  c.yaw += om * SIM.DT;
  c.x += vm * Math.cos(c.yaw) * SIM.DT;
  c.y += vm * Math.sin(c.yaw) * SIM.DT;
  c.v = vn;
  const [i, t, la] = locate(track, c.x, c.y, c.idx);
  let ds = (i + t) * track.seglen - (c.idx + c.sFrac) * track.seglen;
  const L = track.length;
  if (ds > 0.5 * L) ds -= L; else if (ds < -0.5 * L) ds += L;
  c.idx = i; c.sFrac = t; c.lat = la; c.prog += ds; c.t += 1;
  if (Math.abs(la) > track.hw - SIM.CAR_R) { c.crashed = true; c.done = true; }
  if (!c.done && c.prog >= L) { c.finished = true; c.done = true; c.lapTime = c.t * SIM.DT; }
}

function respawn(track, c) {  // put a crashed car back on the centerline, at rest
  const i = c.idx;
  c.x = track.center[i][0]; c.y = track.center[i][1];
  c.yaw = Math.atan2(track.tangent[i][1], track.tangent[i][0]);
  c.v = 0; c.sFrac = 0; c.lat = 0; c.done = false; c.crashed = false; c.collisions += 1;
}

function mlpForward(theta, obs) {
  // 13 -> 32 -> 32 -> 2, tanh everywhere; theta = [W0 (13x32 row-major), b0, W1, b1, W2, b2]
  const dims = [obs.length, 32, 32, 2];
  let h = obs, o = 0;
  for (let l = 0; l < 3; l++) {
    const nin = dims[l], nout = dims[l + 1], out = new Array(nout).fill(0);
    for (let i = 0; i < nin; i++) { const hi = h[i]; const base = o + i * nout; for (let j = 0; j < nout; j++) out[j] += hi * theta[base + j]; }
    o += nin * nout;
    for (let j = 0; j < nout; j++) out[j] = Math.tanh(out[j] + theta[o + j]);
    o += nout;
    h = out;
  }
  return h;
}

function rayFollower(p, obs) {
  const R = SIM.RAY_ANGLES.length;
  const r = obs.slice(0, R).map(x => Math.min(1, Math.max(1e-3, x)));
  const v = obs[R] * SIM.V_SCALE;
  let ws = 0, wa = 0;
  for (let k = 0; k < R; k++) { const w = Math.pow(r[k], p.p); ws += w; wa += w * SIM.RAY_ANGLES[k]; }
  const steer = p.k_steer * (wa / ws) / SIM.MAX_STEER + p.k_center * (r[R - 1] - r[0]);
  let front = 0;
  for (let k = 0; k < R; k++) if (Math.abs(SIM.RAY_ANGLES[k]) < 20 * Math.PI / 180) front = Math.max(front, r[k]);
  front *= SIM.RAY_LEN;
  let vt = Math.sqrt(2 * p.brake * Math.max(front - p.margin, 0));
  vt = Math.min(p.v_max, Math.max(p.v_min, vt));
  return [Math.min(1, Math.max(-1, steer)), Math.min(1, Math.max(-1, 0.5 * (vt - v)))];
}

if (typeof module !== "undefined") module.exports = { SIM, prepTrack, castRay, newCar, observe, stepCar, respawn, mlpForward, rayFollower };
