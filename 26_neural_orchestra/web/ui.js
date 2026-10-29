// Neural Orchestra UI: Web Audio step sequencer + in-browser VAE decoding. Needs DATA and NO.
(() => {
  const { STEPS, NI, CELLS } = NO;
  const $ = (id) => document.getElementById(id);
  const dec = DATA.decoder.kind === "ar" ? NO.loadAR(DATA.decoder) : NO.loadDecoder(DATA.decoder);
  const L = dec.latent, NS = dec.n_styles;
  const oneHot = (k) => Array.from({ length: NS }, (_, i) => (i === k ? 1 : 0));
  const lerp = (a, b, t) => a.map((x, i) => x + (b[i] - x) * t);
  const randn = () => Array.from({ length: L }, () => NO.gaussian(Math.random));

  const S = {
    pat: NO.unpack([]), style: 0, mode: DATA.mode === "threshold" ? "threshold" : "sample", temp: DATA.temperature || 1,
    z: randn(), sv: oneHot(0), u: Float32Array.from({ length: CELLS }, Math.random),
    bpm: 110, feel: true, playing: false, step: 0, nextTime: 0, src: "",
    morph: { a: 0, b: 1, za: randn(), zb: randn(), alpha: 0, auto: false }, setIdx: {},
  };

  // ---------- grid ----------
  const grid = $("grid"), cells = [];
  for (let i = 0; i < NI; i++) {
    const lab = document.createElement("div");
    lab.className = "lab"; lab.textContent = DATA.short[i]; lab.title = DATA.instruments[i];
    lab.style.color = `var(--i${i})`;
    grid.appendChild(lab);
    for (let s = 0; s < STEPS; s++) {
      const c = document.createElement("div");
      c.className = "cell" + (s % 4 === 0 ? " beat" : "");
      if (s === 16) c.style.marginLeft = "6px";
      c.style.setProperty("--c", `var(--i${i})`);
      const tick = document.createElement("div"); tick.className = "tick"; c.appendChild(tick);
      c.addEventListener("click", () => {
        const k = s * NI + i;
        S.pat.h[k] = S.pat.h[k] ? 0 : 1; S.pat.v[k] = S.pat.h[k] ? 0.8 : 0; S.pat.o[k] = 0;
        S.src = "edited by hand"; draw();
      });
      grid.appendChild(c); cells[s * NI + i] = c;
    }
  }
  function draw() {
    let n = 0;
    for (let k = 0; k < CELLS; k++) {
      const c = cells[k], hit = S.pat.h[k];
      c.classList.toggle("hit", !!hit);
      c.style.opacity = hit ? (0.35 + 0.65 * S.pat.v[k]).toFixed(2) : "";
      const t = c.firstChild;
      t.style.display = hit && Math.abs(S.pat.o[k]) > 0.02 ? "block" : "none";
      t.style.left = `${50 + 90 * S.pat.o[k]}%`;
      n += hit;
    }
    $("src").innerHTML = S.src;
    $("stats").textContent = `${n} hits · ${S.bpm} bpm`;
  }
  function setPattern(p, src) { S.pat = p; S.src = src; draw(); }

  // ---------- neural decoding ----------
  function decodeTo(z, sv, label) {
    const t0 = performance.now();
    const T = S.mode === "sample" ? S.temp : 1;
    let k = 0;
    const coin = () => S.u[k++];
    const p = dec.kind === "ar" ? NO.generateAR(dec, z, sv, S.mode, T, coin).pat : NO.toPattern(NO.decode(dec, z, sv, T), S.mode, coin);
    setPattern(p, label + ` <span class="small">· decoded in ${(performance.now() - t0).toFixed(1)} ms</span>`);
  }
  const styleName = (k) => DATA.styles[k];
  const modeLabel = () => (S.mode === "sample" ? `sample T=${S.temp.toFixed(2)}` : "threshold");
  function newGroove() {
    S.z = randn(); S.sv = oneHot(S.style);
    S.u = Float32Array.from({ length: CELLS }, Math.random);
    setBpm(DATA.tempo[styleName(S.style)]);
    decodeTo(S.z, S.sv, `<b>VAE</b> · ${styleName(S.style)} · ${modeLabel()}`);
    padPos = [0, 0]; drawPad();
  }

  // ---------- style chips + compare ----------
  const chips = $("styles");
  DATA.styles.forEach((st, k) => {
    const b = document.createElement("button"); b.className = "chip"; b.textContent = st;
    b.onclick = () => { S.style = k; [...chips.children].forEach((c, j) => c.classList.toggle("sel", j === k)); newGroove(); };
    chips.appendChild(b);
  });
  chips.children[0].classList.add("sel");
  $("gen").onclick = newGroove;
  $("mode").value = S.mode;
  const syncTemp = () => { $("temp").value = Math.round(S.temp * 100); $("tempv").textContent = S.temp.toFixed(2); $("temp").disabled = S.mode !== "sample"; };
  syncTemp();
  $("temp").oninput = (e) => { S.temp = e.target.value / 100; syncTemp(); decodeTo(S.z, S.sv, `<b>VAE</b> · ${styleName(S.style)} · ${modeLabel()}`); };
  $("mode").onchange = (e) => { S.mode = e.target.value; syncTemp(); decodeTo(S.z, S.sv, `<b>VAE</b> · ${styleName(S.style)} · ${modeLabel()}`); };
  const SETNAME = { real: "Real data (test split)", markov_bar: "Bar-Markov baseline", bernoulli: "Bernoulli baseline" };
  document.querySelectorAll("[data-set]").forEach((b) => {
    b.onclick = () => {
      const set = b.dataset.set, st = styleName(S.style), list = DATA.sets[set][st];
      const i = (S.setIdx[set + st] = ((S.setIdx[set + st] ?? -1) + 1) % list.length);
      setBpm(DATA.tempo[st]);
      setPattern(NO.unpack(list[i]), `<b>${SETNAME[set]}</b> · ${st} · #${i + 1}/${list.length}`);
    };
  });

  // ---------- morph ----------
  for (const id of ["ma", "mb"]) DATA.styles.forEach((st, k) => $(id).add(new Option(st, k)));
  $("ma").value = 0; $("mb").value = 1;
  function morphUpdate() {
    const m = S.morph, a = m.alpha;
    S.z = lerp(m.za, m.zb, a); S.sv = lerp(oneHot(m.a), oneHot(m.b), a);
    setBpm(Math.round(DATA.tempo[styleName(m.a)] * (1 - a) + DATA.tempo[styleName(m.b)] * a));
    $("maval").textContent = `α = ${a.toFixed(2)}`;
    decodeTo(S.z, S.sv, `<b>VAE morph</b> · ${styleName(m.a)} → ${styleName(m.b)} · α ${a.toFixed(2)}`);
  }
  $("ma").onchange = (e) => { S.morph.a = +e.target.value; morphUpdate(); };
  $("mb").onchange = (e) => { S.morph.b = +e.target.value; morphUpdate(); };
  $("mroll").onclick = () => { S.morph.za = randn(); S.morph.zb = randn(); morphUpdate(); };
  $("malpha").oninput = (e) => { S.morph.alpha = e.target.value / 1000; morphUpdate(); };
  $("mauto").onclick = (e) => {
    S.morph.auto = !S.morph.auto; e.target.classList.toggle("on", S.morph.auto);
    if (S.morph.auto) { S.morph.alpha = 0; $("malpha").value = 0; morphUpdate(); if (!S.playing) togglePlay(); }
  };
  function onBar() {  // called once per 2-bar loop
    if (!S.morph.auto) return;
    let a = Math.round(S.morph.alpha * 8) + 1;
    if (a > 8) {  // reached B: swap ends and walk back
      a = 0;
      [S.morph.a, S.morph.b, S.morph.za, S.morph.zb] = [S.morph.b, S.morph.a, S.morph.zb, S.morph.za];
      $("ma").value = S.morph.a; $("mb").value = S.morph.b;
    }
    S.morph.alpha = a / 8; $("malpha").value = a * 125; morphUpdate();
  }

  // ---------- baked walks ----------
  DATA.interps.forEach((w, k) => $("walk").add(new Option(`${w.a} → ${w.b}`, k)));
  function walkUpdate() {
    const w = DATA.interps[+$("walk").value], k = +$("walkpt").value;
    setBpm(Math.round(w.bpm[k]));
    setPattern(NO.unpack(w.points[k]), `<b>Baked walk</b> · ${w.a} → ${w.b} · point ${k + 1}/9 (decoded in Python, same as interp/${w.a}_to_${w.b}.wav)`);
    S.z = w.z[k].slice(); const a = k / 8;
    S.sv = lerp(oneHot(DATA.styles.indexOf(w.a)), oneHot(DATA.styles.indexOf(w.b)), a);
  }
  $("walk").onchange = walkUpdate; $("walkpt").oninput = walkUpdate;

  // ---------- latent pad ----------
  const pad = $("pad"), pctx = pad.getContext("2d");
  let padPos = [0, 0], padBase = null, dragging = false;
  function drawPad() {
    const W = pad.width, H = pad.height;
    pctx.clearRect(0, 0, W, H);
    pctx.strokeStyle = "#33333b"; pctx.lineWidth = 1;
    for (let i = 1; i < 6; i++) { const x = (i * W) / 6; pctx.beginPath(); pctx.moveTo(x, 0); pctx.lineTo(x, H); pctx.stroke(); pctx.beginPath(); pctx.moveTo(0, x); pctx.lineTo(W, x); pctx.stroke(); }
    const x = (padPos[0] * 0.5 + 0.5) * W, y = (0.5 - padPos[1] * 0.5) * H;
    pctx.fillStyle = "#ff8a3d"; pctx.beginPath(); pctx.arc(x, y, 9, 0, 7); pctx.fill();
    pctx.fillStyle = "#9a978f"; pctx.font = "11px sans-serif"; pctx.fillText("PC1 →", W - 44, H - 8); pctx.fillText("PC2 ↑", 6, 14);
  }
  function padMove(ev) {
    const r = pad.getBoundingClientRect();
    padPos = [Math.max(-1, Math.min(1, ((ev.clientX - r.left) / r.width) * 2 - 1)), Math.max(-1, Math.min(1, 1 - ((ev.clientY - r.top) / r.height) * 2))];
    const z = padBase.map((b, i) => b + 2.5 * (padPos[0] * DATA.pcs[0][i] + padPos[1] * DATA.pcs[1][i]));
    S.z = z; drawPad();
    decodeTo(z, S.sv, `<b>VAE latent pad</b> · ${S.sv.map((x, i) => x > 0.01 ? DATA.styles[i] : "").filter(Boolean).join("/")} · (${padPos[0].toFixed(2)}, ${padPos[1].toFixed(2)})`);
  }
  pad.addEventListener("pointerdown", (e) => {
    dragging = true; pad.setPointerCapture(e.pointerId);
    if (!padBase) padBase = S.z.map((x, i) => x - 2.5 * (padPos[0] * DATA.pcs[0][i] + padPos[1] * DATA.pcs[1][i]));
    padMove(e);
  });
  pad.addEventListener("pointermove", (e) => dragging && padMove(e));
  pad.addEventListener("pointerup", () => { dragging = false; });
  const origSet = setPattern;
  // any non-pad source resets the pad origin to the new groove
  setPattern = function (p, src) { if (!dragging) padBase = null, padPos = [0, 0], drawPad(); origSet(p, src); };

  // ---------- audio ----------
  let ctx = null, kit = null, master = null;
  function initAudio() {
    ctx = new (window.AudioContext || window.webkitAudioContext)();
    kit = NO.makeKit(ctx.sampleRate).map((x) => { const b = ctx.createBuffer(1, x.length, ctx.sampleRate); b.copyToChannel(x, 0); return b; });
    const comp = ctx.createDynamicsCompressor(); comp.threshold.value = -10; comp.ratio.value = 4;
    master = ctx.createGain(); master.gain.value = $("vol").value / 100;
    master.connect(comp); comp.connect(ctx.destination);
  }
  const phQueue = [];
  function scheduleStep(s, t) {
    const dur = 60 / S.bpm / 4;
    for (let i = 0; i < NI; i++) {
      const k = s * NI + i;
      if (!S.pat.h[k]) continue;
      const src = ctx.createBufferSource(), g = ctx.createGain();
      src.buffer = kit[i]; g.gain.value = NO.GAIN[i] * S.pat.v[k];
      src.connect(g); g.connect(master);
      src.start(Math.max(ctx.currentTime, t + (S.feel ? S.pat.o[k] * dur : 0)));
    }
    phQueue.push([s, t]);
  }
  function tick() {
    while (S.nextTime < ctx.currentTime + 0.12) {
      scheduleStep(S.step, S.nextTime);
      S.nextTime += 60 / S.bpm / 4;
      S.step = (S.step + 1) % STEPS;
      if (S.step === 0) onBar();
    }
  }
  let timer = null, lastPh = -1;
  function frame() {
    if (!S.playing) return;
    while (phQueue.length > 1 && phQueue[1][1] <= ctx.currentTime) phQueue.shift();
    const s = phQueue.length ? phQueue[0][0] : -1;
    if (s !== lastPh) {
      for (let i = 0; i < NI; i++) { if (lastPh >= 0) cells[lastPh * NI + i].classList.remove("ph"); if (s >= 0) cells[s * NI + i].classList.add("ph"); }
      lastPh = s;
    }
    requestAnimationFrame(frame);
  }
  function togglePlay() {
    if (!ctx) initAudio();
    if (ctx.state === "suspended") ctx.resume();
    S.playing = !S.playing;
    $("play").textContent = S.playing ? "■ Stop" : "▶ Play";
    if (S.playing) { S.step = 0; S.nextTime = ctx.currentTime + 0.06; phQueue.length = 0; timer = setInterval(tick, 25); tick(); requestAnimationFrame(frame); }
    else { clearInterval(timer); for (let i = 0; i < NI; i++) if (lastPh >= 0) cells[lastPh * NI + i].classList.remove("ph"); lastPh = -1; }
  }
  function setBpm(b) { S.bpm = b; $("bpm").value = b; $("bpmv").textContent = b; }
  $("play").onclick = togglePlay;
  $("bpm").oninput = (e) => { setBpm(+e.target.value); draw(); };
  $("feel").onchange = (e) => { S.feel = e.target.checked; };
  $("vol").oninput = (e) => { if (master) master.gain.value = e.target.value / 100; };
  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
    if (e.code === "Space") { e.preventDefault(); togglePlay(); }
    if (e.key === "n" || e.key === "N") newGroove();
  });

  $("about").innerHTML = `<b>What you're hearing.</b> Patterns are 32 sixteenth-note steps × 9 drums (kick, snare, clap, closed/open hat, low/high tom, ride, crash), each hit with a velocity and a microtiming nudge (the dark tick inside a cell). The VAE was trained on ${"procedurally generated"} grooves in 8 styles; its decoder (${dec.kind === "ar" ? `latent ${L} + style ${NS} → GRU(${dec.H}) stepping through 32 sixteenths` : `latent ${L} + style ${NS} → ${DATA.decoder.layers.map((l) => l.n_out).join(" → ")}`}) is stored here as int8 weights and evaluated in JavaScript on every click. <b>Loose</b> mode draws each hit as a coin flip with the decoder's probability (one fixed set of coins per groove, so morphs stay smooth); <b>tight</b> keeps only cells with p &gt; 0.5. Drums are synthesized on the fly — no samples.`;

  // initial groove
  S.style = DATA.styles.indexOf("funk"); chips.children[0].classList.remove("sel"); chips.children[S.style].classList.add("sel");
  newGroove(); drawPad();
  window.__NO_STATE = S;
})();
