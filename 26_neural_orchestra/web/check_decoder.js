// node web/check_decoder.js media/demo.html
// Re-runs the baked check vector through the in-page JS decoder and compares with the MLX (Python) output.
const fs = require("fs");
const path = require("path");
const html = fs.readFileSync(process.argv[2], "utf8");
const m = html.match(/const DATA = (\{.*?\});\n/s);
if (!m) { console.error("no DATA"); process.exit(2); }
const DATA = JSON.parse(m[1]);
const NO = require(path.join(__dirname, "core.js"));
const ar = DATA.decoder.kind === "ar";
const dec = ar ? NO.loadAR(DATA.decoder) : NO.loadDecoder(DATA.decoder);
const sv = Array.from({ length: dec.n_styles }, (_, i) => (i === DATA.check.style ? 1 : 0));
const t0 = Date.now();
const out = ar ? NO.generateAR(dec, DATA.check.z, sv, "threshold", 1, Math.random) : NO.decode(dec, DATA.check.z, sv);
const ms = Date.now() - t0;
let maxd = 0, agree = 0;
for (let c = 0; c < NO.CELLS; c++) {
  // for the AR decoder probabilities depend on the (thresholded) history, so compare once the hit histories agree
  maxd = Math.max(maxd, Math.abs(out.p[c] - DATA.check.p[c]));
  const hit = ar ? out.pat.h[c] : +(out.p[c] > 0.5);
  agree += hit === DATA.check.h[c];
}
const kit = NO.makeKit(44100);
const finite = kit.every((b) => b.every(Number.isFinite));
const peak = Math.max(...kit.map((b) => b.reduce((a, x) => Math.max(a, Math.abs(x)), 0)));
console.log(JSON.stringify({ max_abs_prob_diff: maxd, threshold_agreement: agree / NO.CELLS, decode_ms: ms, kit_finite: finite, kit_peak: peak }));
process.exit(maxd < 0.05 && agree / NO.CELLS > 0.99 && finite ? 0 : 1);
