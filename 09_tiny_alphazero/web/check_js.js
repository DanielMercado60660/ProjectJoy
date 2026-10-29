// node web/check_js.js web/weights.json  -> compares JS forward pass with MLX reference outputs
const fs = require("fs");
const AZ = require("./az.js");
const spec = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const net = new AZ.Net(spec);
let worst = 0;
for (const t of spec.tests) {
  const g = new AZ.Game(); for (const c of t.moves) g.play(c);
  const t0 = Date.now(); const out = net.forward(g.features()); const ms = Date.now() - t0;
  // relative error: MLX's GPU convolutions differ from a float64 numpy reference by ~1e-3
  // relative on trained weights (checked), while this JS forward matches float64 closely
  for (let i = 0; i < 7; i++) worst = Math.max(worst, Math.abs(out.logits[i] - t.logits[i]) / (1 + Math.abs(t.logits[i])));
  worst = Math.max(worst, Math.abs(out.value - t.value));
  console.log(JSON.stringify(t.moves), "value js", out.value.toFixed(5), "mlx", t.value.toFixed(5), ms + "ms");
}
let t0 = Date.now(); const g = new AZ.Game(); g.play(3);
const res = AZ.search(net, g, 100);
console.log("100-sim search", Date.now() - t0, "ms; visits", Array.from(res.visits).join(","), "best", AZ.bestMove(res));
console.log("max rel diff", worst.toExponential(2), worst < 2e-3 ? "OK" : "MISMATCH");
process.exit(worst < 2e-3 ? 0 : 1);
