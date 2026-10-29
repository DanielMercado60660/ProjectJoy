// Called by test_js_parity.py: node js_harness.js <input.json>  -> prints JSON results
const fs = require("fs");
const path = require("path");
const T = require(path.join(__dirname, "..", "web", "terra_core.js"));
const inp = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const G = new T.Graph(inp.h, inp.w, inp.elev, inp.kind);
const out = { edge_costs: Array.from(G.cst).map(x => (isFinite(x) ? x : -1)), queries: [], dstar_bad: 0, dstar_checks: 0 };
for (const [s, g] of inp.queries) {
  const d = T.astar(G, s, g, null);
  const a = T.astar(G, s, g, T.octileField(G, g));
  const D = new T.DStarLite(G, s, g); D.compute();
  out.queries.push({ dijkstra: isFinite(d.cost) ? d.cost : -1, astar: isFinite(a.cost) ? a.cost : -1,
                     dstar: isFinite(D.costToGo()) ? D.costToGo() : -1, dstar_path_cost: T.pathCost(G, D.path()) });
}
// D* Lite under random edits + moves vs JS Dijkstra (deterministic LCG)
let seed = 12345; const rnd = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648);
for (const [s, g] of inp.queries) {
  const H = new T.Graph(inp.h, inp.w, inp.elev, inp.kind);
  const D = new T.DStarLite(H, s, g); D.compute();
  let pos = s;
  for (let step = 0; step < 20; step++) {
    const p = D.path();
    if (p.length > 1) { pos = p[Math.min(p.length - 1, 1 + Math.floor(rnd() * 3))]; D.moveTo(pos); }
    if (pos === g) break;
    const cells = [];
    for (let i = 0; i < 4; i++) { const u = Math.floor(rnd() * H.n); if (u !== pos && u !== g) cells.push(u); }
    const kinds = [T.WATER, T.WATER, T.GRASS, T.ROCK, T.FOREST, T.BRIDGE];
    const aff = H.setCells(cells, kinds[Math.floor(rnd() * kinds.length)], rnd() < 0.3 ? rnd() : null);
    D.updateVertices(aff); D.compute();
    const ref = T.astar(H, pos, g, null).cost, got = D.costToGo();
    out.dstar_checks++;
    if (!((ref === Infinity && got === Infinity) || Math.abs(ref - got) < 1e-6)) out.dstar_bad++;
    // also check the JS local rebuild against a full rebuild
  }
  const F = new T.Graph(H.h, H.w, H.elev, H.kind);
  for (let i = 0; i < F.cst.length; i++) if (F.cst[i] !== H.cst[i] && !(isNaN(F.cst[i]) && isNaN(H.cst[i]))) { out.rebuild_mismatch = (out.rebuild_mismatch || 0) + 1; }
}
const gen = T.generate(7, 64);
out.gen_ok = gen.kind.length === 64 * 64 && gen.kind.some(k => k === T.WATER);
console.log(JSON.stringify(out));
