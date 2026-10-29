// Validate media/demo.html offline: extract its script, run it in node (no DOM), and render every
// event x batch x metric x filter combination through the page's own pure render functions.
const fs = require("fs"), vm = require("vm"), path = require("path");
const html = fs.readFileSync(process.argv[2] || path.join(__dirname, "..", "media", "demo.html"), "utf8");
if (/(src|href)=["']https?:/i.test(html)) throw new Error("external resource referenced");
const m = html.match(/<script>([\s\S]*?)<\/script>/);
if (!m) throw new Error("no inline script");
const sandbox = { module: { exports: {} }, console };
vm.runInNewContext(m[1], sandbox);
const P = sandbox.module.exports;
if (!P.DATA || !Array.isArray(P.DATA.events)) throw new Error("DATA not inlined");
let n = 0;
for (const ev of P.DATA.events) {
  P.metaHTML(ev);
  for (const b of ev.batches) {
    for (const metric of ["tput", "lat"]) for (const qonly of [false, true]) for (const log of [false, true]) {
      const opts = { metric, qonly, log };
      const out = P.raceHTML(ev, b.batch, opts);
      const rows = P.laneRows(ev, b.batch, opts);
      if (rows.length && !out.includes(rows[0].athlete)) throw new Error(`race missing ${rows[0].athlete} in ${ev.id}`);
      if (qonly && rows.some(r => r.qualifies === false || r.fell_back)) throw new Error("qualifying filter leaked");
      if (out.includes("NaN") || out.includes("undefined")) throw new Error(`bad value in ${ev.id} b${b.batch}`);
      rows.forEach(r => P.tipHTML(r, ev.unit));
      n++;
    }
    const d = P.detailsHTML(ev, b.batch);
    if (d.includes("undefined")) throw new Error(`details undefined in ${ev.id}`);
  }
}
P.medalsHTML(P.DATA.medals);
console.log(`demo.html OK: ${P.DATA.events.length} events, ${n} views rendered, ${P.DATA.medals.length} medal rows`);
