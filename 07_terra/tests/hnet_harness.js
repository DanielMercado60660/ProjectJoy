// node hnet_harness.js <in.json>: runs web/hnet.js on given features, prints r and timing
const fs = require("fs"), path = require("path");
global.TERRA = require(path.join(__dirname, "..", "web", "terra_core.js"));
const HNET = require(path.join(__dirname, "..", "web", "hnet.js"));
const inp = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
HNET.load(inp.spec);
const G = new TERRA.Graph(inp.h, inp.w, inp.elev, inp.kind);
const x = HNET.features(G, inp.goal);
const t0 = Date.now();
const r = HNET.forward(x, inp.h, inp.w, 8);
const ms = Date.now() - t0;
console.log(JSON.stringify({ features: Array.from(x), r: Array.from(r), ms }));
