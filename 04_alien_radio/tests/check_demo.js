// Headless check of media/demo.html: stub the few DOM calls the page makes, run its script,
// and print the JS-vs-Python self-check plus the live stats text.  Usage: node tests/check_demo.js media/demo.html
const fs = require("fs");
const html = fs.readFileSync(process.argv[2], "utf8");
const script = html.slice(html.indexOf("<script>") + 8, html.lastIndexOf("</script>"));
const els = {};
const el = id => els[id] || (els[id] = {id, innerHTML: "", textContent: "", value: "", options: [], add(o){ this.options.push(o); if (!this.value) this.value = o.value; }, querySelectorAll: () => []});
global.document = {getElementById: el, querySelectorAll: () => []};
global.Option = function(label, value){ this.label = label; this.value = value; };
global.window = global;
global.atob = s => Buffer.from(s, "base64").toString("binary");
eval(script + `
for (const r of D.receivers.slice(0, 4)) { document.getElementById("selR").value = r.id; liveStats(); console.log("pair", document.getElementById("selS").value, "->", r.id, document.getElementById("live").innerHTML.replace(/<[^>]+>/g, " ").replace(/\\s+/g, " ")); }`);
console.log(JSON.stringify(window.__selfcheck));
console.log("live:", els.live.innerHTML.replace(/<[^>]+>/g, " ").replace(/\s+/g, " "));
