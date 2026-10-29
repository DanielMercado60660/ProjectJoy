// Headless check of the JS port in a demo.html: runs the page script with a stub DOM,
// prints the self-check result (JS step vs MLX step, fire mask = all) and timing of 20 stochastic steps.
// Usage: node tests/check_demo_js.mjs media/demo.html
import fs from 'node:fs';
import vm from 'node:vm';

const html = fs.readFileSync(process.argv[2], 'utf8');
const script = html.match(/<script>([\s\S]*)<\/script>/)[1];
const els = {};
function el(id) {
  if (!els[id]) els[id] = {
    id, value: id === 'speed' ? '2' : id === 'brush' ? '6' : 'main', innerHTML: '', textContent: '', style: {}, width: 0, height: 0,
    appendChild(o) { if (this.value === 'main' && o.value) this.value = o.value; }, addEventListener() {},
    getContext() { return { createImageData: (w, h) => ({ data: new Uint8ClampedArray(w * h * 4) }), putImageData() {} }; },
    getBoundingClientRect() { return { left: 0, top: 0, width: 1, height: 1 }; }, toDataURL() { return ''; },
  };
  return els[id];
}
const ctx = {
  document: { getElementById: el, createElement: () => el('tmp' + Math.random()) },
  performance, requestAnimationFrame: () => {}, Math, Float32Array, Uint8Array, Uint8ClampedArray, console,
};
vm.createContext(ctx);
vm.runInContext(script + '\n;globalThis.__api = {step: () => step(state, next, false), getState: () => state, swap: () => { [state, next] = [next, state]; }, mse};', ctx);
console.log('self-check:', els['check'].innerHTML.replace(/<[^>]+>/g, ''));
const api = ctx.__api;
const t0 = Date.now();
for (let i = 0; i < 60; i++) { api.step(); api.swap(); }
console.log(`60 steps from seed: ${(Date.now() - t0)} ms, mse ${api.mse().toFixed(4)}`);
const ok = /max abs error ([0-9.e+-]+)/.exec(els['check'].innerHTML.replace(/<[^>]+>/g, ''));
process.exit(ok && parseFloat(ok[1]) < 1e-3 ? 0 : 1);
