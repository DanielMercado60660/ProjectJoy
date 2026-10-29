// Exercise both canvas paths without a GUI or external dependencies.
const fs = require('fs'), vm = require('vm');
const source = fs.readFileSync(process.argv[2], 'utf8');
const script = source.match(/<script>([\s\S]*?)<\/script>/)[1];
let strokes = 0, nextFrame;
const canvasContext = new Proxy({}, {get: (_, key) => (...args) => { strokes++; }, set: () => true});
const elements = {};
for (const id of ['episode','view','replay','details','info']) {
  elements[id] = {value: '0', options: [], appendChild(x){this.options.push(x)},
                  getContext(){return canvasContext}};
}
const context = vm.createContext({
  document: {getElementById: id => elements[id], createElement: () => ({})},
  performance: {now: () => 0}, requestAnimationFrame: fn => {nextFrame = fn;},
});
vm.runInContext(script, context);
for (let i = 0; i < elements.episode.options.length; i++) {
  elements.episode.value = String(i);
  elements.episode.onchange();
  nextFrame(100);
  if (!elements.details.textContent.includes('episode_')) throw Error('Missing receipt provenance');
}
if (!strokes) throw Error('No drawing commands');
console.log(`R5 demo OK: ${elements.episode.options.length} traces, ${strokes} canvas operations`);
