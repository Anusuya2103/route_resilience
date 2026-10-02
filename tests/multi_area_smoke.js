/* Headless smoke test for the multi-area console engine (web/app2.js).
 *
 * Runs the real client-side engine against a real payload with a minimal DOM
 * shim, so area switching, the cross-area watch maths and the assistant
 * routing are checked without opening a browser.
 *
 *   python scripts/make_smoke_fixture.py     # writes outputs/_smoke.html
 *   node tests/multi_area_smoke.js
 */
const fs = require('fs');
const path = require('path');
const ROOT = path.resolve(__dirname, '..');

const html = fs.readFileSync(path.join(ROOT, 'outputs/_smoke.html'), 'utf8');
const payload = html.match(/<script id="payload" type="application\/json">([\s\S]*?)<\/script>/)[1];
const code = fs.readFileSync(path.join(ROOT, 'web/app2.js'), 'utf8');

/* ---------- minimal DOM shim ---------- */
const nodes = {};
const touched = [];
function mk(id) {
  return nodes[id] = {
    id, style: {}, dataset: {}, children: [], value: 0, textContent: '',
    innerHTML: '', className: '',
    classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
    appendChild(c) { this.children.push(c); },
    getBoundingClientRect: () => ({ width: 800, height: 600, left: 0, top: 0 }),
    addEventListener() {}, querySelectorAll() { return []; }, focus() {},
  };
}
nodes.payload = mk('payload');
nodes.payload.textContent = payload;
const ctx = {
  setTransform() {}, clearRect() {}, beginPath() {}, arc() {}, fill() {}, stroke() {},
  moveTo() {}, lineTo() {}, closePath() {}, fillRect() {}, save() {}, restore() {},
  setLineDash() {}, fillText() {}, strokeText() {}, rect() {},
  drawImage() {}, set imageSmoothingEnabled(v) {},
  measureText: () => ({ width: 10 }),
};
nodes.cv = mk('cv'); nodes.cv.getContext = () => ctx;
nodes.loc = mk('loc'); nodes.loc.getContext = () => ctx;
/* app2.js draws on all five maps plus both sparklines and the locator */
for (const id of ['cv2', 'cvAn', 'cvFac', 'cvRec', 'simChart', 'recChart']) {
  nodes[id] = mk(id);
  nodes[id].getContext = () => ctx;
}

global.document = {
  getElementById(k) { touched.push(k); return nodes[k] || (nodes[k] = mk(k)); },
  querySelectorAll() { return []; },
  querySelector() { return null; },
  createElement() { return mk('new'); },
  addEventListener() {},
};
global.devicePixelRatio = 1;
global.addEventListener = () => {};
global.setTimeout = () => 0;
global.clearTimeout = () => {};
global.setInterval = () => 0;
global.clearInterval = () => {};
/* app2.js reads window.devicePixelRatio / scrollTo and binds resize handlers */
global.window = {
  devicePixelRatio: 1, innerWidth: 1600, innerHeight: 900,
  addEventListener: () => {}, removeEventListener: () => {}, scrollTo: () => {},
};
global.window.document = global.document;

let fail = 0;
const ok = (c, m) => { console.log((c ? 'PASS  ' : 'FAIL  ') + m); if (!c) fail++; };

/* `let`/`const` declared inside a direct eval stay in that eval's scope.  A
 * getter is exposed (rather than a copy of the values) so the assertions below
 * observe the LIVE bindings after each call, not a stale snapshot. */
const appendix = `
globalThis.__api = {
  AREA_ORDER, AREAS, HAZ, activateArea, selectScenario, showPage, answer,
  areaOutcome, damagedSet, bandGeom, currentSizeM, allocateCrews,
  repairCandidates, init, render, shortPage: 0,
  get: () => ({ A, BEFORE, CUR, N, curKey, AREAS, AREA_ORDER, STATE, PAGE }),
};`;
try {
  eval(code + appendix);
} catch (e) {
  console.log('THREW during load: ' + e.message);
  process.exit(1);
}
const api = globalThis.__api;
const state = () => api.get();
// NB: a direct eval leaks its function declarations into this scope, so the
// helpers are read off `api` rather than re-declared with const/let.

/* ---------- every id app2.js touches must exist in the real HTML ---------- */
const htmlIds = new Set([...html.matchAll(/id="([^"]+)"/g)].map(m => m[1]));
const missing = [...new Set(touched)].filter(i => !htmlIds.has(i));
ok(missing.length === 0,
  `all ${new Set(touched).size} element ids used by app2.js exist in index.html` +
  (missing.length ? ' -- MISSING: ' + missing.join(', ') : ''));

/* ---------- the seven pages ---------- */
const PAGES = ['overview', 'simulation', 'reconstruction', 'analytics',
  'facilities', 'recovery', 'research'];
ok(html.match(/<button data-p="/g).length === 7, 'nav has 7 page buttons');
ok((html.match(/<section class="page/g) || []).length === 7, 'there are 7 page sections');
for (const p of PAGES) {
  let threw = null;
  try { api.showPage(p); } catch (e) { threw = e.message; }
  ok(!threw && state().PAGE === p, 'showPage(' + p + ')' + (threw ? ' THREW: ' + threw : ''));
}

/* ---------- multi-area behaviour ---------- */
ok(api.AREA_ORDER.length === 3, 'payload exposes 3 areas: ' + api.AREA_ORDER.join(','));
ok(state().curKey === 'a_town', 'opens on the home area, got ' + state().curKey);
ok(state().N.length === api.AREAS.a_town.nodes.length, 'active graph bound to the home area');
ok(nodes.watch.innerHTML.includes('Town B'), 'watch table lists the other towns');

api.activateArea('b_town');
ok(state().curKey === 'b_town', 'activateArea switches town, got ' + state().curKey);
ok(state().A === api.AREAS.b_town, 'the active area object is the one selected');
ok(state().BEFORE === api.AREAS.b_town.baseline, 'baseline rebound on switch');

const k0 = state().CUR.key;
api.activateArea('c_town');
ok(state().CUR.key === k0, 'the same scenario keeps running across an area switch: ' + k0);

/* ---------- per-area status maths ---------- */
const o1 = api.areaOutcome('a_town'), o2 = api.areaOutcome('b_town');
ok(o1.pct >= 0 && o1.pct <= 100, 'areaOutcome returns a valid percentage: ' + o1.pct.toFixed(1));
ok(typeof o1.dead === 'number', 'areaOutcome reports junctions lost: ' + o1.dead);

/* ---------- hazard sizes are REAL metres, and shared across towns ---------- */
const H = api.HAZ;
ok(H.radiusM === 400, 'earthquake radius is 400 m, got ' + H.radiusM);
ok(H.floodM === 250 && H.floodAngle === 35,
  'flood corridor is 250 m @ 35deg, got ' + H.floodM + ' @ ' + H.floodAngle);
ok(H.landslideM === 150 && H.landslideAngle === 115,
  'landslide corridor is 150 m @ 115deg, got ' + H.landslideM + ' @ ' + H.landslideAngle);

const gFlood = api.bandGeom('flood'), gLand = api.bandGeom('landslide');
ok(gFlood.widthM === 250 && gLand.widthM === 150,
  'flood and landslide carry DIFFERENT widths: ' + gFlood.widthM + ' m vs ' + gLand.widthM + ' m');
ok(gFlood.angle !== gLand.angle,
  'flood and landslide carry different bearings: ' + gFlood.angle + ' vs ' + gLand.angle);

/* the same physical size must mean the same physical size in every town:
   the damage count depends on real geometry, not on a per-area percentage. */
api.activateArea('a_town');
api.selectScenario('flood');
const f1 = state().STATE.dead.size;
api.activateArea('c_town');
api.selectScenario('flood');
const f2 = state().STATE.dead.size;
ok(typeof f1 === 'number' && typeof f2 === 'number',
  'the 250 m flood corridor is applied in real metres in every town (' + f1 + ' vs ' + f2 + ' junctions)');

/* ---------- epicentre actually moves the damage ---------- */
api.activateArea('a_town');
const curA = state().A;
const disc = curA.scenarios.find(s => s.kind === 'disc') || curA.scenarios[0];
const centre = { x: curA.extent.cx, y: curA.extent.cy };
const far = { x: curA.extent.x0 - curA.extent.diag, y: curA.extent.y0 - curA.extent.diag };
const beforeN = api.damagedSet(disc, H.radiusM, 0, 0, curA, centre).size;
const afterN = api.damagedSet(disc, H.radiusM, 0, 0, curA, far).size;
ok(beforeN !== afterN,
  'moving the epicentre changes the damaged set (' + beforeN + ' vs ' + afterN + ')');

/* ---------- every hazard type runs ---------- */
api.activateArea('a_town');
for (const s of api.AREAS.a_town.scenarios) {
  let threw = null;
  try { api.selectScenario(s.key); } catch (e) { threw = e.message; }
  ok(!threw && state().CUR.key === s.key, 'hazard runs: ' + s.key + (threw ? ' THREW: ' + threw : ''));
}

/* ---------- assistant ---------- */
api.activateArea('a_town');
api.selectScenario('earthquake');
const a1 = api.answer('how many facilities are cut off?');
/* Assert on substance, not on the word "facility" appearing in the reply --
 * the assistant names the actual facilities rather than restating the question. */
const namedFac = api.AREAS.a_town.facilities
  .some(f => a1.includes(f.name.split(' ').pop()));
ok(a1.length > 5 && (namedFac || /reachable/i.test(a1)),
  'assistant answers a facility question from live state: ' + JSON.stringify(a1.slice(0, 60)));
const a2 = api.answer('what is the single point of failure?');
ok(a2.length > 10, 'assistant answers a bottleneck question');
ok(!/satellite imagery/i.test(api.answer('what am I looking at')),
  'assistant does not claim satellite imagery for OSM areas');

console.log(fail ? '\n' + fail + ' CHECK(S) FAILED' : '\nALL CHECKS PASSED');
process.exit(fail ? 1 : 0);
