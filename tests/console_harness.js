/*
 * Headless harness: runs the built console against a minimal DOM + canvas stub
 * so the whole init/render path actually executes.  Without this, a runtime
 * error in the engine would leave pages silently blank in a browser.
 *
 *   node tests/console_harness.js [path/to/emergency_dashboard.html]
 */
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const FILE = process.argv[2] ||
  path.join(__dirname, '..', 'outputs', 'emergency_dashboard.html');
const html = fs.readFileSync(FILE, 'utf8');

const payload = html.match(
  /<script id="payload" type="application\/json">([\s\S]*?)<\/script>/);
if (!payload) throw new Error('no payload script in ' + FILE);
/* The engine is the LAST inline <script> in the file (index.html ships the
 * payload script plus the injected engine, and nothing else inline). */
const allInline = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)];
if (!allInline.length) throw new Error('no inlined engine in ' + FILE);
const engine = [null, allInline[allInline.length - 1][1]];

const NAV = ['overview', 'simulation', 'reconstruction', 'analytics',
  'facilities', 'recovery', 'research'];

/* ---- minimal DOM + canvas ---- */
function makeCtx() {
  const noop = () => {};
  return new Proxy({
    canvas: null, measureText: () => ({ width: 10 }),
    createLinearGradient: () => ({ addColorStop: noop }),
    getImageData: () => ({ data: new Uint8ClampedArray(4) }),
  }, { get(t, k) { return k in t ? t[k] : noop; },
       set(t, k, v) { t[k] = v; return true; } });
}

function makeEl(tag) {
  const el = {
    tagName: String(tag || 'div').toUpperCase(),
    dataset: {}, style: {}, children: [], _html: '', textContent: '',
    value: '', className: '', disabled: false, _rows: [],
    addEventListener() {}, removeEventListener() {},
    getContext: () => (el._ctx = el._ctx || makeCtx()),
    getBoundingClientRect: () => ({ left: 0, top: 0, width: 640, height: 400,
      right: 640, bottom: 400 }),
    appendChild(c) { el.children.push(c); return c; },
    removeChild() {}, setAttribute() {}, removeAttribute() {},
    focus() {}, click() {}, closest() { return null; },
    scrollIntoView() {},
  };
  el.classList = {
    _s: new Set(),
    add: (...c) => c.forEach(x => el.classList._s.add(x)),
    remove: (...c) => c.forEach(x => el.classList._s.delete(x)),
    contains: c => el.classList._s.has(c),
    toggle: (c, on) => {
      if (on === undefined) el.classList._s.has(c) ?
        el.classList._s.delete(c) : el.classList._s.add(c);
      else if (on) el.classList._s.add(c); else el.classList._s.delete(c);
    },
  };
  Object.defineProperty(el, 'className', {
    get() { return [...el.classList._s].join(' '); },
    set(v) { el.classList._s = new Set(String(v).split(/\s+/).filter(Boolean)); },
  });
  Object.defineProperty(el, 'innerHTML', {
    get() { return el._html; },
    set(v) {
      el._html = v;
      /* Return one stub row per <tr> so handlers the engine attaches to
       * generated rows (recovery priorities, watch table) still get wired. */
      el._rows = Array.from(
        { length: (String(v).match(/<tr\b/g) || []).length }, () => makeEl('tr'));
      el.querySelectorAll = sel =>
        (sel === 'tbody tr' || sel === 'tr') ? el._rows : [];
    },
  });
  el.querySelector = () => null;
  el.querySelectorAll = () => [];
  return el;
}

const byId = new Map();
const roots = NAV.map(p => {
  const s = makeEl('section');
  s._id = 'page-' + p;
  byId.set('page-' + p, s);
  return s;
});

const navBtns = NAV.map(p => {
  const b = makeEl('button');
  b.dataset = { p };
  return b;
});

const documentStub = {
  readyState: 'complete',
  documentElement: makeEl('html'),
  body: makeEl('body'),
  createElement: makeEl,
  getElementById: id => byId.get(id) || null,
  querySelector: () => null,
  querySelectorAll(sel) {
    if (sel === '#nav button') return navBtns;
    if (sel === '.page') return roots;
    return [];
  },
  addEventListener() {},
};

const CANVASES = ['cv', 'cv2', 'cvAn', 'cvFac', 'cvRec', 'loc', 'simChart', 'recChart'];
const IDS = ['payload', 'pillMode', 'pillNodes', 'pillSev', 'areaSel', 'scn', 'scn2',
  'simCtls', 'simCtls2', 'scnBlurb', 'rTime', 'vTime', 'tlFill', 'btnPlay',
  'btnTimeReset', 'btnDrag', 'btnReset', 'btnZin', 'btnZout', 'btnFit2',
  'mapbar', 'mapbar2', 'anBar', 'facBar', 'recBar', 'legend', 'legend2',
  'anLegend', 'facLegend', 'recLegend', 'mapScalePill', 'kpis', 'connNote',
  'legendBox', 'watch', 'scnCompare', 'anKpis', 'cenBars', 'cenNote',
  'vulnTable', 'vulnNote', 'artTable', 'compTable', 'facKpis', 'facTable',
  'facNote', 'facList', 'critTable', 'facFilters', 'facKindFilterWrap',
  'routePanel', 'routeFrom', 'routeTo', 'routeOut', 'recKpis', 'prioList',
  'teamList', 'recLog', 'recNote', 'recClock', 'rTeams', 'vTeams',
  'btnAllocate', 'btnAuto', 'btnResetRec', 'btnRecPlay', 'pipe3', 'pipeDetail',
  'pipeNote', 'deepglobeNote', 'scopeNote', 'prov', 'resMetrics', 'resNote',
  'resAblation', 'resAblNote', 'resCompute', 'resComputeNote', 'resDataset',
  'resDatasetNote', 'resHeal', 'resHealNote', 'deepScope', 'brief', 'headline',
  'sevBadge', 'ask', 'btnAsk', 'ans', 'quickQ', 'llmNote', 'analyticNote',
  'footProv', 'mapOv', 'mapSim', 'mapAn', 'mapFac', 'mapRec'];
for (const id of [...CANVASES, ...IDS]) {
  byId.set(id, makeEl(CANVASES.includes(id) ? 'canvas' : 'div'));
}
byId.get('rTime').value = '100';
byId.get('rTime').min = '0';
byId.get('rTime').max = '100';
byId.get('rTeams').value = '3';
/* the engine reads the real JSON out of this element */
byId.get('payload').textContent = payload[1];

const windowStub = {
  devicePixelRatio: 1, innerWidth: 1600, innerHeight: 900,
  addEventListener() {}, removeEventListener() {}, scrollTo() {},
};

const ctxObj = {
  document: documentStub, window: windowStub, console, devicePixelRatio: 1,
  setTimeout, clearTimeout, setInterval, clearInterval,
  JSON, Math, Object, Array, Set, Map, String, Number, Boolean, Date, RegExp,
  Error, isFinite, parseInt, parseFloat, Promise, Symbol, Proxy, Uint8ClampedArray,
};
ctxObj.globalThis = ctxObj;
ctxObj.window.document = documentStub;
ctxObj.document.defaultView = windowStub;
vm.createContext(ctxObj);

/* `let`/`const` at top level do NOT become properties of the vm's global
 * object, so expose the handful the harness inspects.  Assigning the same
 * live bindings also keeps STATE/A/HAZ pointing at the engine's real values
 * after activateArea() rebinds them, so read them through getters. */
const EXPOSE = `globalThis.__x = {
  get STATE(){return STATE}, get A(){return A}, get HAZ(){return HAZ},
  get AREA_ORDER(){return AREA_ORDER}, get PAGE(){return PAGE},
  get AREAS(){return AREAS}, get curKey(){return curKey},
  showPage, selectScenario, activateArea, answer, allocateCrews,
  repairCandidates, components, damagedSet, bandGeom, fmtM, currentSizeM,
};`;

/* ---- run the engine ---- */
try {
  vm.runInContext(engine[1] + EXPOSE, ctxObj,
    { filename: 'app2.js', timeout: 180000 });
} catch (e) {
  console.error('ENGINE THREW:', e && e.stack ? e.stack : e);
  process.exit(1);
}
const X = ctxObj.__x;

/* ---- exercise it ---- */
const errors = [];
function check(label, fn) {
  try { const v = fn(); console.log(`  ok    ${label}${v ? ' -> ' + v : ''}`); }
  catch (e) { errors.push(label + ': ' + e.message); console.log(`  FAIL  ${label}: ${e.message}`); }
}

console.log('headless console harness: ' + path.basename(FILE));
const P = JSON.parse(payload[1]);
console.log('  areas: ' + Object.keys(P.areas || {}).length +
            '  pipeline stages: ' + (P.pipeline || []).length +
            '  research models: ' +
            Object.keys((P.research || {}).models || {}).join('/'));

check('engine booted + rendered without throwing', () => {
  if (typeof X.STATE !== 'object') throw new Error('no STATE');
  return 'dead=' + X.STATE.dead.size + ' comps=' + X.STATE.comps.length;
});

for (const p of NAV) {
  check('showPage(' + p + ')', () => {
    X.showPage(p);
    if (X.PAGE !== p) throw new Error('PAGE is ' + X.PAGE);
  });
}

const H = X.HAZ;
console.log(`  hazard sizes: quake ${H.radiusM} m | flood ${H.floodM} m @ ${H.floodAngle}deg ` +
            `| landslide ${H.landslideM} m @ ${H.landslideAngle}deg`);
check('hazard dimensions are 400 / 250@35 / 150@115', () => {
  if (H.radiusM !== 400 || H.floodM !== 250 || H.landslideM !== 150 ||
      H.floodAngle !== 35 || H.landslideAngle !== 115) {
    throw new Error(`got ${H.radiusM}/${H.floodM}@${H.floodAngle}/${H.landslideM}@${H.landslideAngle}`);
  }
  return 'correct';
});

check('every hazard type simulates', () => {
  const out = [];
  for (const s of X.A.scenarios) {
    X.selectScenario(s.key);
    out.push(s.key + '=' + X.STATE.dead.size + 'j');
  }
  return out.join(' ');
});

check('every area activates', () => {
  const out = [];
  for (const k of X.AREA_ORDER) {
    X.activateArea(k);
    out.push(k + '=' + X.STATE.comps.length + 'p');
  }
  return out.join(' ');
});

check('assistant answers from live state', () =>
  X.answer('which junction matters most?').slice(0, 46) + '...');

check('recovery allocation produces crews', () => {
  X.activateArea(X.AREA_ORDER[0]);
  X.allocateCrews();
  return (X.STATE.crews || []).length + ' crew(s)';
});

if (errors.length) {
  console.error('\n' + errors.length + ' CHECK(S) FAILED');
  process.exit(1);
}
console.log('\nALL CHECKS PASSED');
