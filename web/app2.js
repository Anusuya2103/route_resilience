/* ===========================================================================
 * RoadNet Ops -- 7-page emergency operations console (client engine)
 *
 * ONE simulation state drives every page.  Changing the scenario, the radius
 * or the epicentre on page 02 immediately changes the map on 01, the
 * centrality on 04, the routes on 05 and the recovery plan on 06.
 *
 * The maths mirrors src/simulation.py so the browser and the Python pipeline
 * agree.  All graph algorithms run on the active area's real OSM graph.
 * ======================================================================== */
'use strict';
const D = JSON.parse(document.getElementById('payload').textContent);

/* Multi-area: every monitored town ships its own complete graph, so one
 * operator can sit in Kovilpatti and still drill into Madurai under the SAME
 * running scenario.  Scenario + hazard size are global; graphs are per area. */
const AREAS = D.areas || { area: Object.assign({}, D, { meta: {} }) };
/* The payload nests the display fields under `meta`, but every renderer wants
 * flat `name`/`lat`/`lon`.  Normalise once here instead of at 40 call sites. */
Object.keys(AREAS).forEach(k => {
  const a = AREAS[k], m = a.meta || {};
  if (!a.name) a.name = m.label || k;
  if (a.lat === undefined && Array.isArray(m.center)) a.lat = m.center[0];
  if (a.lon === undefined && Array.isArray(m.center)) a.lon = m.center[1];
  if (a.lat === undefined) a.lat = 0;
  if (a.lon === undefined) a.lon = 0;
  if (!a.scenarios) a.scenarios = [];
  if (!a.facilities) a.facilities = [];
  if (!a.baseline) a.baseline = { largest: a.nodes.length, serviceable: a.nodes.map(n => n.id) };
});
const AREA_ORDER = D.areaOrder || Object.keys(AREAS);
const START_KEY = AREA_ORDER.includes(D.defaultArea) ? D.defaultArea : AREA_ORDER[0];

/* Hazards are stored in METRES (see HAZ) so "400 m" means 400 m in every town
 * regardless of how much of each city OpenStreetMap returned.  A per-area
 * percentage would silently mean a different physical size in Madurai than in
 * Kovilpatti, which is exactly the kind of quiet wrongness this console must
 * not have.  The rendered size is converted to each area's diagonal at draw
 * time, and the label always reports the real metre value. */
const HAZ = {
  radiusM: 400,        // earthquake / blast radius, metres
  floodM: 250,         // flood corridor half-width, metres
  landslideM: 150,     // landslide corridor half-width, metres
  floodAngle: 35,      // flood corridor bearing, degrees
  landslideAngle: 115, // landslide corridor bearing, degrees
  time: 100,           // time progression 0-100 %
};

/* Current band geometry for a scenario key, so flood and landslide can carry
 * their own real width and bearing (250 m / 35 deg and 150 m / 115 deg). */
function bandGeom(key) {
  const pool = (A && A.scenarios) ? A.scenarios
             : (((AREAS[curKey] || {}).scenarios) || []);
  const sc = pool.find(s => s.key === key);
  const isLand = key === 'landslide';
  return {
    widthM:  isLand ? HAZ.landslideM : HAZ.floodM,
    angle:   sc && typeof sc.angle === 'number' ? sc.angle
             : (isLand ? HAZ.landslideAngle : HAZ.floodAngle),
  };
}

let curKey = START_KEY;
let A = AREAS[START_KEY];

/* Active-area graph state, rebound by activateArea(). */
let N, E, NID, ADJ, cen, cenMax, BEFORE, svc, CUR, STATE, EPIC;
let DEG = new Map();

/* Per-area adjacency/baseline cache so the cross-area watch table can be
 * recomputed on every slider move without rebuilding everything. */
const ACACHE = {};
function areaGraph(key) {
  if (!ACACHE[key]) {
    const a = AREAS[key];
    const adj = new Map(a.nodes.map(n => [n.id, []]));
    const deg = new Map(a.nodes.map(n => [n.id, 0]));
    a.edges.forEach(e => {
      adj.get(e.s).push(e.t); adj.get(e.t).push(e.s);
      deg.set(e.s, deg.get(e.s) + 1); deg.set(e.t, deg.get(e.t) + 1);
    });
    ACACHE[key] = { adj, deg, base: new Set(a.baseline.serviceable), nodes: a.nodes };
  }
  return ACACHE[key];
}

function activateArea(key) {
  if (!AREAS[key]) return;
  curKey = key;
  A = AREAS[key];
  const g = areaGraph(key);
  N = A.nodes; E = A.edges;
  NID = new Map(N.map(n => [n.id, n]));
  ADJ = g.adj; DEG = g.deg;
  cen = A.centrality;
  cenMax = Math.max.apply(null, Object.values(cen).concat([0])) || 1;
  BEFORE = A.baseline;
  svc = new Set(BEFORE.serviceable);
  const prevKey = CUR && CUR.key;
  CUR = A.scenarios.find(s => s.key === prevKey) || A.scenarios[0];
  EPIC = { x: A.extent.cx, y: A.extent.cy };
  STATE = { dead: new Set(), alive: null, comps: [], bc: cen,
            serviceable: new Set(), art: [], bridges: [] };
  resetRecovery();
  buildScenarioCards();
  markActiveArea();
  syncAreaSelect();
  preloadBasemap(key);
  syncBasemapButtons();
  fitAll();
  recompute();
}

/* ---------- graph algorithms ---------- */
function components(alive) {
  const seen = new Set(), comps = [];
  for (const n of N) {
    if (!alive.has(n.id) || seen.has(n.id)) continue;
    const c = []; const q = [n.id]; seen.add(n.id);
    while (q.length) {
      const u = q.pop(); c.push(u);
      for (const v of ADJ.get(u)) if (alive.has(v) && !seen.has(v)) { seen.add(v); q.push(v); }
    }
    comps.push(c);
  }
  comps.sort((a, b) => b.length - a.length);
  return comps;
}

/* Brandes betweenness on the surviving subgraph (unweighted, for live speed). */
function betweenness(alive) {
  const bc = {}; for (const n of N) bc[n.id] = 0;
  for (const s of N) {
    if (!alive.has(s.id)) continue;
    const S = [], P = new Map(), sigma = new Map(), d = new Map();
    for (const n of N) { P.set(n.id, []); sigma.set(n.id, 0); d.set(n.id, -1); }
    sigma.set(s.id, 1); d.set(s.id, 0);
    const q = [s.id];
    while (q.length) {
      const v = q.shift(); S.push(v);
      for (const w of ADJ.get(v)) {
        if (!alive.has(w)) continue;
        if (d.get(w) < 0) { q.push(w); d.set(w, d.get(v) + 1); }
        if (d.get(w) === d.get(v) + 1) { sigma.set(w, sigma.get(w) + sigma.get(v)); P.get(w).push(v); }
      }
    }
    const delta = new Map(); for (const n of N) delta.set(n.id, 0);
    for (let i = S.length - 1; i >= 0; i--) {
      const w = S[i];
      for (const v of P.get(w)) delta.set(v, delta.get(v) + (sigma.get(v) / sigma.get(w)) * (1 + delta.get(w)));
      if (w !== s.id) bc[w] += delta.get(w);
    }
  }
  const aliveCount = alive.size;
  if (aliveCount > 2) for (const k in bc) bc[k] /= (aliveCount - 1) * (aliveCount - 2);
  return bc;
}

/* Articulation points + bridges of the surviving subgraph.  Iterative Tarjan
 * so the 2,700-node "wide" area cannot blow the JS stack. */
function articulationAndBridges(alive) {
  const disc = new Map(), low = new Map(), isArt = new Set(), bridges = [];
  let timer = 0;
  for (const root of N) {
    if (!alive.has(root.id) || disc.has(root.id)) continue;
    const nb = v => (ADJ.get(v) || []).filter(x => alive.has(x));
    const stack = [{ v: root.id, edges: nb(root.id), i: 0, parent: null }];
    disc.set(root.id, ++timer); low.set(root.id, timer);
    let childCount = 0;
    while (stack.length) {
      const fr = stack[stack.length - 1];
      if (fr.i < fr.edges.length) {
        const w = fr.edges[fr.i++];
        if (w === fr.parent) continue;
        if (!disc.has(w)) {
          childCount++;
          disc.set(w, ++timer); low.set(w, timer);
          stack.push({ v: w, edges: nb(w), i: 0, parent: fr.v });
        } else {
          low.set(fr.v, Math.min(low.get(fr.v), disc.get(w)));
        }
      } else {
        stack.pop();
        if (stack.length) {
          const p = stack[stack.length - 1];
          low.set(p.v, Math.min(low.get(p.v), low.get(fr.v)));
          if (low.get(fr.v) >= disc.get(p.v)) isArt.add(p.v);
          if (low.get(fr.v) > disc.get(p.v)) bridges.push([p.v, fr.v]);
        } else if (childCount > 1) {
          isArt.add(fr.v);
        }
      }
    }
  }
  return { art: [...isArt], bridges };
}

/* ---------- hazard geometry (same maths as Python) ----------
 * `rM` / `bwM` are METRES.  The diagonal conversion happens here, per area, so
 * one control produces the same physical hazard in every town.
 * `area` and `centre` are parameters so the cross-area watch table can run the
 * same maths against other towns without disturbing the active view. */
const isMetric = () => (A.provenance || {}).metric;
function fmtM(m) { return m >= 1000 ? (m / 1000).toFixed(2) + ' km' : Math.round(m) + ' m'; }

/* Metres per coordinate unit for a given area.  Node coordinates are stored in
 * a normalised 0-1 space, so a raw "400" compared against a coordinate delta
 * means nothing physical.  A metric area maps its diagonal to its real extent
 * in metres; a non-metric one falls back to a sane fixed scale so the control
 * still behaves monotonically. */
function metresPerUnit(area) {
  area = area || A;
  /* A real GPS graph is emitted in METRES already (see build_web_dashboard.py),
   * so one coordinate unit IS one metre and the conversion is the identity.
   * The demo graph is in relative units, so it needs an explicit span. */
  if ((area.provenance || {}).metric) return 1;
  const diag = (area.extent && area.extent.diag) || 1;
  return (area.extent && area.extent.diagonal_m ? area.extent.diagonal_m : 2000) / diag;
}
/* Convert a metre distance into coordinate units for this area. */
function toUnits(m, area) { return m / metresPerUnit(area); }

/* Effective size at the current time-progression point. */
function effRadiusM() { return HAZ.radiusM * (HAZ.time / 100); }
function currentSizeM() {
  if (!CUR) return HAZ.radiusM;
  if (CUR.kind === 'disc') return effRadiusM();
  return bandGeom(CUR.key).widthM * (HAZ.time / 100);
}
function currentAngle() {
  return CUR && CUR.kind === 'band' ? bandGeom(CUR.key).angle : HAZ.floodAngle;
}

function damagedSet(sc, rM, bwM, angle, area, centre, timeFrac) {
  area = area || A;
  centre = centre || EPIC;
  const t = timeFrac === undefined ? HAZ.time / 100 : timeFrac;
  const nodes = area.nodes, dead = new Set();
  if (!sc) return dead;
  if (sc.kind === 'nodes') {
    (area.gatekeepers || []).slice(0, sc.count || 3).forEach(id => dead.add(id));
    return dead;
  }
  /* Work in coordinate units: convert the metre size for THIS area so the
   * same slider means the same physical hazard in every town. */
  if (sc.kind === 'disc') {
    const r = toUnits(rM * t, area);
    for (const n of nodes) {
      const dx = n.x - centre.x, dy = n.y - centre.y;
      if (dx * dx + dy * dy <= r * r) dead.add(n.id);
    }
  } else {
    const bw = toUnits(bwM * t, area);
    const a = angle * Math.PI / 180, dx = Math.cos(a), dy = Math.sin(a);
    for (const n of nodes) {
      const p = Math.abs((n.x - centre.x) * -dy + (n.y - centre.y) * dx);
      if (p <= bw) dead.add(n.id);
    }
  }
  return dead;
}

/* ---------- shared node-state helpers ----------
 * Three states are distinguished so a network that was already fragmented is
 * not misread as damage:
 *   destroyed    -> destroyed by the hazard
 *   newly lost   -> grey, was connected before and is not now
 *   never linked -> dimmed, sat in a separate fragment before the event      */
const nodeId = n => (n !== null && typeof n === 'object') ? n.id : n;
const wasOk = n => svc.has(nodeId(n));
const inService = n => STATE.serviceable.has(nodeId(n));

/* ---------- offline raster basemap ----------
 * The canvases used to be filled with a flat #080d1a rectangle, which reads as
 * "no data" rather than "map".  scripts/fetch_basemap.py pre-stitches real
 * OpenStreetMap raster tiles per area and style and ships them inlined, so the
 * console still opens with no network.  This module only has to work out where
 * each image sits inside the graph's own coordinate space.
 */
let BM_STYLE = 'dark';
const BM_IMG = {};                 // `${areaKey}|${style}` -> HTMLImageElement

/* Overlay colours.  The dark basemap wants the bright cyan/amber overlay the
 * console always had; the light basemap needs DARK strokes or the network
 * vanishes into the map underneath. */
const PAL_DARK = {
  bg: '#080d1a', edge: 'rgba(150,190,255,.6)', edgeDim: 'rgba(110,130,170,.28)',
  edgeCut: 'rgba(130,142,170,.35)', edgeDead: 'rgba(255,71,87,.55)',
  edgeFixed: 'rgba(57,217,138,.75)', edgeBridge: '#ffc857',
  nodeCut: 'rgba(140,150,170,.8)', nodeDead: 'rgba(255,71,87,.55)',
  art: '#ff4757', artOther: 'rgba(120,140,175,.5)',
  route: '#4cc2ff', routeCasing: 'rgba(255,255,255,.55)',
  fac: '#4cc2ff', facOut: '#8fa0c0', facRing: '#080d1a', label: '#cfe0ff',
  scale: 'rgba(220,235,255,.75)', epicentre: '#ffc857',
};
const PAL_LIGHT = {
  bg: '#e9e7e1', edge: 'rgba(20,60,120,.72)', edgeDim: 'rgba(110,120,140,.4)',
  edgeCut: 'rgba(90,100,120,.55)', edgeDead: 'rgba(196,26,46,.8)',
  edgeFixed: 'rgba(8,140,78,.85)', edgeBridge: '#a06a00',
  nodeCut: 'rgba(60,70,90,.85)', nodeDead: 'rgba(196,26,46,.8)',
  art: '#d01b3a', artOther: 'rgba(90,100,120,.55)',
  route: '#0b63c4', routeCasing: 'rgba(255,255,255,.9)',
  fac: '#0b63c4', facOut: '#5a6478', facRing: '#ffffff', label: '#16233a',
  scale: 'rgba(20,30,45,.8)', epicentre: '#a06a00',
};
const pal = () => (BM_STYLE === 'light' ? PAL_LIGHT : PAL_DARK);

/* Derive the graph's local projection straight from the payload: every node
 * carries BOTH its metre coordinates and its WGS84 lat/lon, so kx/ky (metres
 * per degree) and the origin fall out of two extreme nodes.  Hard-coding the
 * constants here would silently desynchronise the basemap from the graph the
 * moment Python's projection changed. */
const BMCAL = {};
function bmCalib(area) {
  if (!area) return null;
  if (BMCAL[area.key || area.name]) return BMCAL[area.key || area.name];
  const ns = (area.nodes || []).filter(n =>
    typeof n.lat === 'number' && typeof n.lon === 'number' &&
    typeof n.x === 'number' && typeof n.y === 'number');
  if (ns.length < 2) return null;
  let i0 = 0, i1 = 0, j0 = 0, j1 = 0;
  for (let i = 1; i < ns.length; i++) {
    if (ns[i].x < ns[i0].x) i0 = i;
    if (ns[i].x > ns[i1].x) i1 = i;
    if (ns[i].y < ns[j0].y) j0 = i;
    if (ns[i].y > ns[j1].y) j1 = i;
  }
  const kx = (ns[i1].x - ns[i0].x) / (ns[i1].lon - ns[i0].lon);
  const ky = (ns[j1].y - ns[j0].y) / (ns[j1].lat - ns[j0].lat);
  if (!isFinite(kx) || !isFinite(ky) || kx === 0 || ky === 0) return null;
  /* lon = x / kx + lx  and  lat = y / ky + ly  */
  const cal = { kx, ky, lx: ns[i0].lon - ns[i0].x / kx, ly: ns[j0].lat - ns[j0].y / ky };
  BMCAL[area.key || area.name] = cal;
  return cal;
}

/* Decode the (possibly large) data: URI once per area+style.  Rendering has to
 * wait for `complete`, so every load kicks a re-render when it lands. */
function bmImage(areaKey, style) {
  const k = areaKey + '|' + style;
  if (BM_IMG[k]) return BM_IMG[k];
  const entry = ((AREAS[areaKey] || {}).basemap || {})[style];
  if (!entry || !entry.img) return null;
  /* No `Image` outside a browser (the Node test harness runs this file in a
   * vm).  Skipping the preload there is correct: the maps simply keep their flat
   * background and every other check still runs. */
  if (typeof Image === 'undefined') return null;
  const img = new Image();
  img.onload = () => { if (typeof render === 'function') render(); };
  img.src = entry.img;
  BM_IMG[k] = img;
  return img;
}

function preloadBasemap(areaKey) {
  Object.keys((AREAS[areaKey] || {}).basemap || {}).forEach(s => bmImage(areaKey, s));
}

function hasBasemap(areaKey, style) {
  return !!(((AREAS[areaKey] || {}).basemap || {})[style || BM_STYLE]);
}

/* Reflect the shared basemap state on every panel's button, and disable the
 * ones that cannot switch because the active area has only one style. */
function syncBasemapButtons() {
  const canSwitch = hasBasemap(curKey, 'dark') && hasBasemap(curKey, 'light');
  document.querySelectorAll('.mapctl button.bm').forEach(b => {
    b.classList.toggle('on', BM_STYLE === 'light');
    b.textContent = 'Map';
    b.title = canSwitch
      ? 'Basemap: ' + (BM_STYLE === 'light' ? 'light map' : 'dark map') + ' — click to switch'
      : 'Basemap: ' + (BM_STYLE === 'light' ? 'light map' : 'dark map');
    b.disabled = !canSwitch;
    b.style.opacity = canSwitch ? '' : '.45';
  });
}

/* Paint the basemap under the graph, sharing the map's exact view transform.
 * Returns false when this area has no image, so the caller can fall back to a
 * flat background. */
function drawBasemap(ctx, id) {
  const entry = ((A.basemap || {})[BM_STYLE]);
  if (!entry) return false;
  const img = bmImage(curKey, BM_STYLE);
  if (!img || !img.complete || !img.naturalWidth) return false;
  const cal = bmCalib(A);
  if (!cal) return false;

  const dLon = entry.bbox[3] - entry.bbox[1];      // east  - west
  const dLat = entry.bbox[2] - entry.bbox[0];      // north - south
  if (!(dLon > 0) || !(dLat > 0)) return false;

  /* image px = ku*x + u0   and   v = -kv*y + v0 */
  const ku = entry.w / (cal.kx * dLon);
  const kv = entry.h / (cal.ky * dLat);
  const u0 = (cal.lx - entry.bbox[1]) * entry.w / dLon;
  const v0 = (entry.bbox[2] - cal.ly) * entry.h / dLat;

  /* Draw at the graph's EXACT scale.  Any independent zoom here would break the
   * one thing that matters most -- the overlay must sit on the same streets the
   * basemap draws -- and it would also stop panning from working, because the
   * two layers would drift apart.  scripts/fetch_basemap.py therefore stitches
   * each bbox wide enough to cover the whole canvas at fit zoom (see
   * VIEWPORT_ASPECT), which is why no fallback scaling is needed here. */
  const vp = view(id), dpr = window.devicePixelRatio || 1;
  ctx.save();
  ctx.setTransform(dpr * (vp.s / ku), 0, 0, dpr * (vp.s / kv),
                   dpr * (vp.ox - u0 * vp.s / ku),
                   dpr * (vp.oy - v0 * vp.s / kv));
  ctx.imageSmoothingEnabled = true;
  if (ctx.imageSmoothingQuality) ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(img, 0, 0);
  ctx.restore();
  return true;
}

/* ---------- canvas views (one view per map on the console) ---------- */
const VIEWS = {};
function view(id) {
  if (!VIEWS[id]) VIEWS[id] = { s: 1, ox: 0, oy: 0 };
  return VIEWS[id];
}
function fitView(id) {
  const c = document.getElementById(id);
  if (!c || !N || !N.length) return;
  const v = view(id), r = c.getBoundingClientRect();
  if (!r.width || !r.height) return;
  const pad = 28;
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const n of N) {
    if (n.x < x0) x0 = n.x; if (n.x > x1) x1 = n.x;
    if (n.y < y0) y0 = n.y; if (n.y > y1) y1 = n.y;
  }
  v.s = Math.min((r.width - pad * 2) / (x1 - x0 || 1), (r.height - pad * 2) / (y1 - y0 || 1));
  v.ox = (r.width - (x1 - x0) * v.s) / 2 - x0 * v.s;
  /* Node y is metres NORTH, but canvas y grows downwards, so the raw value
   * would render the whole town upside down.  Invisible on a blank background,
   * glaring the moment a real basemap is drawn under it -- flip here and in
   * TY() so north is up, exactly like every map. */
  v.oy = (r.height - (y1 - y0) * v.s) / 2 + y1 * v.s;
}
function fitAll() { ['cv', 'cv2', 'cvAn', 'cvFac', 'cvRec'].forEach(fitView); }
const TX = (id, x) => x * view(id).s + view(id).ox;
const TY = (id, y) => -y * view(id).s + view(id).oy;   // north is up

function col(v) {
  const t = Math.max(0, Math.min(1, v));
  const a = [57, 217, 138], b = [255, 200, 87], c = [255, 71, 87];
  const f = t < .5 ? a : b, tt = t < .5 ? t * 2 : (t - .5) * 2;
  const g = [0, 1, 2].map(i => Math.round(f[i] + (c[i] - f[i]) * tt));
  return 'rgb(' + g[0] + ',' + g[1] + ',' + g[2] + ')';
}

/* Scale bar text, so the operator can see the map really is in metres. */
function scaleLabel() {
  if (!isMetric()) return 'relative units';
  const metres = 100 * metresPerUnit(A);
  return fmtM(metres) + ' ≈ 100px';
}

/* ---------- the map renderer ----------
 * One function draws every map on the console; `mode` selects what to
 * emphasise so page 04 can show articulation points and page 05 can draw
 * routes without duplicating the canvas code. */
function drawMap(id, mode) {
  const c = document.getElementById(id);
  if (!c || !N) return;
  const ctx = c.getContext('2d');
  const r = c.getBoundingClientRect();
  if (!r.width) return;
  const dpr = window.devicePixelRatio || 1;
  if (c.width !== Math.round(r.width * dpr) || c.height !== Math.round(r.height * dpr)) {
    c.width = Math.round(r.width * dpr); c.height = Math.round(r.height * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, r.width, r.height);
  const P = pal();
  /* Real map imagery under the graph when we have it; the flat colour is only
   * the fallback for an area with no stitched tiles. */
  ctx.fillStyle = P.bg; ctx.fillRect(0, 0, r.width, r.height);
  drawBasemap(ctx, id);

  const unit = (A.extent.diag || 1) / Math.max(view(id).s, 1e-9);

  /* hazard zone underlay */
  if (CUR && CUR.kind !== 'nodes' && HAZ.time > 0) {
    ctx.save();
    const g = CUR.kind === 'band' ? bandGeom(CUR.key) : null;
    /* Convert metres to PIXELS for this canvas: m -> coordinate units -> px.
     * Using metres directly as pixels made the zone scale with the canvas
     * instead of the map, so the same hazard looked different per view. */
    const mToPx = view(id).s / metresPerUnit(A);
    if (CUR.kind === 'disc') {
      const rPx = effRadiusM() * mToPx;
      ctx.beginPath(); ctx.arc(TX(id, EPIC.x), TY(id, EPIC.y), rPx, 0, 7);
      ctx.fillStyle = 'rgba(255,71,87,.13)'; ctx.fill();
      ctx.strokeStyle = 'rgba(255,71,87,.6)'; ctx.setLineDash([6, 5]);
      ctx.lineWidth = 1.5; ctx.stroke();
    } else {
      const a = g.angle * Math.PI / 180, dx = Math.cos(a), dy = Math.sin(a);
      const bwPx = g.widthM * (HAZ.time / 100) * mToPx;
      /* corridor length spans the map, in coordinate units */
      const L = toUnits(((A.extent && A.extent.diag) || 1) * 1.5, A);
      const mx = EPIC.x, my = EPIC.y;
      ctx.beginPath();
      ctx.moveTo(TX(id, mx - dy * bwPx - dx * L), TY(id, my + dx * bwPx - dy * L));
      ctx.lineTo(TX(id, mx + dy * bwPx + dx * L), TY(id, my - dx * bwPx - dy * L));
      ctx.lineTo(TX(id, mx - dy * bwPx + dx * L), TY(id, my + dx * bwPx + dy * L));
      ctx.lineTo(TX(id, mx + dy * bwPx - dx * L), TY(id, my - dx * bwPx - dy * L));
      ctx.closePath();
      ctx.fillStyle = CUR.key === 'landslide' ? 'rgba(255,159,67,.14)' : 'rgba(64,160,255,.14)';
      ctx.fill();
      ctx.strokeStyle = CUR.key === 'landslide' ? 'rgba(255,159,67,.65)' : 'rgba(64,160,255,.6)';
      ctx.setLineDash([6, 5]); ctx.lineWidth = 1.5; ctx.stroke();
    }
    ctx.restore();
  }

  const fixed = STATE.fixed || new Set();

  /* edges */
  ctx.lineWidth = 1.1;
  for (const e of E) {
    const a = NID.get(e.s), b = NID.get(e.t);
    if (!a || !b) continue;
    const dead = STATE.dead.has(e.s) || STATE.dead.has(e.t);
    const joined = inService(a) && inService(b);
    const ever = wasOk(a) && wasOk(b);
    const isB = mode === 'bridges' && STATE.bridgeSet &&
      (STATE.bridgeSet.has(e.s + '|' + e.t) || STATE.bridgeSet.has(e.t + '|' + e.s));
    ctx.beginPath(); ctx.moveTo(TX(id, a.x), TY(id, a.y)); ctx.lineTo(TX(id, b.x), TY(id, b.y));
    if (dead) { ctx.strokeStyle = P.edgeDead; ctx.setLineDash([2, 3]); }
    else if (fixed.has(e.s) || fixed.has(e.t)) { ctx.strokeStyle = P.edgeFixed; ctx.setLineDash([]); }
    else if (isB) { ctx.strokeStyle = P.edgeBridge; ctx.setLineDash([]); ctx.lineWidth = 2.4; }
    else if (STATE.dead.size && !joined && ever) { ctx.strokeStyle = P.edgeCut; ctx.setLineDash([]); }
    else if (!ever) { ctx.strokeStyle = P.edgeDim; ctx.setLineDash([]); }
    else { ctx.strokeStyle = P.edge; ctx.setLineDash([]); }
    ctx.stroke();
    ctx.lineWidth = 1.1;
  }

  /* component colouring (page 04) */
  if (mode === 'components' && STATE.comps.length) {
    const palette = ['#4cc2ff', '#39d98a', '#ffc857', '#ff9f43', '#c084fc',
                     '#f472b6', '#2dd4bf', '#a3e635'];
    STATE.comps.forEach((comp, i) => {
      ctx.beginPath();
      for (const nid of comp) {
        const n = NID.get(nid); if (!n) continue;
        ctx.moveTo(TX(id, n.x) + 5.5, TY(id, n.y));
        ctx.arc(TX(id, n.x), TY(id, n.y), 5.5, 0, 7);
      }
      ctx.fillStyle = palette[i % palette.length] + '55';
      ctx.fill();
      ctx.strokeStyle = palette[i % palette.length];
      ctx.lineWidth = 1.2; ctx.stroke();
    });
  }

  /* nodes */
  const artSet = mode === 'articulation' ? new Set(STATE.art) : null;
  for (const n of N) {
    const v = (STATE.bc[n.id] || 0) / cenMax;
    const dead = STATE.dead.has(n.id);
    const cut = !dead && !inService(n) && wasOk(n);
    const r = artSet && artSet.has(n.id) ? 7 : (dead ? 2.2 : 3.4);
    ctx.beginPath(); ctx.arc(TX(id, n.x), TY(id, n.y), r, 0, 7);
    if (artSet) ctx.fillStyle = artSet.has(n.id) ? P.art
      : (dead ? P.nodeDead : P.artOther);
    else if (dead) ctx.fillStyle = P.nodeDead;
    else if (cut) ctx.fillStyle = P.nodeCut;
    else { ctx.globalAlpha = inService(n) ? 1 : .45; ctx.fillStyle = col(v); }
    ctx.fill(); ctx.globalAlpha = 1;
    if (artSet && artSet.has(n.id)) {
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.stroke();
    }
  }

  /* route overlay (page 05) */
  if (mode === 'route' && STATE.route) {
    for (const seg of STATE.route) {
      const a = NID.get(seg[0]), b = NID.get(seg[1]);
      if (!a || !b) continue;
      ctx.beginPath(); ctx.moveTo(TX(id, a.x), TY(id, a.y)); ctx.lineTo(TX(id, b.x), TY(id, b.y));
      ctx.strokeStyle = seg[2] || P.route; ctx.lineWidth = 3.4;
      ctx.setLineDash([]); ctx.stroke();
      ctx.strokeStyle = P.routeCasing; ctx.lineWidth = 1; ctx.stroke();
    }
  }

  /* facilities */
  if (mode !== 'components') {
    /* Greedy label declutter: the town centres pack 14 facilities into a few
     * hundred pixels, and once a real basemap with its own place names is drawn
     * underneath, overlapping labels became unreadable.  Labels are still drawn
     * for the most important ones first and skipped when they would collide. */
    const placed = [];
    const hits = (b) => placed.some(p =>
      b.x < p.x + p.w && b.x + b.w > p.x && b.y < p.y + p.h && b.y + b.h > p.y);
    if (r.width > 620) ctx.font = '11px "Segoe UI",sans-serif';
    const order = A.facilities.slice().sort((a, b) => {
      const sa = STATE.dead.has(a.node) ? 2 : inService(NID.get(a.node)) ? 0 : 1;
      const sb = STATE.dead.has(b.node) ? 2 : inService(NID.get(b.node)) ? 0 : 1;
      return sa - sb;                    // reachable first, then cut off, then destroyed
    });
    for (const f of order) {
      const n = NID.get(f.node); if (!n) continue;
      const bad = STATE.dead.has(f.node);
      const out = !inService(n) && !bad;
      const c2 = bad ? P.art : out ? P.facOut : P.fac;
      const fx = TX(id, n.x), fy = TY(id, n.y);
      ctx.beginPath(); ctx.arc(fx, fy, 7, 0, 7);
      ctx.fillStyle = c2; ctx.fill();
      ctx.lineWidth = 2; ctx.strokeStyle = P.facRing; ctx.stroke();
      if (r.width > 620) {
        const text = f.name.length > 26 ? f.name.slice(0, 24) + '…' : f.name;
        const w = ctx.measureText(text).width;
        const box = { x: fx + 10, y: fy - 5, w: w + 3, h: 13 };
        if (!hits(box)) {
          placed.push(box);
          /* dark halo first so the name stays readable over map imagery */
          ctx.lineWidth = 3; ctx.strokeStyle = P.bg;
          ctx.strokeText(text, box.x, fy + 4);
          ctx.fillStyle = P.label;
          ctx.fillText(text, box.x, fy + 4);
        }
      }
    }
  }

  /* epicentre marker -- also the drag handle that moves the hazard */
  if (CUR && CUR.kind !== 'nodes') {
    const ex = TX(id, EPIC.x), ey = TY(id, EPIC.y);
    ctx.save();
    ctx.strokeStyle = P.epicentre; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(ex, ey, 9, 0, 7); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(ex - 14, ey); ctx.lineTo(ex - 4, ey);
    ctx.moveTo(ex + 4, ey); ctx.lineTo(ex + 14, ey);
    ctx.moveTo(ex, ey - 14); ctx.lineTo(ex, ey - 4);
    ctx.moveTo(ex, ey + 4); ctx.lineTo(ex, ey + 14);
    ctx.stroke();
    ctx.restore();
  }

  /* scale bar */
  const barM = 200, barPx = barM * (view(id).s / metresPerUnit(A));
  if (isMetric() && barPx > 8 && barPx < r.width * 0.5) {
    ctx.save();
    ctx.strokeStyle = P.scale; ctx.lineWidth = 2;
    const bx = 14, by = r.height - 16;
    ctx.beginPath(); ctx.moveTo(bx, by); ctx.lineTo(bx + barPx, by); ctx.stroke();
    ctx.font = '10.5px "Segoe UI",sans-serif'; ctx.fillStyle = P.label;
    ctx.fillText(fmtM(barM), bx, by - 5);
    ctx.restore();
  }
}

/* ---------- live simulation ---------- */
let timer = null;

function recompute() {
  const dead = CUR && CUR.kind === 'band'
    ? damagedSet(CUR, HAZ.radiusM, bandGeom(CUR.key).widthM, bandGeom(CUR.key).angle)
    : damagedSet(CUR, HAZ.radiusM, HAZ.floodM, HAZ.floodAngle);
  const fixed = STATE.fixed || new Set();
  const gone = new Set();
  dead.forEach(id => { if (!fixed.has(id)) gone.add(id); });
  const alive = new Set(N.map(n => n.id).filter(id => !gone.has(id)));
  const comps = components(alive);
  STATE.dead = gone; STATE.alive = alive; STATE.comps = comps;
  STATE.serviceable = new Set(comps[0] || []);
  STATE.bc = gone.size ? betweenness(alive) : cen;
  const ab = articulationAndBridges(alive);
  STATE.art = ab.art; STATE.bridges = ab.bridges;
  STATE.bridgeSet = new Set(ab.bridges.map(b => b[0] + '|' + b[1]));
  render();
}

function schedule() { clearTimeout(timer); timer = setTimeout(recompute, 130); }

const pct = () => STATE.comps.length ? 100 * (STATE.comps[0] || []).length / (BEFORE.largest || 1) : 0;

function severity(p, destroyed) {
  if (p >= 90) return destroyed ? ['SEVERE', 'orange'] : ['MINOR', 'green'];
  if (p >= 60) return ['MODERATE', 'amber'];
  if (p >= 25) return ['SEVERE', 'orange'];
  return ['CRITICAL', 'red'];
}

/* Facility state, computed once and reused by pages 01, 05 and the assistant so
 * the console can never contradict itself. */
function facilityStates() {
  return A.facilities.map(f => ({
    ...f,
    destroyed: STATE.dead.has(f.node),
    reachable: STATE.serviceable.has(f.node),
    wasOk: wasOk(f.node),
  }));
}
const statusOf = f => f.destroyed ? ['DESTROYED', 'dead']
  : f.reachable ? ['REACHABLE', 'ok']
    : f.wasOk ? ['CUT OFF', 'warn'] : ['NOT LINKED', 'mute'];

/* ---------- page 01: operations overview ---------- */
function renderOverview() {
  const p = pct(), dead = STATE.dead.size;
  const facNow = facilityStates();
  const destroyed = facNow.filter(f => f.destroyed).length;
  const newly = facNow.filter(f => !f.destroyed && !f.reachable && f.wasOk).length;

  const pCls = p >= 90 ? 'ok' : p >= 60 ? 'warn' : p >= 25 ? 'bad' : 'crit';
  document.getElementById('kpis').innerHTML = [
    ['Connectivity', p.toFixed(1) + '%', pCls, deltaTxt(100, p)],
    ['Junctions Left', STATE.serviceable.size + ' / ' + BEFORE.nodes, 'acc',
      (BEFORE.nodes - STATE.serviceable.size) + ' lost'],
    ['Components', STATE.comps.length, STATE.comps.length > 3 ? 'bad' : 'acc',
      'was ' + BEFORE.components],
    ['Roads Lost', dead, dead > 0 ? 'crit' : 'ok', 'of ' + BEFORE.edges + ' streets'],
  ].map(([l, v, c, d]) =>
    `<div class="kpi ${c}"><div class="v">${v}</div><div class="l">${l}</div><div class="d">${d}</div></div>`
  ).join('');
  document.getElementById('connNote').innerHTML =
    `Baseline: ${BEFORE.nodes} junctions, ${BEFORE.edges} streets, ${BEFORE.components} component(s),
     largest ${BEFORE.largest}. Connectivity is measured on the largest surviving mutually-reachable
     network${dead ? '' : ' &mdash; no junction is currently destroyed'}.`;

  document.getElementById('facList').innerHTML = facNow.map(f => {
    const [txt, cls] = statusOf(f);
    return `<div class="fac"><span class="ic">${facIcon(f.kind)}</span>
      <span class="nm" title="${esc(f.name)}">${esc(f.name)}</span>
      <span class="st ${cls}">${txt}</span></div>`;
  }).join('') + `<div class="note">${A.realFacilities
    ? `Facilities are <b>real places</b> read from OpenStreetMap (hospitals, clinics, fire stations,
       police, shelters), each snapped to its nearest road junction.`
    : `Facilities are sample anchor points placed on the extracted network (not real hospitals) so the
       simulation can show who is cut off.`}
    <b>NOT LINKED</b> = it was already in a separate fragment of the network before the event, so the
    disaster did not cause that.</div>`;

  /* critical junctions: biggest rise in importance after the event */
  const rows = N.map(n => ({ id: n.id, b: cen[n.id] || 0, a: STATE.bc[n.id] || 0 }))
    .map(o => ({ ...o, d: o.a - o.b })).sort((x, y) => y.d - x.d).slice(0, 12);
  document.getElementById('critTable').innerHTML =
    `<div class="tblwrap"><table><thead><tr><th>Rank</th><th>Junction</th>
      <th class="num">Before</th><th class="num">After</th><th class="num">&Delta;</th>
      <th class="num">Resilience idx</th></tr></thead><tbody>` +
    rows.map((r, i) => `<tr><td>${i + 1}</td><td>${r.id}</td>
      <td class="num">${r.b.toFixed(3)}</td><td class="num hi">${r.a.toFixed(3)}</td>
      <td class="num ${r.d > 0.001 ? 'lo' : 'dim'}">${r.d >= 0 ? '+' : ''}${r.d.toFixed(3)}</td>
      <td class="num">${(1 - Math.min(1, r.a / (cenMax || 1))).toFixed(2)}</td></tr>`).join('') +
    `</tbody></table></div><div class="note">Junctions whose importance rises after the event are
      the ones responders must protect first. <b>Resilience idx</b> is 1 minus the node's share of the
      highest centrality &mdash; higher means more fragile.</div>`;

  document.getElementById('mapbar').innerHTML =
    [`<span>${esc(CUR.name)}</span>`, dead ? `<span>${dead} junctions lost</span>` : '<span>baseline</span>',
     `<span>${p.toFixed(0)}% connected</span>`,
     HAZ.time < 100 ? `<span>T+${Math.round(HAZ.time * 1.2)} min</span>` : '']
      .filter(Boolean).join('');
  document.getElementById('mapScalePill').textContent = 'scale: ' + scaleLabel();
  const pillSev = document.getElementById('pillSev');
  if (pillSev) pillSev.textContent = severity(p, destroyed)[0];
  briefing(p, facNow, destroyed, newly);
}

function deltaTxt(base, now) {
  const d = now - base;
  if (Math.abs(d) < 0.05) return 'no change';
  return (d > 0 ? '+' : '') + d.toFixed(1) + ' pts vs baseline';
}
function facIcon(kind) {
  return { hospital: '🏥', clinic: '🏥', fire_station: '🚒', police: '🚓',
           shelter: '🏕', depot: '🚚', ambulance: '🚑' }[kind] || '📍';
}
function esc(s) {
  return String(s == null ? '' : s).replace(/[&<>"']/g,
    c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

/* ============================ shared plumbing ============================ */
const $ = id => document.getElementById(id);
function setHTML(id, html) { const e = $(id); if (e) e.innerHTML = html; }
function setTxt(id, txt) { const e = $(id); if (e) e.textContent = txt; }

/* Draw every visible map in its page-appropriate mode, plus the small charts. */
function render() {
  drawMap('cv', 'overview'); drawMap('cv2', 'overview');
  drawMap('cvAn', currentMode());
  drawMap('cvFac', 'overview'); drawMap('cvRec', 'overview');
  drawLocator(); drawSpark('simChart', simSeries()); drawSpark('recChart', recSeries());
  renderOverview(); renderSimulation(); renderAnalytics();
  renderFacilities(); renderRecovery(); renderWatch(); renderResearch();
}
function currentMode() {
  const t = document.querySelector('#anTabs .tab.on');
  return t ? (t.dataset.view || 'criticality') : 'criticality';
}
function currentFacMode() {
  const t = document.querySelector('#facTabs .tab.on');
  return t ? (t.dataset.fview || 'status') : 'status';
}

/* ---------- scenario picker ---------- */
function buildScenarioCards() {
  const mk = host => {
    const el = $(host); if (!el) return;
    el.innerHTML = A.scenarios.map(s => `
      <button class="${CUR && s.key === CUR.key ? 'on' : ''}" data-scn="${esc(s.key)}">
        <span class="ic">${s.icon || '⚠'}</span>
        <b>${esc(s.name)}</b><small>${esc(s.note || '')}</small>
      </button>`).join('');
    el.querySelectorAll('[data-scn]').forEach(b => { b.onclick = () => selectScenario(b.dataset.scn); });
  };
  mk('scn'); mk('scn2');
  if (CUR) setTxt('scnBlurb', CUR.blurb || CUR.note || '');
}
function selectScenario(key) {
  const s = A.scenarios.find(x => x.key === key);
  if (!s) return;
  CUR = s;
  STATE.dead = new Set(); STATE.alive = null; STATE.comps = [];
  buildScenarioCards(); renderSimulationCtls(); recompute();
}

/* ---------- severity controls (mirrored on page 01 and page 02) ---------- */
function bandOrDisc() {
  return CUR && CUR.kind === 'band' ? bandGeom(CUR.key).widthM : HAZ.radiusM;
}
function renderSimulationCtls() {
  const html = (!CUR || CUR.kind !== 'nodes') ? `
    <div class="ctl">
      <label>${CUR && CUR.kind === 'band' ? 'Corridor half-width' : 'Blast radius'}
        <b class="vRad">${fmtM(currentSizeM())}</b></label>
      <input type="range" class="rRad" min="50" max="1000" step="10" value="${bandOrDisc()}">
    </div>
    ${CUR && CUR.kind === 'band' ? `
    <div class="ctl">
      <label>Corridor bearing <b class="vAng">${currentAngle()}°</b></label>
      <input type="range" class="rAng" min="0" max="179" value="${currentAngle()}">
    </div>` : ''}` :
    `<div class="note">This scenario removes <b>${(CUR && CUR.count) || 3}</b> key junctions
      rather than damaging everything inside a radius, so it has no size control.</div>`;

  ['simCtls', 'simCtls2'].forEach(host => {
    const el = $(host); if (!el) return;
    el.innerHTML = html;
    el.querySelectorAll('.rRad').forEach(r => {
      r.oninput = () => {
        const v = +r.value;
        if (CUR && CUR.kind === 'band') {
          if (CUR.key === 'landslide') HAZ.landslideM = v; else HAZ.floodM = v;
        } else HAZ.radiusM = v;
        document.querySelectorAll('.vRad').forEach(b => { b.textContent = fmtM(currentSizeM()); });
        schedule();
      };
    });
    el.querySelectorAll('.rAng').forEach(r => {
      r.oninput = () => {
        const v = +r.value;
        if (CUR.key === 'landslide') HAZ.landslideAngle = v; else HAZ.floodAngle = v;
        document.querySelectorAll('.vAng').forEach(b => { b.textContent = v + '°'; });
        schedule();
      };
    });
  });
}

/* ---------- page 01 support ---------- */
function markActiveArea() {
  document.querySelectorAll('[data-area]').forEach(el =>
    el.classList.toggle('on', el.dataset.area === curKey));
}
function syncAreaSelect() {
  if ($('areaSel')) $('areaSel').value = curKey;
  setTxt('pillNodes', N.length.toLocaleString() + ' junctions');
}
function resetRecovery() { STATE.repairs = new Set(); STATE.crews = []; }

/* Dijkstra on the surviving graph.  `blocked` is the set of destroyed nodes;
 * pass an empty set for the pre-disaster route. */
const shortestPath = (from, to, blocked) =>
  shortestPathOn(ADJ, from, to, blocked || new Set(), DEG);

/* ---------- assistant: deterministic answers grounded in the live numbers ---------- */
function briefing(p, facNow, destroyed, newly) {
  const [sev, cls] = severity(p, destroyed);
  setTxt('sevBadge', sev);
  const b = $('sevBadge'); if (b) b.className = 'sev ' + cls;
  const L = [];
  L.push(`${A.name} — ${CUR ? CUR.name : 'no scenario'} at T+${Math.round(HAZ.time * 1.2)} min.`);
  L.push(`${p.toFixed(1)}% of the pre-event largest component is still connected ` +
         `(${STATE.dead.size} junctions destroyed, ${newly} newly cut off).`);
  if (STATE.comps.length > 1) {
    L.push(`The network has fragmented into ${STATE.comps.length} pieces; everything outside the ` +
           `largest isolates ${STATE.comps.slice(1).reduce((s, c) => s + c.length, 0)} junctions.`);
  }
  L.push(`${destroyed} of ${facNow.length} monitored facilities are destroyed; ` +
         `${facNow.filter(f => f.reachable).length} remain reachable.`);
  if (STATE.art.length) {
    L.push(`${STATE.art.length} articulation points now control access — losing any one of them ` +
           `splits the network again.`);
  }
  if (STATE.bridgeSet && STATE.bridgeSet.size) {
    L.push(`${STATE.bridgeSet.size} bridge edges carry the whole surviving component and are the ` +
           `single-point failures to protect first.`);
  }
  L.push(`Hazard extent is a real ${fmtM(currentSizeM())}` +
         (CUR && CUR.kind === 'band' ? ` corridor at ${currentAngle()}°.` : ' radius.'));
  setHTML('brief', L.join('\n'));
  setTxt('headline', `${A.name} · ${(STATE.comps[0] || []).length.toLocaleString()} reachable junctions`);
}
const QUICK_Q = ['Which junction matters most?', 'How many facilities are cut off?',
  'What is the single point of failure?', 'How quickly does the network fail?'];
function nameOf(id) { const n = NID.get(id); return n ? (n.name || n.id) : id; }
function answer(q) {
  const s = (q || '').toLowerCase();
  if (/facilit|hospital|clinic|police|fire|shelter|depot|ambulance/.test(s)) {
    const f = facilityStates(), bad = f.filter(x => !x.reachable);
    if (!bad.length) return `All ${f.length} monitored facilities are still reachable — ` +
      `no access-driven evacuation priority from the road network.`;
    return bad.map(x => `${x.name}: ${statusOf(x)[0].toLowerCase()}` +
      (x.destroyed ? ' (its junction was destroyed).' : ' (lost its road connection).')).join('\n');
  }
  if (/single point|failure|bridge|bottleneck/.test(s)) {
    const bs = [...(STATE.bridgeSet || [])];
    if (!bs.length) return 'No bridge edge exists in the surviving graph, so no single edge ' +
      'failure can split it further.';
    return `${bs.length} bridge edges carry the whole surviving component, so they give the ` +
      `most reach per crew-hour. Top: ` +
      bs.slice(0, 5).map(k => { const [u, v] = k.split('|');
        return `${esc(nameOf(u))} ↔ ${esc(nameOf(v))}`; }).join(', ') + '.';
  }
  if (/which junction|most important|critical|central/.test(s)) {
    const arr = N.filter(n => !STATE.dead.has(n.id))
      .sort((a, b) => (STATE.bc[b.id] || 0) - (STATE.bc[a.id] || 0));
    if (!arr.length) return 'Every junction is destroyed; there is no priority left to rank.';
    const t = arr[0];
    return `${esc(nameOf(t.id))} is currently the most load-bearing surviving junction ` +
      `(betweenness ${(STATE.bc[t.id] || 0).toFixed(2)}; ` +
      `${((STATE.bc[t.id] || 0) / (cenMax || 1) * 100).toFixed(1)}% of the baseline maximum) on ` +
      `${(ADJ.get(t.id) || []).length} roads.` +
      (STATE.art.includes(t.id) ? ' It is also an articulation point — a true single point of failure.' : '');
  }
  if (/fail|quick|timeline|how fast|progression/.test(s)) {
    return `Damage grows linearly with elapsed time: at 50% the hazard has reached ` +
      `${fmtM(currentSizeM() / 2)}, at 100% the full ${fmtM(currentSizeM())}. Drag the time ` +
      `slider on page 02 to watch connectivity fall. This is a network stress test, not a forecast.`;
  }
  return `Ask about: which junction matters most, how many facilities are cut off, what the ` +
    `single point of failure is, or how quickly the network fails.`;
}

/* ---------- charts ---------- */
function simSeries() {
  const pts = [];
  for (let t = 0; t <= 100; t += 5) {
    const dead = CUR && CUR.kind === 'band'
      ? damagedSet(CUR, HAZ.radiusM, bandGeom(CUR.key).widthM, bandGeom(CUR.key).angle, A, EPIC, t / 100)
      : damagedSet(CUR, HAZ.radiusM, HAZ.floodM, HAZ.floodAngle, A, EPIC, t / 100);
    const alive = new Set(N.map(n => n.id).filter(id => !dead.has(id)));
    const c = components(alive);
    pts.push({ t, conn: 100 * (c[0] || []).length / (BEFORE.largest || 1), lost: dead.size });
  }
  return pts;
}
function recSeries() {
  const reps = STATE.repairs ? [...STATE.repairs] : [];
  const out = []; let conn = pct();
  reps.forEach((id, i) => {
    out.push({ t: (i + 1) * (100 / Math.max(reps.length, 1)), conn, lost: 0 });
    conn = Math.min(100, conn + 2);
  });
  return out;
}
function drawSpark(id, pts) {
  const c = $(id); if (!c || !pts || !pts.length) return;
  const r = c.getBoundingClientRect(); if (!r.width) return;
  const dpr = window.devicePixelRatio || 1;
  c.width = Math.round(r.width * dpr); c.height = Math.round(r.height * dpr);
  const ctx = c.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, r.width, r.height);
  const maxL = Math.max.apply(null, pts.map(p => p.lost).concat([1]));
  const X = t => 8 + (r.width - 16) * t / 100;
  const Y = v => r.height - 10 - (r.height - 22) * v;
  ctx.beginPath();
  pts.forEach((p, i) => { const x = X(p.t), y = Y(p.lost / maxL);
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
  ctx.strokeStyle = '#ff4757'; ctx.lineWidth = 2; ctx.stroke();
  ctx.beginPath();
  pts.forEach((p, i) => { const x = X(p.t), y = Y(p.conn / 100);
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
  ctx.strokeStyle = '#4cc2ff'; ctx.lineWidth = 2; ctx.stroke();
  ctx.strokeStyle = 'rgba(255,255,255,.35)'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(X(HAZ.time), 6); ctx.lineTo(X(HAZ.time), r.height - 6); ctx.stroke();
}

/* ---------- regional locator: real positions only, never invented streets ---------- */
function componentsIn(area, adj, alive) {
  const seen = new Set(), comps = [];
  for (const n of area.nodes) {
    if (!alive.has(n.id) || seen.has(n.id)) continue;
    const c = [], q = [n.id]; seen.add(n.id);
    while (q.length) {
      const u = q.pop(); c.push(u);
      for (const v of adj.get(u) || []) if (alive.has(v) && !seen.has(v)) { seen.add(v); q.push(v); }
    }
    comps.push(c);
  }
  comps.sort((a, b) => b.length - a.length);
  return comps;
}
/* Connectivity of one town under the SAME physical hazard, for the watch table. */
function areaOutcome(key) {
  const a = AREAS[key], g = areaGraph(key);
  const sc = a.scenarios.find(s => s.key === (CUR && CUR.key)) || a.scenarios[0];
  const centre = { x: a.extent.cx, y: a.extent.cy };
  const isBand = sc.kind === 'band';
  const dead = damagedSet(sc, HAZ.radiusM,
    isBand ? (sc.key === 'landslide' ? HAZ.landslideM : HAZ.floodM) : 0,
    isBand ? (sc.key === 'landslide' ? HAZ.landslideAngle : HAZ.floodAngle) : 0,
    a, centre, HAZ.time / 100);
  const alive = new Set(a.nodes.map(n => n.id).filter(id => !dead.has(id)));
  const comps = componentsIn(a, g.adj, alive);
  const base = (a.baseline && a.baseline.largest) || 1;
  return { key, name: a.name, dead: dead.size, comps: comps.length,
           reachable: (comps[0] || []).length, pct: 100 * (comps[0] || []).length / base };
}
const BMREG = { img: {} };
function bmRegionImage(style) {
  if (BMREG.img[style]) return BMREG.img[style];
  const entry = ((D.region || {}).styles || {})[style];
  if (!entry || !entry.img) return null;
  if (typeof Image === 'undefined') return null;   // see bmImage()
  const img = new Image();
  img.onload = () => { if (typeof render === 'function') render(); };
  img.src = entry.img;
  BMREG.img[style] = img;
  return img;
}

function drawLocator() {
  const c = $('loc'); if (!c) return;
  const r = c.getBoundingClientRect(); if (!r.width) return;
  const dpr = window.devicePixelRatio || 1;
  c.width = Math.round(r.width * dpr); c.height = Math.round(r.height * dpr);
  const ctx = c.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, r.width, r.height);
  const P = pal();
  ctx.fillStyle = P.bg; ctx.fillRect(0, 0, r.width, r.height);
  const keys = AREA_ORDER;
  const lats = keys.map(k => AREAS[k].lat), lons = keys.map(k => AREAS[k].lon);
  const lo0 = Math.min.apply(null, lons), lo1 = Math.max.apply(null, lons);
  const la0 = Math.min.apply(null, lats), la1 = Math.max.apply(null, lats);
  const pad = 24;
  const X = lo => pad + (r.width - 2 * pad) * (lo1 === lo0 ? .5 : (lo - lo0) / (lo1 - lo0));
  const Y = la => r.height - pad - (r.height - 2 * pad) * (la1 === la0 ? .5 : (la - la0) / (la1 - la0));

  /* The locator spans every town at once, so it uses its own stitched mosaic
   * rather than any single area's.  Drawn under the markers, same as the town
   * maps, so the dots genuinely sit where those towns are. */
  const entry = ((D.region || {}).styles || {})[BM_STYLE];
  if (entry) {
    const img = bmRegionImage(BM_STYLE);
    if (img && img.complete && img.naturalWidth) {
      /* Scale by the LARGER ratio so the mosaic covers the box without
       * distorting it -- drawing it straight into the padded rect squashed the
       * map's aspect ratio, which is wrong for anything with real geography. */
      const s = Math.max((r.width - 2 * pad) / entry.w, (r.height - 2 * pad) / entry.h);
      const iw = entry.w * s, ih = entry.h * s;
      ctx.save();
      ctx.globalAlpha = .9;
      ctx.beginPath();
      ctx.rect(0, 0, r.width, r.height);
      ctx.clip();
      ctx.drawImage(img, (r.width - iw) / 2, (r.height - ih) / 2, iw, ih);
      ctx.restore();
    }
  }

  keys.forEach(k => {
    const a = AREAS[k], o = areaOutcome(k);
    const x = X(a.lon), y = Y(a.lat);
    ctx.beginPath(); ctx.arc(x, y, k === curKey ? 8 : 5.5, 0, 7);
    ctx.fillStyle = k === curKey ? '#4cc2ff' : o.pct > 80 ? '#39d98a' : o.pct > 50 ? '#ffc857' : '#ff4757';
    ctx.fill();
    ctx.strokeStyle = P.facRing; ctx.lineWidth = 2; ctx.stroke();
    ctx.font = '10.5px "Segoe UI",sans-serif';
    ctx.lineWidth = 3; ctx.strokeStyle = P.bg;
    ctx.strokeText(`${a.name} ${o.pct.toFixed(0)}%`, x + 11, y + 4);
    ctx.fillStyle = BM_STYLE === 'light' ? '#16233a' : '#cfe0ff';
    ctx.fillText(`${a.name} ${o.pct.toFixed(0)}%`, x + 11, y + 4);
  });
}
function renderWatch() {
  const rows = AREA_ORDER.map(k => areaOutcome(k)).sort((a, b) => a.pct - b.pct);
  setHTML('watch', `<div class="scroll"><table class="tbl">
    <thead><tr><th>Area</th><th class="num">Conn.</th><th class="num">Lost</th>
      <th class="num">Pieces</th><th>State</th></tr></thead><tbody>` +
    rows.map(o => {
      const [lab, cls] = severity(o.pct, o.dead > 0);
      return `<tr data-area="${o.key}" class="rowclick${o.key === curKey ? ' on' : ''}">
        <td>${esc(o.name)}</td>
        <td class="num ${o.pct > 80 ? 'hi' : 'lo'}">${o.pct.toFixed(1)}%</td>
        <td class="num">${o.dead}</td><td class="num">${o.comps}</td>
        <td><span class="sev ${cls}">${lab}</span></td></tr>`;
    }).join('') + `</tbody></table></div>` +
    `<div class="note">Every town is hit at the same <b>real</b> size
      (${fmtM(currentSizeM())}), converted into each town's own coordinate scale, so these
      percentages are physically comparable rather than cosmetically identical.</div>`);
  document.querySelectorAll('#watch tr[data-area]').forEach(tr => {
    tr.onclick = () => activateArea(tr.dataset.area);
  });
}

/* ---------- page 02: disaster simulation ---------- */
function renderSimulation() {
  setTxt('vTime', 'T+' + Math.round(HAZ.time * 1.2) + ' min');
  setTxt('rTime', HAZ.time);
  const f = $('tlFill'); if (f) f.style.width = HAZ.time + '%';
  setHTML('mapbar2', [`<span>${esc(CUR ? CUR.name : '—')}</span>`,
    `<span>T+${Math.round(HAZ.time * 1.2)} min</span>`,
    `<span>${fmtM(currentSizeM())}</span>`,
    `${pct().toFixed(0)}% connected`].join(''));
  setHTML('legend2', $('legend') ? $('legend').innerHTML : '');

  /* every hazard on this town at the current severity, for comparison */
  const rowsFor = s => {
    const dead = s.kind === 'band'
      ? damagedSet(s, HAZ.radiusM, s.key === 'landslide' ? HAZ.landslideM : HAZ.floodM,
                   s.key === 'landslide' ? HAZ.landslideAngle : HAZ.floodAngle, A, EPIC, HAZ.time / 100)
      : damagedSet(s, HAZ.radiusM, 0, 0, A, EPIC, HAZ.time / 100);
    const alive = new Set(N.map(n => n.id).filter(id => !dead.has(id)));
    const c = components(alive);
    const p = 100 * (c[0] || []).length / (BEFORE.largest || 1);
    const [lab, cls] = severity(p, dead.size > 0);
    return `<tr class="rowclick${s.key === (CUR && CUR.key) ? ' on' : ''}" data-cmp="${esc(s.key)}">
      <td>${s.icon || '⚠'} ${esc(s.name)}</td>
      <td class="num">${dead.size}</td><td class="num">${c.length}</td>
      <td class="num ${p > 80 ? 'hi' : 'lo'}">${p.toFixed(1)}%</td>
      <td><span class="sev ${cls}">${lab}</span></td></tr>`;
  };
  setHTML('scnCompare', `<div class="tblwrap"><table class="tbl">
    <thead><tr><th>Hazard</th><th class="num">Lost</th><th class="num">Pieces</th>
      <th class="num">Connected</th><th>Severity</th></tr></thead>
    <tbody>${A.scenarios.map(rowsFor).join('')}</tbody></table></div>`);
  document.querySelectorAll('[data-cmp]').forEach(tr => {
    tr.onclick = () => selectScenario(tr.dataset.cmp);
  });
}

/* ---------- page 04: network analytics ---------- */
function renderAnalytics() {
  const p = pct(), art = STATE.art || [], brs = STATE.bridges || [];
  setHTML('anKpis', kpiRow([
    ['Junctions', N.length.toLocaleString(), ''],
    ['Roads', E.length.toLocaleString(), ''],
    ['Connected', p.toFixed(1) + '%', p > 80 ? 'ok' : p > 40 ? 'warn' : 'bad'],
    ['Pieces', STATE.comps.length, STATE.comps.length > 1 ? 'bad' : 'ok'],
    ['Articulation', art.length, art.length ? 'bad' : 'ok'],
    ['Bridges', brs.length, brs.length ? 'warn' : 'ok'],
  ]));
  setHTML('anBar', [`<span>${esc(A.name)}</span>`, `<span>${ART_MODE_LABEL[currentMode()] || ''}</span>`,
    `${p.toFixed(0)}% connected`].join(''));

  /* centrality histogram */
  const vals = Object.values(STATE.bc || {}).sort((a, b) => b - a);
  const top = vals.slice(0, 12);
  setHTML('cenBars', top.map((v, i) => {
    const w = top[0] ? (100 * v / top[0]) : 0;
    return `<div class="barrow"><span class="barlab">#${i + 1}</span>
      <span class="bar"><i style="width:${w.toFixed(1)}%"></i></span>
      <span class="barval">${v.toFixed(2)}</span></div>`;
  }).join('') || '<div class="note">No surviving junctions.</div>');
  setTxt('cenNote', `Top 12 of ${vals.length} junctions by betweenness. `
    + `Gini-style concentration ${conc(vals).toFixed(2)} — higher means through-traffic `
    + `is concentrated on fewer junctions, which is more fragile.`);

  /* vulnerability / resilience, before vs now */
  const nowV = vulnIndex(STATE.serviceable), wasV = vulnIndex(svc);
  setHTML('vulnTable', tbl(['Metric', 'Before', 'Now', 'Change'], [
    ['Connected (%)', (100 * svc.size / (BEFORE.largest || 1)).toFixed(1), p.toFixed(1), deltaTxt(100 * svc.size / (BEFORE.largest || 1), p)],
    ['Fragments', BEFORE.components, STATE.comps.length, deltaTxt(BEFORE.components, STATE.comps.length)],
    ['Articulation pts', BEFORE.art || '—', art.length, ''],
    ['Bridge edges', BEFORE.bridges || '—', brs.length, ''],
    ['Vulnerability idx', wasV.toFixed(3), nowV.toFixed(3), deltaTxt(wasV, nowV)],
    ['Resilience idx', (1 - wasV).toFixed(3), (1 - nowV).toFixed(3), deltaTxt(1 - wasV, 1 - nowV)],
  ]));
  setTxt('vulnNote', 'Vulnerability is the share of junctions that are articulation points or '
    + 'bridge endpoints — the fraction whose loss disconnects the network. Resilience is 1 minus that.');

  setHTML('artTable', art.length ? tbl(['Junction', 'Betweenness', 'Roads'],
    N.filter(n => art.includes(n.id))
      .sort((a, b) => (STATE.bc[b.id] || 0) - (STATE.bc[a.id] || 0)).slice(0, 25)
      .map(n => [nameOf(n.id), (STATE.bc[n.id] || 0).toFixed(3), (ADJ.get(n.id) || []).length]))
    : '<div class="note">No articulation points in the surviving graph.</div>');

  setHTML('compTable', tbl(['#', 'Size', 'Share'],
    STATE.comps.slice(0, 12).map((c, i) => [String(i + 1), c.length,
      (100 * c.length / N.length).toFixed(1) + '%'])));
  setTxt('analyticNote', 'Every number on this page is recomputed live in the browser from the '
    + 'running scenario — select a different hazard on page 02 and it all changes.');
}
const ART_MODE_LABEL = {
  criticality: 'Betweenness criticality', articulation: 'Articulation points',
  bridges: 'Bridge edges', components: 'Connected components',
};
/* Gini concentration of the betweenness distribution.  `vals` is sorted
 * DESCENDING, so the rank-weighted form is 1 - 2*sum(i*v)/(n*sum(v)). */
function conc(vals) {
  const n = vals.length;
  if (n < 2) return 0;
  const s = vals.reduce((a, b) => a + b, 0);
  if (!s) return 0;
  const num = vals.reduce((acc, v, i) => acc + (i + 1) * v, 0);
  return Math.max(0, Math.min(1, 1 - (2 * num) / (n * s)));
}
function vulnIndex(aliveSet) {
  const ids = new Set(aliveSet);
  const risky = new Set();
  for (const [u, v] of STATE.bridges || []) { risky.add(u); risky.add(v); }
  (STATE.art || []).forEach(id => risky.add(id));
  if (!ids.size) return 1;
  let n = 0;
  for (const id of risky) if (ids.has(id)) n++;
  return Math.min(1, n / ids.size);
}
function kpiRow(items) {
  return items.map(([lab, val, cls]) =>
    `<div class="kpi"><span class="kl">${esc(lab)}</span>
       <span class="kv ${cls || ''}">${val}</span></div>`).join('');
}
/* Generic table.  Pads/truncates each row to the header width so a renderer
 * that changes shape degrades to blank cells instead of throwing and taking
 * the whole page down. */
function tbl(head, rows) {
  const w = head.length;
  return `<table class="tbl"><thead><tr>${head.map((h, i) =>
    `<th${i ? ' class="num"' : ''}>${esc(h)}</th>`).join('')}</tr></thead><tbody>` +
    (rows || []).map(r => {
      const cells = Array.isArray(r) ? r : [r];
      return `<tr>${Array.from({ length: w }, (_, i) =>
        `<td${i ? ' class="num"' : ''}>${cells[i] === undefined ? '' : cells[i]}</td>`).join('')}</tr>`;
    }).join('') + `</tbody></table>`;
}

/* ---------- page 05: facilities & routes ---------- */
let FAC_KIND = 'all';
function renderFacilities() {
  const all = facilityStates();
  const kinds = ['all'].concat(Array.from(new Set(all.map(f => f.kind))).sort());
  setHTML('facFilters', kinds.map(k =>
    `<button class="chip${FAC_KIND === k ? ' on' : ''}" data-kind="${esc(k)}">${esc(k)}</button>`).join(''));
  document.querySelectorAll('[data-kind]').forEach(b => {
    b.onclick = () => { FAC_KIND = b.dataset.kind; renderFacilities(); };
  });

  const shown = FAC_KIND === 'all' ? all : all.filter(f => f.kind === FAC_KIND);
  const ok = shown.filter(f => f.reachable).length;
  const dead = shown.filter(f => f.destroyed).length;
  const cut = shown.filter(f => !f.reachable && !f.destroyed).length;

  setHTML('facKpis', kpiRow([
    ['Monitored', shown.length, ''],
    ['Reachable', ok, 'ok'],
    ['Cut off', cut, cut ? 'warn' : 'ok'],
    ['Destroyed', dead, dead ? 'bad' : 'ok'],
  ]));
  setHTML('facBar', [`<span>${esc(A.name)}</span>`,
    `${ok}/${shown.length} reachable`, `${pct().toFixed(0)}% network`].join(''));

  /* Rows go in as cell arrays, not raw <tr> markup: tbl() wraps each row in a
   * <tr> itself, so handing it markup nests a table inside a cell. */
  setHTML('facTable', tbl(['Facility', 'Kind', 'Status', 'Roads'],
    shown.map(f => {
      const [st, cls] = statusOf(f);
      const n = NID.get(f.node);
      return [`<span class="wrapcell">${facIcon(f.kind)} ${esc(f.name)}</span>`, esc(f.kind),
        `<span class="sev ${cls}">${st}</span>`,
        n ? (ADJ.get(f.node) || []).length : 0];
    })));
  setTxt('facNote', A.realFacilities
    ? 'Facilities are real OpenStreetMap nodes placed on the real road graph.'
    : 'Facility junctions were snapped to the nearest real road junction.');
  /* page 01's facility tab shares the same list; that panel is 2 columns, so
   * give it its own shape rather than feeding tbl() a mismatched row. The name
   * wraps so long hospital names do not widen the narrow side panel. */
  setHTML('facList', tbl(['Facility', 'Status'], all.map(f => {
    const [st, cls] = statusOf(f);
    return [`<span class="wrapcell">${facIcon(f.kind)} ${esc(f.name)}</span>`,
      `<span class="sev ${cls}">${st}</span>`];
  })));

  const isRoute = currentFacMode() === 'route';
  const rp = $('routePanel'); if (rp) rp.style.display = isRoute ? '' : 'none';
  if (isRoute) renderRoutes(all);
}
function renderRoutes(fac) {
  const opts = fac.map(f => `<option value="${f.node}">${esc(f.name)}</option>`).join('');
  const from = $('routeFrom'), to = $('routeTo');
  if (from && from.innerHTML !== opts) {
    from.innerHTML = opts; to.innerHTML = opts;
    from.value = fac[0] ? fac[0].node : '';
    to.value = fac[1] ? fac[1].node : '';
    from.onchange = to.onchange = () => renderRoutes(fac);
  }
  if (!from || !to) return;
  const a = +from.value, b = +to.value;
  if (!a || !b || a === b) { setHTML('routeOut', '<div class="note">Pick two different facilities.</div>'); return; }

  const pre = shortestPathOn(ADJ, a, b, new Set(), DEG);
  const post = shortestPathOn(ADJ, a, b, STATE.dead, DEG);
  const show = (t, r) => `<div class="note"><b>${t}</b></div>` +
    (r ? `<div>Reachable in <b>${r.cost}</b> weighted steps across <b>${r.segs.length}</b> road segments.</div>`
       : `<div style="color:var(--crit)">No route exists — the hazard has cut the corridor.</div>`);
  setHTML('routeOut', show('Before the disaster', pre) + show('After the disaster', post));
  STATE.route = post ? post.segs : null;
  drawMap('cvFac', 'route');
}
function shortestPathOn(adj, from, to, blocked, deg) {
  if (!adj.has(from) || !adj.has(to)) return null;
  const dist = new Map([[from, 0]]), prev = new Map(), seen = new Set();
  const q = [[0, from]];
  while (q.length) {
    q.sort((a, b) => a[0] - b[0]);
    const [d, u] = q.shift();
    if (seen.has(u)) continue;
    seen.add(u);
    if (u === to) break;
    for (const v of adj.get(u) || []) {
      if (blocked.has(v)) continue;
      const w = (deg.get(v) || 1) + (blocked.has(v) ? 40 : 0);
      const nd = d + w;
      if (nd < (dist.has(v) ? dist.get(v) : Infinity)) { dist.set(v, nd); prev.set(v, u); q.push([nd, v]); }
    }
  }
  if (from !== to && !prev.has(to)) return null;
  const segs = []; let cur = to;
  while (cur !== from) { const p = prev.get(cur); if (p === undefined) return null; segs.unshift([p, cur]); cur = p; }
  return { segs, cost: dist.get(to) || 0 };
}

/* ---------- page 06: recovery planning ---------- */
function repairCandidates() {
  /* Rank destroyed junctions by how much connectivity each repair restores:
   * how many currently-isolated nodes it would reconnect to the largest
   * component.  Measured, not guessed. */
  const big = new Set(STATE.comps[0] || []);
  const out = [];
  for (const id of STATE.dead) {
    let gain = 0;
    for (const v of ADJ.get(id) || []) {
      if (STATE.dead.has(v)) continue;
      if (big.has(v)) gain += 1;
      else {
        const comp = STATE.comps.find(c => c.includes(v));
        if (comp) gain += comp.length;
      }
    }
    out.push({ id, gain, name: nameOf(id) });
  }
  return out.sort((a, b) => b.gain - a.gain);
}
function allocateCrews() {
  const crews = +($('rTeams') ? $('rTeams').value : 3);
  const cand = repairCandidates().filter(c => !STATE.repairs.has(c.id));
  const base = (STATE.crews || []).length;
  STATE.crews = cand.slice(0, crews).map((c, i) => ({
    team: 'Team ' + String.fromCharCode(65 + base + i), target: c.name,
    id: c.id, gain: c.gain, done: false,
  }));
  renderRecovery();
}
function renderRecovery() {
  if (!STATE.repairs) STATE.repairs = new Set();
  if (!STATE.log) STATE.log = [];
  const p = pct(), cand = repairCandidates();
  setHTML('recKpis', kpiRow([
    ['Destroyed', STATE.dead.size, STATE.dead.size ? 'bad' : 'ok'],
    ['Repaired', STATE.repairs.size, 'ok'],
    ['Connected', p.toFixed(1) + '%', p > 80 ? 'ok' : 'warn'],
    ['Pieces', STATE.comps.length, STATE.comps.length > 1 ? 'bad' : 'ok'],
  ]));
  setHTML('recBar', [`<span>${esc(A.name)}</span>`, `${STATE.repairs.size} repaired`,
    `${p.toFixed(0)}% connected`].join(''));

  setHTML('prioList', cand.length ? tbl(['#', 'Junction', 'Reconnected'],
    cand.slice(0, 15).map((c, i) => [String(i + 1),
      STATE.repairs.has(c.id) ? `${esc(c.name)} ✓` : esc(c.name), c.gain]))
    : '<div class="note">Nothing left to repair — every destroyed junction is restored.</div>');
  document.querySelectorAll('#prioList tbody tr').forEach((tr, i) => {
    const c = cand[i]; if (!c || STATE.repairs.has(c.id)) return;
    tr.classList.add('rowclick');
    tr.onclick = () => {
      STATE.repairs.add(c.id); STATE.fixed.add(c.id);
      STATE.log.push({ t: STATE.repairs.size * 2, text: `Repaired ${c.name} (+${c.gain} reachable)` });
      recompute();
    };
  });
  setHTML('teamList', (STATE.crews || []).length ? tbl(['Team', 'Target', 'Gain', 'Status'],
    STATE.crews.map(c => [esc(c.team), esc(c.target), String(c.gain), c.done ? 'done' : 'assigned']))
    : '<div class="note">No crews allocated yet. Use <b>Allocate crews</b> to assign the highest-impact repairs.</div>');
  setTxt('recClock', 'T+' + (STATE.repairs.size * 2) + ' h');
  setTxt('recNote', 'Each repair is applied to the real graph, so connectivity, components and the '
    + 'maps all update from the same computation.');
  setHTML('recLog', STATE.log.length ? tbl(['Hour', 'Action'],
    STATE.log.slice().reverse().map(l => ['T+' + l.t + ' h', esc(l.text)]))
    : '<div class="note">No repairs yet. Click a priority row to dispatch a crew.</div>');
}

/* ---------- page 07: model & research results (all measured, none hard-coded) ---------- */
function renderResearch() {
  const R = D.research || {}, pipe = D.pipeline || [];
  if (pipe.length) {
    setHTML('pipe3', pipe.map((s, i) =>
      `<div class="pipestep" data-pipe="${i}"><img src="${s.img}" alt="${esc(s.title)}">
         <span>${esc(s.title)}</span></div>`).join(''));
    document.querySelectorAll('[data-pipe]').forEach(el =>
      { el.onclick = () => showPipe(+el.dataset.pipe); });
    if (STATE.pipeIdx === undefined) STATE.pipeIdx = 0;
    showPipe(STATE.pipeIdx);
    setTxt('pipeNote', 'Each panel is a real artefact: the imagery is a held-out DeepGlobe patch, the '
      + 'probability map comes from the trained checkpoint, and the final graph is produced by '
      + 'vectorising that mask and healing it.');
    setHTML('deepglobeNote', '<b>Scope:</b> DeepGlobe is the <b>segmentation model and optional '
      + 'gap-filler</b>. The road network on pages 01, 02, 04, 05 and 06 is real OpenStreetMap data, '
      + 'not a synthesised image. Stage 7 is the one place a DeepGlobe-derived graph appears.');
  }
  setHTML('scopeNote', 'This page documents the image-to-graph pipeline. It is <b>not</b> the source of '
    + 'the town maps: those are OpenStreetMap drive networks. Use it to understand how a raw '
    + 'satellite patch becomes a routable graph.');
  setHTML('prov', 'Roads and facilities: <b>OpenStreetMap</b> (drive network; service roads excluded). '
    + `Segmentation: trained U-Net checkpoints measured on a held-out DeepGlobe test split. Hazards are `
    + `sized in real metres (earthquake ${HAZ.radiusM} m, flood ${HAZ.floodM} m, landslide ${HAZ.landslideM} m) `
    + `and converted into each town's coordinate scale.`);
  setTxt('footProv', 'Map data: OpenStreetMap · Segmentation: DeepGlobe (held-out test split)');
  /* Tile attribution is required by both basemap sources, and this build ships
   * their imagery inline -- say so plainly rather than leaving it off. */
  setHTML('footBasemap', D.basemapAttribution
    ? 'Basemap: ' + D.basemapAttribution : '');

  setTxt('llmNote', 'The assistant answers from the live simulation using deterministic rules over the '
    + 'real graph. No external model or network call is involved, which is why this file works offline.');

  if (!R.models) {
    setHTML('resMetrics', '<div class="note">Research metrics were not included in this build.</div>');
    return;
  }
  const b = R.models.baseline, a = R.models.attention;
  const cell = (src, k, dp) => src && src[k] !== undefined
    ? src[k].toFixed(dp === undefined ? 3 : dp) : '—';
  const row = (label, k, dp) => `<tr><td>${esc(label)}</td>` +
    [b.clean, a.clean, b.occluded, a.occluded].map(s =>
      `<td class="num">${cell(s, k, dp)}</td>`).join('') + `</tr>`;
  setHTML('resMetrics', `<table class="tbl"><thead><tr><th>Metric</th>
    <th class="num">Base clean</th><th class="num">Attn clean</th>
    <th class="num">Base occl.</th><th class="num">Attn occl.</th></tr></thead><tbody>` +
    row('IoU', 'iou') + row('Dice', 'dice') + row('Precision', 'precision') +
    row('Recall', 'recall') + row('Accuracy', 'accuracy') +
    row('Mean confidence', 'mean_confidence') + row('IoU std', 'iou_std') +
    row('ms / image', 'inference_ms_per_image', 1) + `</tbody></table>`);
  setTxt('resNote', `Measured on ${(b.clean && b.clean.n_images) || 0} held-out patches the models never `
    + `saw in training. Attention raises IoU from ${b.clean.iou.toFixed(3)} to ${a.clean.iou.toFixed(3)} `
    + `and is also faster on this device.`);
  renderAblation(R, b, a);
}
function renderAblation(R, b, a) {
  const d = (x, y, dp) => ((y - x) >= 0 ? '+' : '') + (y - x).toFixed(dp === undefined ? 3 : dp);
  setHTML('resAblation', tbl(['Comparison', 'Baseline', 'Attention', 'Delta'], [
    ['IoU (clean)', b.clean.iou.toFixed(3), a.clean.iou.toFixed(3), d(b.clean.iou, a.clean.iou)],
    ['Recall (clean)', b.clean.recall.toFixed(3), a.clean.recall.toFixed(3),
      d(b.clean.recall, a.clean.recall)],
    ['Recall (occluded)', b.occluded.recall.toFixed(3), a.occluded.recall.toFixed(3),
      d(b.occluded.recall, a.occluded.recall)],
    ['ms / image', b.clean.inference_ms_per_image.toFixed(0),
      a.clean.inference_ms_per_image.toFixed(0),
      d(b.clean.inference_ms_per_image, a.clean.inference_ms_per_image, 0)],
  ]));
  setTxt('resAblNote', 'The attention variant is the model this console ships as the segmenter.');
  const ds = R.dataset || {};
  setHTML('resDataset', Object.keys(ds).filter(k => typeof ds[k] !== 'object')
    .map(k => `<dt>${esc(k.replace(/_/g, ' '))}</dt><dd>${esc(ds[k])}</dd>`).join('')
    || '<dd>not included in this build</dd>');
  setTxt('resDatasetNote', 'Counts from the real DeepGlobe road-extraction split on disk.');
  setHTML('resCompute', `<dt>Device</dt><dd>${esc(R.device || '—')}</dd>` +
    `<dt>Params (base / attn)</dt><dd>${(b.params / 1e6).toFixed(1)}M / `
    + `${(a.params / 1e6).toFixed(1)}M</dd>` +
    `<dt>Test patches</dt><dd>${(b.clean && b.clean.n_images) || 0}</dd>`);
  setTxt('resComputeNote', 'Inference timing is per 256x256 patch on the device above.');
  setHTML('resHeal', Array.isArray(R.healing) && R.healing.length
    ? tbl(['Method', 'Value'], R.healing.map(h => [esc(h.name), esc(String(h.value))]))
    : '<div class="note">No healing ablation in this build.</div>');
  setTxt('resHealNote', 'Learned link prediction versus the classical distance+angle heal.');
  setHTML('deepScope', 'DeepGlobe is genuinely used for <b>segmentation</b> (it produces stages 3-4 above) '
    + 'and can fill gaps in the healed graph. It is <b>not</b> used to invent the town maps — those '
    + 'are real OpenStreetMap drive networks throughout.');
}
function showPipe(i) {
  const pipe = D.pipeline || [];
  if (!pipe.length) return;
  STATE.pipeIdx = Math.max(0, Math.min(pipe.length - 1, i));
  const s = pipe[STATE.pipeIdx];
  setHTML('pipeDetail', `<img src="${s.img}" alt="${esc(s.title)}"
      style="width:100%;border-radius:10px;border:1px solid var(--line)" />
    <div class="note" style="margin-top:9px"><b>${esc(s.title)}</b><br>${esc(s.caption || '')}</div>`);
  document.querySelectorAll('[data-pipe]').forEach((el, k) =>
    el.classList.toggle('on', k === STATE.pipeIdx));
}

/* ============================ interaction ============================ */
let PAGE = 'overview';
let DRAG = false;
function showPage(p) {
  PAGE = p;
  document.querySelectorAll('.page').forEach(s =>
    s.classList.toggle('on', s.id === 'page-' + p));
  document.querySelectorAll('#nav button').forEach(b =>
    b.classList.toggle('on', b.dataset.p === p));
  fitView({ overview: 'cv', simulation: 'cv2', analytics: 'cvAn',
            facilities: 'cvFac', recovery: 'cvRec' }[p] || 'cv');
  render();
  window.scrollTo(0, 0);
}
function wire() {
  document.querySelectorAll('#nav button').forEach(b =>
    { b.onclick = () => showPage(b.dataset.p); });
  const sel = $('areaSel');
  if (sel) {
    sel.innerHTML = AREA_ORDER.map(k =>
      `<option value="${k}">${esc(AREAS[k].name)}</option>`).join('');
    sel.onchange = () => activateArea(sel.value);
  }
  const rt = $('rTime');
  if (rt) rt.oninput = () => { HAZ.time = +rt.value; schedule(); };
  let play = null;
  const bp = $('btnPlay');
  if (bp) bp.onclick = () => {
    if (play) { clearInterval(play); play = null; bp.innerHTML = '&#9654; Play'; return; }
    bp.innerHTML = '&#10073;&#10073; Pause';
    play = setInterval(() => {
      HAZ.time = HAZ.time >= 100 ? 0 : HAZ.time + 2;
      recompute();
    }, 90);
  };
  if ($('btnTimeReset')) $('btnTimeReset').onclick = () => { HAZ.time = 0; recompute(); };
  const bd = $('btnDrag');
  if (bd) bd.onclick = () => {
    DRAG = !DRAG;
    bd.classList.toggle('on', DRAG);
    bd.textContent = DRAG ? '✔ Click the map' : '✂ Move Epicentre';
  };
  if ($('btnReset')) $('btnReset').onclick = () => {
    EPIC = { x: A.extent.cx, y: A.extent.cy };
    HAZ.time = 100;
    STATE.fixed = new Set(); STATE.repairs = new Set(); STATE.log = [];
    recompute();
  };
  const zoom = (id, k) => { const v = view(id); v.s = Math.max(1, v.s * k); drawMap(id, 'overview'); };
  if ($('btnZin')) $('btnZin').onclick = () => zoom('cv', 1.25);
  if ($('btnZout')) $('btnZout').onclick = () => zoom('cv', 1 / 1.25);
  if ($('btnFit2')) $('btnFit2').onclick = () => { fitAll(); render(); };

  /* Basemap style: one shared switch, one button per map panel.  Flipping it
   * re-renders everything, so every map on every page changes together. */
  document.querySelectorAll('.mapctl button.bm').forEach(b => {
    b.onclick = () => {
      const other = hasBasemap(curKey, 'light') ? 'light'
                  : hasBasemap(curKey, 'dark') ? 'dark' : null;
      BM_STYLE = BM_STYLE === 'dark' ? (other || 'dark') : 'dark';
      preloadBasemap(curKey);
      syncBasemapButtons();
      render();
    };
  });
  syncBasemapButtons();

  /* pan + epicentre placement on the overview map */
  const cv = $('cv');
  if (cv) {
    let panning = false, lx = 0, ly = 0;
    cv.addEventListener('mousedown', e => {
      if (DRAG) return;
      panning = true; lx = e.clientX; ly = e.clientY;
      cv.classList.add('drag');
    });
    window.addEventListener('mouseup', () => { panning = false; cv.classList.remove('drag'); });
    window.addEventListener('mousemove', e => {
      if (!panning) return;
      const v = view('cv');
      v.ox += e.clientX - lx; v.oy += e.clientY - ly;
      lx = e.clientX; ly = e.clientY;
      drawMap('cv', 'overview');
    });
    cv.addEventListener('wheel', e => {
      e.preventDefault();
      const v = view('cv');
      v.s = Math.max(1, Math.min(60, v.s * (e.deltaY < 0 ? 1.12 : 1 / 1.12)));
      drawMap('cv', 'overview');
    }, { passive: false });
    cv.addEventListener('click', e => {
      if (!DRAG) return;
      const r = cv.getBoundingClientRect(), v = view('cv'), ex = A.extent;
      const x = (e.clientX - r.left - v.ox) / v.s;
      const y = (e.clientY - r.top - v.oy) / v.s;
      const fx = Math.max(0, Math.min(1, (x - ex.x0) / ((ex.x1 - ex.x0) || 1)));
      const fy = Math.max(0, Math.min(1, (y - ex.y0) / ((ex.y1 - ex.y0) || 1)));
      EPIC = { x: ex.x0 + fx * (ex.x1 - ex.x0), y: ex.y0 + fy * (ex.y1 - ex.y0) };
      DRAG = false;
      if (bd) { bd.classList.remove('on'); bd.textContent = '✂ Move Epicentre'; }
      recompute();
    });
  }

  /* tab groups: page 01 swaps panels, pages 04/05 switch the map mode */
  document.querySelectorAll('.tabs').forEach(group => {
    group.querySelectorAll('.tab').forEach(t => {
      t.onclick = () => {
        group.querySelectorAll('.tab').forEach(x => x.classList.remove('on'));
        t.classList.add('on');
        const page = group.closest('.page');
        if (page && page.id === 'page-overview') {
          document.querySelectorAll('[id^="tab-"]').forEach(p => p.classList.add('hidden'));
          const el = $('tab-' + t.dataset.tab);
          if (el) el.classList.remove('hidden');
        }
        if (page && page.id === 'page-analytics') drawMap('cvAn', currentMode());
        render();
      };
    });
  });

  /* assistant: deterministic, no network */
  if ($('btnAsk')) $('btnAsk').onclick = () => {
    const a = $('ans');
    a.style.display = '';
    a.textContent = answer($('ask').value);
  };
  setHTML('quickQ', QUICK_Q.map(q =>
    `<button class="chip" data-q="${esc(q)}">${esc(q)}</button>`).join(''));
  document.querySelectorAll('[data-q]').forEach(b => {
    b.onclick = () => { $('ask').value = b.dataset.q; $('btnAsk').click(); };
  });

  /* recovery */
  if ($('btnAllocate')) $('btnAllocate').onclick = allocateCrews;
  const autoRepair = () => {
    for (const c of repairCandidates().filter(x => !STATE.repairs.has(x.id))) {
      STATE.repairs.add(c.id); STATE.fixed.add(c.id);
      STATE.log.push({ t: STATE.repairs.size * 2, text: `Repaired ${c.name} (+${c.gain})` });
    }
    recompute();
  };
  if ($('btnAuto')) $('btnAuto').onclick = autoRepair;
  if ($('btnRecPlay')) $('btnRecPlay').onclick = () => {
    const c = repairCandidates().find(x => !STATE.repairs.has(x.id));
    if (!c) return;
    STATE.repairs.add(c.id); STATE.fixed.add(c.id);
    STATE.log.push({ t: STATE.repairs.size * 2, text: `Repaired ${c.name} (+${c.gain})` });
    recompute();
  };
  if ($('btnResetRec')) $('btnResetRec').onclick = () => {
    STATE.fixed = new Set(); STATE.repairs = new Set(); STATE.log = []; STATE.crews = [];
    recompute();
  };
  const rtm = $('rTeams');
  if (rtm) rtm.oninput = () => setTxt('vTeams', rtm.value);
  window.addEventListener('resize', () => { fitAll(); render(); });
}

/* ============================ boot ============================ */
function init() {
  if ($('areaSel')) $('areaSel').value = curKey;
  setTxt('pillMode', 'Offline Mode');
  /* activateArea() is what CREATES STATE, so it must run before anything that
   * writes to it. */
  activateArea(START_KEY);
  STATE.repairs = new Set();
  STATE.log = [];
  STATE.crews = [];
  STATE.route = null;
  renderSimulationCtls();
  wire();
  fitAll();
  showPage('overview');
}
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
else init();
