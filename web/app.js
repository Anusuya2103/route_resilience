/* RoadNet Ops - client-side emergency simulation engine.
 *
 * Mirrors src/simulation.py and src/emergency_assistant.py in JavaScript so the
 * sliders respond instantly without a round trip to Python.  Python remains the
 * source of truth for the pre-computed briefings embedded in the payload.
 */
const D = JSON.parse(document.getElementById('payload').textContent);

/* Multi-area console.  Every monitored town ships with its own complete graph,
 * so one operator can sit in Kovilpatti and still drill into Thoothukudi or
 * Madurai under the SAME running scenario.  The scenario is global (same hazard
 * everywhere); the graph, facilities and numbers are per area.
 *
 * Legacy single-area payloads (no `areas` key) are wrapped so the engine below
 * only ever has to deal with one shape. */
const AREAS = D.areas || { area: Object.assign({}, D, { meta: {} }) };
const AREA_ORDER = D.areaOrder || Object.keys(AREAS);
const START_KEY = AREA_ORDER.includes(D.defaultArea) ? D.defaultArea : AREA_ORDER[0];
let curKey = START_KEY;

/* The active area.  All area-scoped reads go through this object. */
let A = AREAS[START_KEY];

/* Graph state for the ACTIVE area (rebuilt by activateArea). */
let N, E, NID, ADJ, cen, cenMax, BEFORE, svc, CUR, STATE;

/* Per-area adjacency/baseline cache, so the "all areas" watch table can be
 * recomputed on every slider move without rebuilding everything. */
const ACACHE = {};
function areaGraph(key) {
  if (!ACACHE[key]) {
    const a = AREAS[key];
    const adj = new Map(a.nodes.map(n => [n.id, []]));
    a.edges.forEach(e => { adj.get(e.s).push(e.t); adj.get(e.t).push(e.s); });
    ACACHE[key] = { adj, base: new Set(a.baseline.serviceable), nodes: a.nodes };
  }
  return ACACHE[key];
}

function activateArea(key) {
  if (!AREAS[key]) return;
  curKey = key;
  A = AREAS[key];
  N = A.nodes; E = A.edges;
  NID = new Map(N.map(n => [n.id, n]));
  ADJ = areaGraph(key).adj;
  cen = A.centrality;
  cenMax = Math.max(...Object.values(cen)) || 1;
  BEFORE = A.baseline;
  svc = new Set(BEFORE.serviceable);
  // keep the SAME hazard running when switching towns
  const prevKey = CUR && CUR.key;
  CUR = A.scenarios.find(s => s.key === prevKey) || A.scenarios[0];
  EPIC = { x: A.extent.cx, y: A.extent.cy };
  STATE = { r: A.scale, bw: A.bandScale, ang: A.bandAngle,
            dead: new Set(), alive: null, comps: [], bc: cen, serviceable: new Set() };
  document.getElementById('rRadius').value = A.scale;
  document.getElementById('rBand').value = A.bandScale;
  document.getElementById('rAngle').value = A.bandAngle;
  document.getElementById('vRadius').textContent = fmtSize(A.scale);
  document.getElementById('vBand').textContent = fmtSize(A.bandScale);
  buildScenarioCards();
  markActiveArea();
  fit();
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

/* Brandes betweenness on the surviving subgraph (weights ignored for speed;
   matches the unweighted structure closely enough for live feedback). */
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

/* ---------- hazard geometry (same maths as Python) ----------
 * radius / bandW arrive as a PERCENTAGE of the graph diagonal, so the controls
 * behave identically whether the graph is in pixels, metres or 0-1 coords.
 * For a real OSM graph those percentages are derived from real metric radii, so
 * a "22%" control still means a physically meaningful distance. */
const SIZE_UNIT_A = () => A.sizeUnit || 'px';
function fmtSize(pct) {
  if (SIZE_UNIT_A() !== 'm') return pct + '% of region';
  const v = (pct / 100) * A.extent.diag;
  return v >= 1000 ? (v / 1000).toFixed(2) + ' km' : Math.round(v) + ' m';
}
/* `area` and `centre` are parameters so the "all areas" watch table can run the
 * same maths against other towns without disturbing the active view.
 * The disc/band centre comes from EPIC -- previously it was pinned to the graph
 * centre, so dragging the epicentre moved the drawn circle but not the damage. */
function damagedSet(sc, radiusPct, bandPct, angle, area, centre) {
  area = area || A;
  centre = centre || EPIC;
  const nodes = area.nodes, diag = area.extent.diag;
  const dead = new Set();
  if (sc.kind === 'nodes') {
    (area.gatekeepers || []).slice(0, sc.count || 3).forEach(id => dead.add(id));
    return dead;
  }
  if (sc.kind === 'disc') {
    const r = (radiusPct / 100) * diag;
    for (const n of nodes) { const dx = n.x - centre.x, dy = n.y - centre.y; if (dx * dx + dy * dy <= r * r) dead.add(n.id); }
  } else {
    const bw = (bandPct / 100) * diag;
    const a = angle * Math.PI / 180, dx = Math.cos(a), dy = Math.sin(a);
    for (const n of nodes) { const p = Math.abs((n.x - centre.x) * -dy + (n.y - centre.y) * dx); if (p <= bw) dead.add(n.id); }
  }
  return dead;
}

/* ---------- canvas ---------- */
const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
let view = { s: 1, ox: 0, oy: 0 }, hover = null, dragMode = false, dragging = false, last = null;
let EPIC = { x: A.extent.cx, y: A.extent.cy };
/* CUR and STATE are declared with the other area state at the top of the file
 * and (re)initialised by activateArea() -- re-declaring them here would be a
 * duplicate let binding and a hard SyntaxError. */

/* shared node-state helpers, used by the renderer AND the panels so they can
 * never disagree.  Three states are distinguished so a network that was already
 * fragmented is not misread as damage:
 *   destroyed    -> destroyed by the hazard
 *   newly lost   -> grey, was connected before and is not now
 *   never linked -> dimmed, sat in a separate fragment before the event        */
/* svc is declared with the other area state at the top and reset by
 * activateArea() -- do not re-declare it here. */
/* Accepts EITHER a node object or a bare node id.  Facility code calls these
 * with f.node (an id) while the renderer calls them with node objects; reading
 * `.id` off a number yields undefined, which silently reported every facility
 * as "never connected before" regardless of the real baseline. */
const nodeId = n => (n !== null && typeof n === 'object') ? n.id : n;
const wasOk = n => svc.has(nodeId(n));
const inService = n => STATE.serviceable.has(nodeId(n));

function fit() {
  const r = cv.getBoundingClientRect(), pad = 26;
  const xs = N.map(n => n.x), ys = N.map(n => n.y);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  view.s = Math.min((r.width - pad * 2) / (x1 - x0 || 1), (r.height - pad * 2) / (y1 - y0 || 1));
  view.ox = (r.width - (x1 - x0) * view.s) / 2 - x0 * view.s;
  view.oy = (r.height - (y1 - y0) * view.s) / 2 - y0 * view.s;
}
const TX = x => x * view.s + view.ox, TY = y => y * view.s + view.oy;

function col(v) {
  const t = Math.max(0, Math.min(1, v));
  const a = [57, 217, 138], b = [255, 200, 87], c = [255, 71, 87];
  const [f, tt] = t < .5 ? [a, t * 2] : [b, (t - .5) * 2];
  const g = [0, 1, 2].map(i => Math.round(f[i] + (c[i] - f[i]) * tt));
  return `rgb(${g[0]},${g[1]},${g[2]})`;
}

function draw() {
  const r = cv.getBoundingClientRect();
  if (cv.width !== r.width * devicePixelRatio || cv.height !== r.height * devicePixelRatio) {
    cv.width = r.width * devicePixelRatio; cv.height = r.height * devicePixelRatio;
  }
  ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  ctx.clearRect(0, 0, cv.width, cv.height);

  const sc = CUR.kind === 'nodes' ? null : CUR;
  if (sc) {                                   // hazard zone underlay
    ctx.save();
    if (CUR.kind === 'disc') {
      const rPx = (STATE.r / 100) * A.extent.diag * view.s;
      ctx.beginPath(); ctx.arc(TX(EPIC.x), TY(EPIC.y), rPx, 0, 7);
      ctx.fillStyle = 'rgba(255,71,87,.13)'; ctx.fill();
      ctx.strokeStyle = 'rgba(255,71,87,.6)'; ctx.setLineDash([6, 5]); ctx.lineWidth = 1.5; ctx.stroke();
    } else {
      const a = STATE.ang * Math.PI / 180, dx = Math.cos(a), dy = Math.sin(a);
      const bwPx = (STATE.bw / 100) * A.extent.diag * view.s;
      const L = 4000, mx = EPIC.x, my = EPIC.y;
      ctx.beginPath();
      ctx.moveTo(TX(mx - dy * bwPx - dx * L), TY(my + dx * bwPx - dy * L));
      ctx.lineTo(TX(mx + dy * bwPx + dx * L), TY(my - dx * bwPx - dy * L));
      ctx.lineTo(TX(mx - dy * bwPx + dx * L), TY(my + dx * bwPx + dy * L));
      ctx.lineTo(TX(mx + dy * bwPx - dx * L), TY(my - dx * bwPx - dy * L));
      ctx.closePath();
      ctx.fillStyle = 'rgba(64,160,255,.14)'; ctx.fill();
      ctx.strokeStyle = 'rgba(64,160,255,.6)'; ctx.setLineDash([6, 5]); ctx.lineWidth = 1.5; ctx.stroke();
    }
    ctx.restore();
  }

  // edges
  ctx.lineWidth = 1.1;
  for (const e of E) {
    const a = NID.get(e.s), b = NID.get(e.t);
    const dead = STATE.dead.has(e.s) || STATE.dead.has(e.t);
    const joined = inService(a) && inService(b);
    const ever = wasOk(a) && wasOk(b);
    ctx.beginPath(); ctx.moveTo(TX(a.x), TY(a.y)); ctx.lineTo(TX(b.x), TY(b.y));
    if (dead) { ctx.strokeStyle = 'rgba(255,71,87,.5)'; ctx.setLineDash([2, 3]); }
    else if (STATE.dead.size && !joined && ever) { ctx.strokeStyle = 'rgba(130,142,170,.35)'; ctx.setLineDash([]); }
    else if (!ever) { ctx.strokeStyle = 'rgba(110,130,170,.28)'; ctx.setLineDash([]); }
    else { ctx.strokeStyle = 'rgba(150,190,255,.6)'; ctx.setLineDash([]); }
    ctx.stroke();
  }
  // nodes
  for (const n of N) {
    const v = (STATE.bc[n.id] || 0) / cenMax;
    const dead = STATE.dead.has(n.id);
    const cut = !dead && !inService(n) && wasOk(n);
    ctx.beginPath(); ctx.arc(TX(n.x), TY(n.y), dead ? 2.2 : (n.id === hover ? 6 : 3.4), 0, 7);
    if (dead) ctx.fillStyle = 'rgba(255,71,87,.5)';
    else if (cut) ctx.fillStyle = 'rgba(140,150,170,.75)';
    else if (!inService(n)) ctx.globalAlpha = .45, ctx.fillStyle = col(v);
    else ctx.fillStyle = col(v);
    ctx.fill(); ctx.globalAlpha = 1;
    if (n.id === hover) { ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke(); }
  }
  // facilities
  for (const f of A.facilities) {
    const n = NID.get(f.node); if (!n) continue;
    const bad = STATE.dead.has(f.node), out = STATE.dead.size && !inService(n);
    const col2 = bad ? '#ff4757' : out ? '#8fa0c0' : '#4cc2ff';
    ctx.beginPath(); ctx.arc(TX(n.x), TY(n.y), 7, 0, 7);
    ctx.fillStyle = col2; ctx.fill();
    ctx.lineWidth = 2; ctx.strokeStyle = '#0b1020'; ctx.stroke();
    ctx.font = '11px Segoe UI, sans-serif'; ctx.fillStyle = '#cfe0ff';
    ctx.fillText(f.name, TX(n.x) + 10, TY(n.y) + 4);
  }
}

/* ---------- live simulation ---------- */
let timer = null;

function recompute() {
  const dead = damagedSet(CUR, STATE.r, STATE.bw, STATE.ang);
  const alive = new Set(N.map(n => n.id).filter(id => !dead.has(id)));
  const comps = components(alive);
  STATE.dead = dead; STATE.alive = alive; STATE.comps = comps;
  STATE.serviceable = new Set(comps[0] || []);
  STATE.bc = dead.size ? betweenness(alive) : cen;
  render();
}

function schedule() {
  clearTimeout(timer);
  timer = setTimeout(recompute, 130);          // debounce while dragging sliders
}

const pct = () => STATE.comps.length ? 100 * (STATE.comps[0] || []).length / (BEFORE.largest || 1) : 0;

function severity(p, destroyed) {
  if (p >= 90) return destroyed ? ['SEVERE', 'orange'] : ['MINOR', 'green'];
  if (p >= 60) return ['MODERATE', 'amber'];
  if (p >= 25) return ['SEVERE', 'orange'];
  return ['CRITICAL', 'red'];
}

function render() {
  draw();
  const p = pct(), dead = STATE.dead.size;
  const facNow = A.facilities.map(f => ({
    ...f, destroyed: STATE.dead.has(f.node),
    reachable: STATE.serviceable.has(f.node)
  }));
  const destroyed = facNow.filter(f => f.destroyed).length;
  const newly = facNow.filter(f => !f.destroyed && !f.reachable && wasOk(f.node)).length;

  // KPI tiles
  document.getElementById('kpis').innerHTML = [
    ['Connectivity', p.toFixed(0) + '%'],
    ['Junctions left', `${STATE.serviceable.size}/${BEFORE.nodes}`],
    ['Components', STATE.comps.length],
    ['Roads lost', dead],
  ].map(([l, v]) => `<div class="kpi"><div class="v">${v}</div><div class="l">${l}</div></div>`).join('');
  document.getElementById('connNote').textContent =
    `Baseline: ${BEFORE.nodes} junctions, ${BEFORE.components} component(s), largest ${BEFORE.largest}. ` +
    `Connectivity is measured on the largest surviving mutually-reachable network.`;

  // facilities
  document.getElementById('facList').innerHTML = facNow.map(f => {
    const s = f.destroyed ? '<span class="st bad">DESTROYED</span>'
      : f.reachable ? '<span class="st ok">REACHABLE</span>'
        : wasOk(f.node) ? '<span class="st warn">CUT OFF</span>'
          : '<span class="st" style="color:#7c8aa5">NOT LINKED</span>';
    return `<div class="fac"><span>${f.name}</span>${s}</div>`;
  }).join('') + `<div class="note">${A.realFacilities
    ? `Facilities are <b>real places</b> read from OpenStreetMap (hospitals, clinics, fire
       stations, police, shelters), each snapped to its nearest road junction.`
    : `Facilities are sample anchor points placed on the extracted
       network (not real hospitals) so the simulation can show who is cut off.`}
     <b>NOT LINKED</b> = it was already in a separate fragment of the network
     before the event, so the disaster did not cause that.</div>`;

  // critical junction table (pre vs post)
  const rows = N.map(n => ({ id: n.id, b: cen[n.id] || 0, a: STATE.bc[n.id] || 0 }))
    .map(o => ({ ...o, d: o.a - o.b })).sort((x, y) => y.d - x.d).slice(0, 12);
  document.getElementById('critTable').innerHTML =
    `<table><tr><th>Junction</th><th>Before</th><th>After</th><th>Change</th></tr>` +
    rows.map(r => `<tr><td>${r.id}</td><td class="num">${r.b.toFixed(3)}</td>` +
      `<td class="num">${r.a.toFixed(3)}</td>` +
      `<td class="num ${r.d > 0.001 ? 'lo' : ''}">${r.d >= 0 ? '+' : ''}${r.d.toFixed(3)}</td></tr>`).join('') +
    `</table><div class="note">Junctions whose importance rises after the event are the ones
       responders must protect first.</div>`;

  document.getElementById('mapbar').innerHTML =
    [`<span>${CUR.name}</span>`, dead ? `<span>${dead} junctions lost</span>` : '<span>baseline</span>',
     `<span>${p.toFixed(0)}% connected</span>`].join('');

  briefing(p, facNow, destroyed, newly);
  renderWatch();
  drawLocator();
}

/* ---------- briefing (mirrors emergency_assistant.py) ---------- */
function briefing(p, facNow, destroyed, newly) {
  const [sev, clr] = severity(p, destroyed);
  const lost = STATE.dead.size;
  const top = N.map(n => ({ id: n.id, d: (STATE.bc[n.id] || 0) - (cen[n.id] || 0) }))
    .sort((a, b) => b.d - a.d)[0];
  // "cut off" = was connected before and is not any more (NOT pre-existing gaps)
  const cut = facNow.filter(f => !f.destroyed && !f.reachable && wasOk(f.node));
  const never = facNow.filter(f => !f.destroyed && !f.reachable && !wasOk(f.node));
  const alive = facNow.filter(f => f.reachable && !f.destroyed);
  let step = 1;
  const pri = [];
  if (cut.length) { pri.push(`${step++}. Restore access to ${cut.map(f => f.name).join(', ')} — connected before the event, cut off now.`); }
  if (top && top.d > 0) pri.push(`${step++}. Junction ${top.id} is now the most load-bearing point (centrality +${top.d.toFixed(3)}) — protect it before any other junction.`);
  const gone = facNow.filter(f => f.destroyed);
  if (gone.length) pri.push(`${step++}. ${gone[0].name} is physically destroyed — triage there cannot be routed until a route is restored.`);
  if (p >= 90) pri.push(`${step++}. Network redundancy absorbs this event; maintain normal patrols.`);

  const note = p >= 90 ? (destroyed ? 'the network survives but a response facility is destroyed — local access, not connectivity, is the problem'
    : 'the network absorbs this event')
    : p >= 60 ? 'degradation is contained but response times rise in the affected corridor'
      : p >= 25 ? 'large parts of the network are cut off — emergency access is the priority'
        : 'the serviceable network has collapsed';

  document.getElementById('sevBadge').textContent = sev;
  document.getElementById('sevBadge').className = 'sev ' + clr;
  document.getElementById('headline').textContent = `${sev}: ${lost} junctions lost, ${p.toFixed(0)}% still connected`;
  document.getElementById('brief').innerHTML =
    `<h4>Situation</h4>${CUR.name} on the ${BEFORE.nodes}-junction network. ${lost} junctions destroyed.
Components ${BEFORE.components} → ${STATE.comps.length}; largest reachable network holds
${STATE.serviceable.size} of ${BEFORE.largest} junctions (${p.toFixed(1)}%).
${sev} — ${note}.
<h4>Impact</h4>${cut.length ? 'Newly cut off: ' + cut.map(f => f.name).join(', ') + '.' : 'No facility that was connected before is cut off by this event.'}
${never.length ? 'Already isolated before this event (pre-existing gap in the extracted network): ' + never.map(f => f.name).join(', ') + '.' : ''}
${gone.length ? 'Destroyed outright: ' + gone.map(f => f.name).join(', ') + '.' : ''}
${alive.length ? 'Still reachable: ' + alive.map(f => f.name).join(', ') + '.' : ''}
<h4>Priority</h4>${pri.map(x => '• ' + x).join('\n')}`;
}

/* ---------- Q&A (deterministic, answers from the live state) ---------- */
function answer(q) {
  const p = pct();
  const t = q.toLowerCase();
  const facNow = A.facilities.map(f => ({ ...f, destroyed: STATE.dead.has(f.node), reachable: STATE.serviceable.has(f.node) }));
  const top = N.map(n => ({ id: n.id, d: (STATE.bc[n.id] || 0) - (cen[n.id] || 0), a: STATE.bc[n.id] || 0 }))
    .sort((x, y) => y.d - x.d)[0];
  if (/area|areas|town|cities|city|where|which place|monitor|everywhere|region|kovilpatti|thoothukudi|madurai/.test(t) && AREA_ORDER.length > 1) {
    const rows = AREA_ORDER.map(quickStatus).sort((x, y) => x.p - y.p);
    const worst = rows.slice(0, 3).map(r =>
      `${r.label} ${r.p.toFixed(0)}% (${r.lost} junctions lost, ${r.cut} facilities cut off)`).join('; ');
    return `You are viewing ${(A.meta && A.meta.label) || curKey}. Across all ${rows.length} monitored
      areas under the same ${(CUR && CUR.name) || 'scenario'}, the worst affected are: ${worst}.
      Click any area in the "Monitored areas" panel to open it.`;
  }
  if (/isolat|cut off|unreach|who is|who can|linked/.test(t)) {
    const cut = facNow.filter(f => !f.reachable && !f.destroyed);
    const fresh = cut.filter(f => wasOk(f.node));
    const old = cut.filter(f => !wasOk(f.node));
    let msg = fresh.length ? `Cut off by the current ${CUR.name.toLowerCase()}: ${fresh.map(f => f.name).join(', ')}.`
      : 'Nothing that was connected before is cut off by the current settings.';
    if (old.length) msg += ` Already in a separate fragment before the event (pre-existing gap): ${old.map(f => f.name).join(', ')}.`;
    return msg;
  }
  if (/crit|importan|protect|junction|gatekeeper/.test(t))
    return top ? `Junction ${top.id} matters most right now: its betweenness centrality rose to ${top.a.toFixed(3)} (change +${top.d.toFixed(3)}) because the event removed its alternatives.` : 'No junction stands out yet.';
  if (/connect|percent|how bad|status|severity/.test(t))
    return `Connectivity is ${p.toFixed(1)}% of the pre-event largest component (${STATE.serviceable.size}/${BEFORE.largest} junctions), across ${STATE.comps.length} component(s). Severity: ${severity(p, facNow.filter(f => f.destroyed).length)[0]}.`;
  if (/what|how|explain|what.s going/.test(t))
    return (A.provenance || {}).metric
      ? `You are looking at the real street network of ${(A.meta && A.meta.label) || curKey}, ` +
        `downloaded from OpenStreetMap. The red disc/band is the disaster; red dashed roads are ` +
        `destroyed, greyed-out junctions can no longer reach the main network, and the remaining ` +
        `nodes are coloured by criticality (green low → red high). Blue dots are real hospitals, ` +
        `fire stations, police and shelters.`
      : `You are looking at a road network extracted from satellite imagery. The red disc/band is ` +
        `the disaster; red dashed roads are destroyed, greyed-out junctions can no longer reach the ` +
        `main network, and the remaining nodes are coloured by criticality (green low → red high).`;
  return `Ask about: which facilities are cut off · which junction matters most · how bad is the connectivity · what am I looking at.`;
}

/* ---------- pipeline + model tabs ---------- */
function buildTabs() {
  document.getElementById('pipe').innerHTML = D.pipeline.map((s, i) => `
    <div class="stage"><div class="ph"><span class="num">${i + 1}</span>${s.title}</div>
    <img src="${s.img}" alt="${s.title}"><div class="pc">${s.caption}</div></div>`).join('');
  const m = D.model;
  const isOsm = !!(A.provenance || {}).metric;
  document.getElementById('modelTable').innerHTML =
    `<table><tr><th>Model</th><th>IoU</th><th>Dice</th><th>Occlusion-Recall</th></tr>
     <tr><td>U-Net baseline (Part A)</td><td class="num">${m.base.iou.toFixed(3)}</td>
         <td class="num">${m.base.dice.toFixed(3)}</td><td class="num">${m.base.occ.toFixed(3)}</td></tr>
     <tr><td>Attention model (Part C)</td><td class="num hi">${m.att.iou.toFixed(3)}</td>
         <td class="num hi">${m.att.dice.toFixed(3)}</td><td class="num hi">${m.att.occ.toFixed(3)}</td></tr>
     <tr><td>Delta</td><td class="num">+${(m.att.iou - m.base.iou).toFixed(3)}</td>
         <td class="num">+${(m.att.dice - m.base.dice).toFixed(3)}</td>
         <td class="num lo">+${(m.att.occ - m.base.occ).toFixed(3)}</td></tr></table>`;
  document.getElementById('modelNote').innerHTML =
    `The attention model's gain is larger on occluded imagery than on clean imagery — that is the
     evidence it targets occlusion specifically. Healing: ${m.heal}` +
    (isOsm
      ? `<br><br><b>Where DeepGlobe does and does not appear:</b> the map above is
         <b>real OpenStreetMap vector data</b> — no image segmentation was involved in drawing it.
         DeepGlobe is what <b>trained</b> this U-Net (8,570 satellite tiles; 200 real patches used
         for train/val/test, giving the IoU and Dice above). The model is therefore a
         <i>fallback extractor</i>, not the map source: its job would be to recover roads in stretches
         where OpenStreetMap coverage is thin, then vectorise them into the same graph so a town like
         Kovilpatti is not penalised for incomplete OSM surveying. Nothing in the numbers on this
         page was used to compute the connectivity, destroyed-edge or cut-off-facility figures
         on the map — those come entirely from the OSM graph and the simulated hazard.`
      : '');

  // The "How the roads are formed" tab only makes sense when the map really was
  // built from imagery; hide it rather than leave an empty panel.
  const pipeTab = document.querySelector('.tab[data-tab="pipe"]');
  if (pipeTab && !(D.pipeline || []).length) {
    pipeTab.classList.add('hidden');
    document.getElementById('tab-pipe')?.classList.add('hidden');
  }

  document.querySelectorAll('.tab').forEach(t => t.onclick = () => {
    document.querySelectorAll('.tab').forEach(x => x.classList.remove('on'));
    document.querySelectorAll('[id^=tab-]').forEach(x => x.classList.add('hidden'));
    t.classList.add('on');
    document.getElementById('tab-' + t.dataset.tab).classList.remove('hidden');
  });
}

/* ---------- controls + interaction ---------- */
const ICONS = { earthquake: '🔴', flood: '🌊', blackout: '🚧' };

/* Scenario cards are rebuilt on every area change because each town's hazard
 * geometry is centred and sized from its own extent. */
function buildScenarioCards() {
  const el = document.getElementById('scn');
  if (!el) return;
  el.innerHTML = A.scenarios.map((s, i) =>
    `<button data-i="${i}" class="${s === CUR ? 'on' : ''}"><span class="ic">${ICONS[s.key] || '⚠'}</span>
     <span><b>${s.name}</b><small>${s.blurb}</small></span></button>`).join('');
  el.querySelectorAll('button').forEach(b => b.onclick = () => {
    el.querySelectorAll('button').forEach(x => x.classList.remove('on'));
    b.classList.add('on');
    CUR = A.scenarios[+b.dataset.i];
    syncControls();
    recompute();
  });
  syncControls();
}

function syncControls() {
  document.getElementById('ctlRadius').style.display = CUR.kind === 'disc' ? '' : 'none';
  document.getElementById('ctlBand').style.display = CUR.kind === 'band' ? '' : 'none';
  document.getElementById('ctlAngle').style.display = CUR.kind === 'band' ? '' : 'none';
}

function buildControls() {
  const r = document.getElementById('rRadius'), b = document.getElementById('rBand'), a = document.getElementById('rAngle');
  r.value = A.scale; b.value = A.bandScale; a.value = A.bandAngle || 35;
  r.oninput = () => { STATE.r = +r.value; document.getElementById('vRadius').textContent = fmtSize(STATE.r); schedule(); };
  b.oninput = () => { STATE.bw = +b.value; document.getElementById('vBand').textContent = fmtSize(STATE.bw); schedule(); };
  a.oninput = () => { STATE.ang = +a.value; document.getElementById('vAngle').textContent = STATE.ang + '°'; schedule(); };
  document.getElementById('btnReset').onclick = () => {
    EPIC = { x: A.extent.cx, y: A.extent.cy };
    r.value = A.scale; STATE.r = A.scale; document.getElementById('vRadius').textContent = fmtSize(STATE.r);
    b.value = A.bandScale; STATE.bw = A.bandScale; document.getElementById('vBand').textContent = fmtSize(STATE.bw);
    a.value = A.bandAngle || 35; STATE.ang = A.bandAngle || 35;
    document.getElementById('vAngle').textContent = STATE.ang + '°';
    recompute();
  };
  document.getElementById('btnFit').onclick = () => { fit(); draw(); };
  document.getElementById('btnDrag').onclick = () => {
    dragMode = !dragMode;
    document.getElementById('btnDrag').textContent = dragMode ? '✥ Click map to place' : '✥ Move epicentre';
  };
  document.getElementById('btnAsk').onclick = () => {
    const q = document.getElementById('ask').value.trim();
    document.getElementById('ans').textContent = q ? answer(q) : 'Type a question first.';
  };
  document.getElementById('ask').addEventListener('keydown', e => {
    if (e.key === 'Enter') document.getElementById('btnAsk').click();
  });
}

/* ---------- multi-area monitoring ----------
 * Runs the SAME hazard against every monitored town so the operator can see, at
 * a glance, which places are in trouble while sitting in any one of them.
 * Deliberately cheap: BFS only, no betweenness, so it stays live while the
 * sliders are being dragged. */
function quickStatus(key) {
  const a = AREAS[key], g = areaGraph(key);
  const sc = a.scenarios.find(s => s.key === (CUR && CUR.key)) || a.scenarios[0];
  const centre = { x: a.extent.cx, y: a.extent.cy };
  const dead = damagedSet(sc, STATE.r, STATE.bw, STATE.ang, a, centre);
  const adj = g.adj, base = g.base;
  // largest surviving component, seeded from any surviving baseline node
  let svcSet = new Set(), best = -1;
  for (const n of a.nodes) {
    if (dead.has(n.id) || !base.has(n.id) || svcSet.has(n.id)) continue;
    const comp = new Set([n.id]); const q = [n.id];
    while (q.length) {
      const u = q.pop();
      for (const v of adj.get(u) || []) {
        if (dead.has(v) || comp.has(v)) continue;
        comp.add(v); q.push(v);
      }
    }
    if (comp.size > best) { best = comp.size; svcSet = comp; }
  }
  let comps = 0;
  const seen = new Set();
  for (const n of a.nodes) {
    if (dead.has(n.id) || !base.has(n.id) || seen.has(n.id)) continue;
    comps++; const q = [n.id]; seen.add(n.id);
    while (q.length) {
      const u = q.pop();
      for (const v of adj.get(u) || []) if (!dead.has(v) && !seen.has(v)) { seen.add(v); q.push(v); }
    }
  }
  let cut = 0, gone = 0;
  for (const f of a.facilities) {
    if (dead.has(f.node)) { gone++; continue; }
    if (!svcSet.has(f.node) && base.has(f.node)) cut++;
  }
  return { key, label: (a.meta && a.meta.label) || key,
           role: (a.meta && a.meta.role) || 'watch',
           nodes: a.baseline.nodes, lost: dead.size, comps,
           p: a.baseline.largest ? 100 * svcSet.size / a.baseline.largest : 0,
           cut, gone };
}

function statusColour(p) {
  return p >= 90 ? '#39d98a' : p >= 60 ? '#ffc857' : p >= 25 ? '#ff9f43' : '#ff4757';
}

/* Keep the "on" highlight on whichever chip matches the open area. */
function markActiveArea() {
  document.querySelectorAll('#arealist .chip').forEach(c =>
    c.classList.toggle('on', c.dataset.k === curKey));
}

function buildAreaChips() {
  const el = document.getElementById('arealist');
  if (!el) return;
  el.innerHTML = AREA_ORDER.map(k =>
    `<span class="chip${k === curKey ? ' on' : ''}" data-k="${k}">${(AREAS[k].meta && AREAS[k].meta.label) || k}</span>`).join('');
  el.querySelectorAll('.chip').forEach(c => c.onclick = () => activateArea(c.dataset.k));
}

function renderWatch() {
  const box = document.getElementById('watch');
  if (!box) return;
  const rows = AREA_ORDER.map(quickStatus).sort((x, y) => x.p - y.p);
  box.innerHTML = rows.map(r =>
    `<div class="wrow${r.key === curKey ? ' here' : ''}" data-k="${r.key}">
      <div class="wn">${r.label}${r.role === 'home' ? ' <b class="tag">HOME</b>' : ''}</div>
      <div class="wbar"><i style="width:${Math.max(2, Math.min(100, r.p)).toFixed(1)}%;background:${statusColour(r.p)}"></i></div>
      <div class="wv">${r.p.toFixed(0)}%</div>
      <div class="ws">${r.lost} lost · ${r.cut} cut off</div>
    </div>`).join('') +
    `<div class="note">Every area is running the <b>same</b> ${(CUR && CUR.name) || 'scenario'} at the
     same real-world size. Click a row to open that town. Roads between towns are not drawn —
     the whole Tamil Nadu network is too large for OpenStreetMap to serve in one query, so
     this console never invents streets it has not actually downloaded.</div>`;
  box.querySelectorAll('.wrow').forEach(el => el.onclick = () => activateArea(el.dataset.k));
}

/* Regional locator: positions only, no fabricated street geometry. */
function drawLocator() {
  const c = document.getElementById('loc'), reg = D.region;
  if (!c || !reg || !reg.bbox) return;
  const r = c.getBoundingClientRect();
  if (c.width !== r.width * devicePixelRatio) { c.width = r.width * devicePixelRatio; c.height = r.height * devicePixelRatio; }
  const g = c.getContext('2d');
  g.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  g.clearRect(0, 0, c.width, c.height);
  const [s, w, n, e] = reg.bbox;
  const kx = Math.cos(((s + n) / 2) * Math.PI / 180);
  const pad = 16;
  const sc = Math.min((r.width - pad * 2) / ((e - w) * kx || 1),
                      (r.height - pad * 2) / (n - s || 1));
  const X = lon => pad + (lon - w) * kx * sc;
  const Y = lat => pad + (n - lat) * sc;
  g.font = '10px Segoe UI, sans-serif';
  for (const m of reg.markers) {
    const st = quickStatus(m.key);
    const x = X(m.center[1]), y = Y(m.center[0]);
    g.beginPath(); g.arc(x, y, m.key === curKey ? 8 : 5, 0, 7);
    g.fillStyle = statusColour(st.p); g.fill();
    g.lineWidth = m.key === curKey ? 2.5 : 1;
    g.strokeStyle = m.key === curKey ? '#fff' : '#0b1020'; g.stroke();
    if (m.key === curKey || m.role === 'home') {
      g.fillStyle = '#cfe0ff';
      g.fillText((m.label || '').split(' ')[0], x + 11, y + 3);
    }
  }
}

function init() {
  activateArea(START_KEY);

  // Say plainly where the map came from -- real OSM streets and a synthetic
  // demo grid are very different claims and must not look alike.
  const pv = A.provenance || {};
  const prov = document.getElementById('prov');
  if (prov) prov.innerHTML = pv.metric
    ? `Map data: <b>real streets from OpenStreetMap</b> — ${AREA_ORDER.length} monitored areas. ` +
      `Open map is <b>${pv.place || 'the selected area'}</b> (${pv.radius_m} m radius, ` +
      `${BEFORE.nodes} junctions, ${BEFORE.edges} streets). All distances are <b>real metres</b>. ` +
      `Hazards: ${pv.radius_label} blast, ${pv.band_label} flood half-width.`
    : `Map data: <b>procedurally generated demo network</b> — not a real place and not ` +
      `real satellite imagery. Coordinates are relative 0–1 grid values, not GPS.`;
  document.getElementById('pillNodes').textContent = BEFORE.nodes + ' junctions';
  document.getElementById('pillMode').textContent =
    (D.llm ? 'LLM ready' : 'offline engine') + ' · ' +
    (pv.metric ? 'real OSM streets' : 'synthetic demo network');
  document.getElementById('llmNote').textContent = D.llm
    ? 'An LLM key is configured (EMERGENCY_LLM=1): the facts are still computed locally and the model only rephrases them.'
    : 'This assistant runs fully offline: it reads the live simulation numbers and composes the briefing with a deterministic engine. Set OPENAI_API_KEY and EMERGENCY_LLM=1 to let a language model phrase it (numbers are always computed locally).';

  buildControls(); buildTabs(); buildAreaChips(); fit(); recompute();
  addEventListener('resize', () => { fit(); draw(); drawLocator(); });

  cv.addEventListener('wheel', e => {
    e.preventDefault();
    const r = cv.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
    const k = e.deltaY < 0 ? 1.12 : 1 / 1.12;
    view.ox = mx - (mx - view.ox) * k; view.oy = my - (my - view.oy) * k; view.s *= k;
    draw();
  });
  cv.addEventListener('mousemove', e => {
    const r = cv.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    if (dragging) { view.ox += mx - last.x; view.oy += my - last.y; last = { x: mx, y: my }; draw(); return; }
    let best = null, bd = 144;
    for (const n of N) {
      const d = (TX(n.x) - mx) ** 2 + (TY(n.y) - my) ** 2;
      if (d < bd) { bd = d; best = n.id; }
    }
    if (best !== hover) { hover = best; draw(); }
    cv.title = best ? 'junction ' + best : '';
  });
  cv.addEventListener('mousedown', e => {
    const r = cv.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
    if (dragMode) {
      EPIC = { x: (mx - view.ox) / view.s, y: (my - view.oy) / view.s };
      dragMode = false; document.getElementById('btnDrag').textContent = '✥ Move epicentre';
      recompute(); return;
    }
    dragging = true; last = { x: mx, y: my }; cv.classList.add('drag');
  });
  addEventListener('mouseup', () => { dragging = false; cv.classList.remove('drag'); });
  cv.addEventListener('mouseleave', () => { hover = null; draw(); });
}
init();

