"""
Emergency scenario simulation for the extracted road graph.

Turns the static graph into something with an obvious *use*: pick a disaster,
see which roads are destroyed, which areas get cut off, and which junctions
matter after the event.

The simulation is deterministic (seeded) and lives in Python so it is testable
and reproducible; the web dashboard re-implements the same BFS in JavaScript so
the user can move sliders and watch the network react live.

Damage model (deliberately simple and explainable):
  * an earthquake = a disc of destroyed road around an epicentre,
  * a flood     = a band (a wide corridor) along one axis,
  * a closure   = a set of hand-picked gatekeeper junctions taken out.
"""

from __future__ import annotations

import json
import math
import sys
from collections import deque
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from src import utils
from src import graph_io


# WGS84 approximations -- plenty accurate for a few-km neighbourhood, which is
# the scale a road-resilience scenario is about.
_M_PER_DEG_LAT = 110574.0
_M_PER_DEG_LON_EQUATOR = 111320.0


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def is_gps_graph(G: nx.Graph) -> bool:
    """True when node coordinates are real WGS84 lat/lon (not pixel/relative)."""
    return str(G.graph.get("coordinate_mode", "")).lower() == "gps"


def _origin(G: nx.Graph):
    """(lon0, lat0) centre of the graph, cached on the graph object."""
    if "_lon0" not in G.graph:
        lats = [d["lat"] for _, d in G.nodes(data=True) if "lat" in d]
        lons = [d["lon"] for _, d in G.nodes(data=True) if "lon" in d]
        G.graph["_lon0"] = sum(lons) / len(lons) if lons else 0.0
        G.graph["_lat0"] = sum(lats) / len(lats) if lats else 0.0
    return G.graph["_lon0"], G.graph["_lat0"]


def lonlat_to_meters(lon: float, lat: float, lon0: float, lat0: float):
    """Equirectangular projection of (lon, lat) to local metres east/north of
    (lon0, lat0).  The cos(lat0) factor is what keeps the aspect ratio correct:
    one degree of longitude is shorter than one degree of latitude away from
    the equator, and without it the map is visibly stretched."""
    k = _M_PER_DEG_LON_EQUATOR * math.cos(math.radians(lat0))
    return (lon - lon0) * k, (lat - lat0) * _M_PER_DEG_LAT


def graph_extent(G: nx.Graph):
    """(min_x, min_y, max_x, max_y) in METRES for a GPS graph, otherwise in the
    graph's own units (pixels/relative)."""
    if is_gps_graph(G):
        lon0, lat0 = _origin(G)
        pts = [lonlat_to_meters(d["lon"], d["lat"], lon0, lat0)
               for _, d in G.nodes(data=True) if "lat" in d and "lon" in d]
        if pts:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            return min(xs), min(ys), max(xs), max(ys)
    xs = [d["x"] for _, d in G.nodes(data=True) if "x" in d]
    ys = [d["y"] for _, d in G.nodes(data=True) if "y" in d]
    if not xs:                       # geo graph: fall back to lon/lat
        xs = [d["lon"] for _, d in G.nodes(data=True) if "lon" in d]
        ys = [d["lat"] for _, d in G.nodes(data=True) if "lat" in d]
    return min(xs), min(ys), max(xs), max(ys)


def node_xy(G: nx.Graph, n):
    """Node position in METRES (GPS graph) or graph units (pixel/relative)."""
    d = G.nodes[n]
    if is_gps_graph(G) and "lat" in d and "lon" in d:
        lon0, lat0 = _origin(G)
        return lonlat_to_meters(d["lon"], d["lat"], lon0, lat0)
    return (d.get("x", d.get("lon", 0.0)), d.get("y", d.get("lat", 0.0)))


def edge_midpoint(G: nx.Graph, u, v):
    (ux, uy), (vx, vy) = node_xy(G, u), node_xy(G, v)
    return ((ux + vx) / 2.0, (uy + vy) / 2.0)


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

def build_scenarios(G: nx.Graph) -> list:
    """Data-driven scenarios sized to the actual graph extent.

    On a real GPS graph the hazards are sized in METRES (config.HAZARD_*_M), so
    "400 m blast radius" means 400 m regardless of how much city was downloaded.
    On the relative/pixel demo graph we fall back to fractions of the diagonal.
    """
    x0, y0, x1, y1 = graph_extent(G)
    w, h = (x1 - x0) or 1.0, (y1 - y0) or 1.0
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    diag = math.hypot(w, h)
    metric = is_gps_graph(G) and config.USE_METRIC_HAZARDS

    if metric:
        radius = config.HAZARD_EARTHQUAKE_RADIUS_M
        half_w = config.HAZARD_FLOOD_HALF_WIDTH_M
        slide_w = config.HAZARD_LANDSLIDE_HALF_WIDTH_M
        units = "m"
    else:
        radius = config.HAZARD_EARTHQUAKE_FRACTION * diag
        half_w = config.HAZARD_FLOOD_HALF_WIDTH_FRACTION * diag
        slide_w = config.HAZARD_LANDSLIDE_HALF_WIDTH_FRACTION * diag
        units = "px"

    return [
        {"key": "earthquake", "name": "Earthquake",
         "blurb": f"A blast centred on the middle of the region destroys every "
                  f"road within {radius:.0f} {units}.",
         "kind": "disc", "cx": cx, "cy": cy,
         "radius": radius, "radius_label": f"{radius:.0f} {units}"},
        {"key": "flood", "name": "Flood",
         "blurb": f"A river cuts across the region as a band {2 * half_w:.0f} "
                  f"{units} wide; roads inside it are submerged.",
         "kind": "band", "angle": config.HAZARD_FLOOD_ANGLE_DEG,
         "half_width": half_w, "half_width_label": f"{half_w:.0f} {units}"},
        {"key": "landslide", "name": "Landslide",
         "blurb": f"A debris slide runs down a narrow {2 * slide_w:.0f} {units} "
                  f"corridor; only roads inside the slide are lost.",
         "kind": "band", "angle": config.HAZARD_LANDSLIDE_ANGLE_DEG,
         "half_width": slide_w, "half_width_label": f"{slide_w:.0f} {units}",
         "family": "corridor"},
        {"key": "blackout", "name": "Key junction closures",
         "blurb": "No physical destruction - the top gatekeeper junctions are "
                  "simply closed (maintenance / congestion / attack).",
         "kind": "nodes", "count": config.HAZARD_BLACKOUT_NODES,
         "family": "closure"},
    ]


def apply_scenario(G: nx.Graph, scenario: dict, gatekeepers: list | None = None):
    """Remove the road network inside the hazard zone.

    Returns (G_damaged, damage_info).
    """
    H = G.copy()
    kind = scenario["kind"]
    failed_nodes, failed_edges = set(), set()

    if kind == "disc":
        cx, cy, r = scenario["cx"], scenario["cy"], scenario["radius"]
        for n in list(H.nodes()):
            nx_, ny_ = node_xy(H, n)
            if math.hypot(nx_ - cx, ny_ - cy) <= r:
                failed_nodes.add(n)
    elif kind == "band":
        ang = math.radians(scenario["angle"])
        x0, y0, x1, y1 = graph_extent(G)
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        dx, dy = math.cos(ang), math.sin(ang)
        for n in list(H.nodes()):
            nx_, ny_ = node_xy(H, n)
            if abs((nx_ - cx) * (-dy) + (ny_ - cy) * dx) <= scenario["half_width"]:
                failed_nodes.add(n)
    elif kind == "nodes":
        failed_nodes.update(list(gatekeepers or [])[: scenario.get("count", 3)])

    # Count destroyed edges BEFORE deleting the nodes.  Removing the nodes first
    # also removes their incident edges, so doing it in this order reported
    # "0 edges destroyed" for every scenario.
    failed_edges = {(u, v) for u, v in H.edges()
                    if u in failed_nodes or v in failed_nodes}
    for n in failed_nodes:
        if n in H:
            H.remove_node(n)

    info = {"failed_node_count": len(failed_nodes),
            "failed_edge_count": len(failed_edges),
            "failed_nodes": sorted(int(n) for n in failed_nodes)}
    return H, info


# ---------------------------------------------------------------------------
# Places (facilities) so "who is cut off" is meaningful
# ---------------------------------------------------------------------------

def load_real_facilities(G: nx.Graph) -> list | None:
    """Look up REAL emergency facilities from OSM POIs near the graph.

    Returns [{name, kind, lon, lat, node}] snapped to the nearest network node,
    or None when the graph is not a real GPS graph, the setting is off, or the
    lookup fails (the caller then falls back to evenly spread anchors).

    Cached in one JSON file keyed by the graph's own centre + radius, so a
    multi-area build keeps every area's facilities instead of overwriting them.
    """
    if not is_gps_graph(G) or not config.OSM_USE_REAL_FACILITIES:
        return None

    # Derive the search centre/radius from the GRAPH, not from config, so this
    # works unchanged for every area in a multi-area build.
    lats = [d["lat"] for _, d in G.nodes(data=True) if "lat" in d]
    lons = [d["lon"] for _, d in G.nodes(data=True) if "lon" in d]
    if not lats:
        return None
    clat, clon = sum(lats) / len(lats), sum(lons) / len(lons)
    lat0, lon0 = _origin(G)
    tx, ty = lonlat_to_meters(clon, clat, lon0, lat0)
    radius_m = max(math.hypot(px - tx, py - ty)
                   for px, py in (lonlat_to_meters(lo, la, lon0, lat0)
                                  for lo, la in zip(lons, lats)))
    # The search must scale with the graph.  A hard 3 km cap silently reported
    # ZERO facilities for areas auto-expanded to a larger radius (Virudhunagar
    # has no tagged amenities within 3 km but 4 hospitals within 4.8 km), which
    # quietly demoted a real town to synthetic anchor points.
    search_m = int(min(max(radius_m * 1.25, 300), config.OSM_FACILITY_SEARCH_MAX_M))

    cache = config.OSM_FACILITY_CACHE
    key = f"{clat:.4f},{clon:.4f},{search_m}"
    blob: dict = {}
    if cache.exists():
        try:
            loaded = json.loads(cache.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                blob = loaded
        except (json.JSONDecodeError, OSError):
            blob = {}                   # corrupt cache -> just refetch
    if blob.get(key):
        return blob[key]

    try:
        import osmnx as ox
    except ImportError:
        print("[facilities] osmnx unavailable - falling back to anchor points")
        return None

    pois: list = []
    # One Overpass round-trip for every tag group rather than one per group:
    # the amenity key is shared, so a single merged query is both faster and far
    # gentler on the rate-limited public endpoints.
    tags: dict = {}
    for tag_strings in config.OSM_FACILITY_TAGS.values():
        for ts in tag_strings:
            k, _, v = ts.partition("=")
            tags.setdefault(k, []).append(v)
    tags = {k: sorted(set(v)) for k, v in tags.items()}
    tag_to_kinds: dict = {}
    for kind, tag_strings in config.OSM_FACILITY_TAGS.items():
        for ts in tag_strings:
            k, _, v = ts.partition("=")
            tag_to_kinds.setdefault((k, v), []).append(kind)

    rows = None
    if config.OSM_PREFER_CORE_API and search_m <= config.OSM_MAX_CORE_API_RADIUS_M:
        # Same amenity data, but from the core API, which was answering in
        # seconds while every Overpass mirror was timing out.
        got = graph_io.facilities_from_core_api((clat, clon), search_m)
        if got is not None:
            rows = got
            print(f"[facilities] core API returned {len(rows)} candidate POIs")
    if rows is None:
        # Only reached when the core API could not answer at all.  An empty
        # result from a working fetch is kept as-is: retrying Overpass then
        # would just stall on the rate-limited mirrors for no gain.
        try:
            feats = graph_io.overpass_call(
                lambda: ox.features_from_point((clat, clon), tags, dist=search_m),
                f"facilities @ ({clat:.3f},{clon:.3f})")
            if feats is not None:
                rows = [r for _, r in feats.iterrows()]
        except Exception as exc:
            print(f"[facilities] Overpass POI lookup failed ({exc})")

    if rows is not None:
        for row in rows:
            row = dict(row)
            # Three different row shapes have to be handled here:
            #   * osmnx 2.x GeoDataFrame -> a `geometry` column (possibly a
            #     building polygon, so fall back to its centroid)
            #   * osmnx 1.x             -> plain `x` / `y` columns
            #   * core API              -> plain `lon` / `lat` columns
            geom = row["geometry"] if "geometry" in row else None
            if geom is not None:
                if not hasattr(geom, "x"):          # Polygon / MultiPolygon
                    geom = geom.centroid
                plon, plat = float(geom.x), float(geom.y)
            elif "lon" in row and "lat" in row:
                plon, plat = float(row["lon"]), float(row["lat"])
            elif "x" in row and "y" in row:
                plon, plat = float(row["x"]), float(row["y"])
            else:
                continue
            if not (math.isfinite(plon) and math.isfinite(plat)):
                continue
            raw = row.get("name")
            if raw is None or (isinstance(raw, float) and math.isnan(raw)):
                raw = None
            amenity = row.get("amenity")
            kinds = tag_to_kinds.get(("amenity", amenity), [])
            if not kinds:
                continue
            # A hospital that is also a fire station is reported under the first
            # matching kind only, so the map does not stack duplicate markers.
            kind = kinds[0]
            label = str(raw) if raw else str(amenity or kind).replace("_", " ").title()
            pois.append({"name": label, "kind": kind, "lon": plon, "lat": plat})

    if not pois:
        return None

    # Snap each POI to its nearest intersection and drop duplicates that land on
    # the same node, so the map does not stack five markers on one junction.
    nodes = list(G.nodes())
    out, used = [], set()
    for p in pois:
        tx, ty = lonlat_to_meters(p["lon"], p["lat"], *_origin(G))
        best = min(nodes, key=lambda n: (node_xy(G, n)[0] - tx) ** 2 +
                   (node_xy(G, n)[1] - ty) ** 2)
        if best in used:
            continue
        used.add(best)
        out.append({**p, "node": int(best)})

    # Keep it usable: one per kind first, then fill by proximity to the centre.
    by_kind: dict = {}
    for f in out:
        by_kind.setdefault(f["kind"], []).append(f)
    ordered: list = []
    for kind in config.OSM_FACILITY_TAGS:
        ordered.extend(by_kind.get(kind, [])[:2])
    for f in out:
        if f not in ordered:
            ordered.append(f)
    ordered = ordered[:14]

    try:
        utils.ensure_dirs(cache.parent)
        blob[key] = ordered            # keep every other area's entries intact
        cache.write_text(json.dumps(blob, indent=1), encoding="utf-8")
    except OSError:
        pass
    print(f"[facilities] {len(ordered)} real OSM facilities for "
          f"({clat:.4f}, {clon:.4f}) r={search_m} m "
          f"({', '.join(sorted(by_kind)) or 'none'})")
    return ordered


def place_facilities(G: nx.Graph, pois: list | None = None) -> list:
    """Facilities to answer "is this place still reachable?".

    Uses REAL OSM POIs when they are available; otherwise falls back to evenly
    spread anchor points, which are NOT real hospitals and are labelled as such.
    """
    if pois:
        return [{"name": p["name"], "kind": p["kind"], "node": int(p["node"]),
                 "lon": p["lon"], "lat": p["lat"]} for p in pois]

    x0, y0, x1, y1 = graph_extent(G)
    specs = [
        ("Central Hospital", 0.50, 0.50, "hospital"),
        ("Fire Station North", 0.25, 0.22, "fire"),
        ("Police Station East", 0.78, 0.40, "police"),
        ("Shelter West", 0.18, 0.74, "shelter"),
        ("Depot South-East", 0.82, 0.78, "depot"),
    ]
    nodes = list(G.nodes())
    out = []
    for name, fx, fy, kind in specs:
        if not nodes:
            break
        tx, ty = x0 + fx * (x1 - x0), y0 + fy * (y1 - y0)
        best = min(nodes, key=lambda n: (node_xy(G, n)[0] - tx) ** 2 +
                   (node_xy(G, n)[1] - ty) ** 2)
        out.append({"name": name, "kind": kind, "node": int(best),
                    "x": tx, "y": ty})
    return out


# ---------------------------------------------------------------------------
# Analysis of the damage
# ---------------------------------------------------------------------------

def reachable_from(G_damaged: nx.Graph, source: int) -> set:
    """BFS from `source` over the damaged network."""
    if source not in G_damaged:
        return set()
    seen = {source}
    q = deque([source])
    while q:
        n = q.popleft()
        for m in G_damaged.neighbors(n):
            if m not in seen:
                seen.add(m)
                q.append(m)
    return seen


def analyze_damage(G: nx.Graph, G_damaged: nx.Graph, facilities: list,
                   centrality: dict | None = None) -> dict:
    """Before/after comparison: connectivity, isolation, and critical nodes."""
    comps_before = sorted((len(c) for c in nx.connected_components(G)), reverse=True)
    comps_after = sorted((len(c) for c in nx.connected_components(G_damaged)),
                         reverse=True) if G_damaged.number_of_nodes() else [0]

    serviceable = set()
    if G_damaged.number_of_nodes():
        serviceable = max(nx.connected_components(G_damaged), key=len)

    # what was reachable BEFORE the event - needed so the briefing reports only
    # places the disaster NEWLY cut off (the graph may already be fragmented)
    before_serviceable = set()
    if G.number_of_nodes():
        before_serviceable = max(nx.connected_components(G), key=len)

    facility_status = []
    for f in facilities:
        alive = f["node"] in G_damaged
        was_ok = f["node"] in before_serviceable
        now_ok = f["node"] in serviceable
        facility_status.append({
            **f,
            "destroyed": not alive,
            "reachable_before": was_ok,
            "reachable": now_ok,
            "newly_isolated": was_ok and not now_ok,
        })

    post_centrality = {}
    if G_damaged.number_of_nodes() > 2:
        sub = G_damaged.subgraph(serviceable)
        post_centrality = {int(k): float(v) for k, v in
                           nx.betweenness_centrality(sub, weight="weight").items()}
    pre_centrality = {int(k): float(v) for k, v in (centrality or {}).items()}

    movers = [{"node": n, "before": pre_centrality.get(n, 0.0), "after": c,
               "delta": c - pre_centrality.get(n, 0.0)}
              for n, c in post_centrality.items()]
    movers.sort(key=lambda d: -d["delta"])

    return {
        "nodes_before": G.number_of_nodes(),
        "nodes_after": G_damaged.number_of_nodes(),
        "edges_before": G.number_of_edges(),
        "edges_after": G_damaged.number_of_edges(),
        "components_before": len(comps_before),
        "components_after": len(comps_after),
        "largest_before": comps_before[0] if comps_before else 0,
        "largest_after": comps_after[0] if comps_after else 0,
        "connectivity_pct": round(
            100.0 * (comps_after[0] if comps_after else 0) / max(comps_before[0], 1), 1),
        "serviceable_nodes": len(serviceable),
        "facilities": facility_status,
        "isolated_facilities": [f["name"] for f in facility_status
                                if not f["reachable"]],
        "newly_isolated_facilities": [f["name"] for f in facility_status
                                      if f["newly_isolated"]],
        "post_centrality_top": movers[:10],
    }
