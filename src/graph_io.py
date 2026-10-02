"""
PART C -- Shared graph contract I/O.

This module owns the ONLY interface between the extraction half and the
analysis / dashboard half:

    outputs/nodes.json
    outputs/edges.json
    outputs/graph.graphml

Node IDs are ALWAYS integers and consistent across the three files.

`load_graph()` transparently supports three sources (config.GRAPH_SOURCE):
    "osm"              -> download a small neighborhood via OSMnx (cached), with
                           an offline fallback to a built-in grid graph
    "teammate_export"  -> nodes.json + edges.json produced by the graph stage
    "graphml"          -> graph.graphml

Weight policy: edges without a numeric weight fall back to weight = 1.0 with a
printed note (the documented fallback).  Units are documented as pixels (or
meters for OSM edges, whose `length` attribute is used).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import networkx as nx

from src import utils
import config


# ---------------------------------------------------------------------------
# In-memory graph construction from the shared JSON files
# ---------------------------------------------------------------------------

def nodes_edges_to_graph(nodes: list, edges: list, default_weight: float = 1.0) -> nx.Graph:
    """Build an undirected nx.Graph from the shared nodes/edges schema (C1).

    Invalid references and non-numeric weights are handled gracefully and
    reported instead of crashing the pipeline.
    """
    G = nx.Graph()
    for e in nodes:
        try:
            nid = int(e["id"])
        except (KeyError, TypeError, ValueError):
            print(f"[graph] nodes.json contains a non-integer id {e.get('id')!r}; skipping")
            continue
        G.add_node(nid, lat=float(e.get("lat", 0.5)), lon=float(e.get("lon", 0.5)))

    for e in edges:
        try:
            s, t = int(e["source"]), int(e["target"])
        except (KeyError, TypeError, ValueError):
            print(f"[graph] skipping edge with bad source/target: {e!r}")
            continue
        if s not in G or t not in G:
            print(f"[graph] INVALID EDGE {s} -> {t}: references a missing node (skipped)")
            continue
        w = e.get("weight", None)
        if w is None:
            w = default_weight
            print(f"[graph] edge {s}->{t} has no weight; using fallback weight="
                  f"{default_weight} (documented)")
        try:
            w = float(w)
        except (TypeError, ValueError):
            print(f"[graph] edge {s}->{t} has non-numeric weight {w!r}; using "
                  f"fallback weight={default_weight}")
            w = default_weight
        if G.has_edge(s, t):
            G[s][t]["weight"] = min(G[s][t]["weight"], w)
        else:
            G.add_edge(s, t, weight=w)
    return G


def _load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(Path(path).read_text())


def load_teammate_export():
    """Mode 1 -- nodes.json + edges.json (teammate export)."""
    nodes = _load_json(config.TEAMMATE_NODES)
    edges = _load_json(config.TEAMMATE_EDGES)
    if nodes is None or edges is None:
        print("Teammate export not found. Expected:")
        print(f"    {config.TEAMMATE_NODES}")
        print(f"    {config.TEAMMATE_EDGES}")
        print("Run `python main.py --stage graph` or `python main.py --stage demo` first.")
        return None
    print(f"[graph] teammate export: {len(nodes)} nodes, {len(edges)} edges")
    return nodes_edges_to_graph(nodes, edges)


def load_graphml(path=config.TEAMMATE_GRAPHML):
    """Mode 2 -- graph.graphml (node IDs are always cast back to integers)."""
    path = Path(path)
    if not path.exists():
        print(f"GraphML file not found: {path}")
        return None
    G = nx.read_graphml(str(path))
    if not all(isinstance(n, int) for n in G.nodes):
        try:
            G = nx.relabel_nodes(G, {n: int(n) for n in G.nodes})
        except (TypeError, ValueError):
            G = nx.convert_node_labels_to_integers(G, first_label=0)
    for _u, _v, d in G.edges(data=True):
        w = d.get("weight", None)
        d["weight"] = float(w) if w is not None else 1.0
    print(f"[graph] graphml: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


# ---------------------------------------------------------------------------
# OSM placeholder (Part C, mode 3)
# ---------------------------------------------------------------------------

def _haversine_m(a, b) -> float:
    """Great-circle distance in metres between two {lat,lon} dicts."""
    rlat1, rlon1 = math.radians(a["lat"]), math.radians(a["lon"])
    rlat2, rlon2 = math.radians(b["lat"]), math.radians(b["lon"])
    dphi, dlmb = rlat2 - rlat1, rlon2 - rlon1
    h = math.sin(dphi / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlmb / 2) ** 2
    return 2.0 * 6371000.0 * math.asin(math.sqrt(h))


def _osm_to_undirected_shared(gd):
    """Convert an OSMnx MultiDiGraph into a simple undirected nx.Graph using
    edge 'length' (metres) as weight, with sequential integer node IDs."""
    Gu = nx.Graph()
    for nid, d in gd.nodes(data=True):
        Gu.add_node(nid, lat=float(d["y"]), lon=float(d["x"]))
    for u, v, d in gd.edges(data=True):
        w = d.get("length", None)
        if w is None:
            w = _haversine_m(Gu.nodes[u], Gu.nodes[v])
        if Gu.has_edge(u, v):
            Gu[u][v]["weight"] = min(Gu[u][v]["weight"], float(w))
        else:
            Gu.add_edge(u, v, weight=float(w))
    G = nx.convert_node_labels_to_integers(Gu, first_label=0, ordering="sorted")
    G.graph["coordinate_mode"] = "gps"
    G.graph["unit"] = "meters"
    return G


def synthetic_grid_graph(rows: int = 7, cols: int = 8) -> nx.Graph:
    """Built-in DEMO graph (relative pixel coordinates) for fully-offline demos.

    This is the honest fallback when the OSM download fails: coordinates are
    normalized [0,1] positions labelled as relative, never fake GPS.
    """
    G = nx.Graph()
    for r in range(rows):
        for c in range(cols):
            nid = r * cols + c
            G.add_node(nid, lat=float(1.0 - r / max(rows - 1, 1)),
                       lon=float(c / max(cols - 1, 1)))
    for r in range(rows):
        for c in range(cols):
            nid = r * cols + c
            if c + 1 < cols:
                G.add_edge(nid, nid + 1, weight=1.0)
            if r + 1 < rows:
                G.add_edge(nid, nid + cols, weight=1.0)
            if c + 1 < cols and r + 1 < rows:
                G.add_edge(nid, nid + cols + 1, weight=math.sqrt(2.0))  # diagonals
    G.graph["coordinate_mode"] = "relative"
    G.graph["unit"] = "pixels (relative demo grid)"
    G.graph["fallback"] = True
    print("[osm] using built-in DEMO grid graph (relative pixel coordinates, "
          "NOT geographic GPS). Set GRAPH_SOURCE and/or connectivity to change this.")
    return G


def load_osm_placeholder():
    """Mode 3 -- real neighborhood from OSMnx, cached on disk; offline fallback.

    Caches nodes.json / edges.json / graph.graphml under outputs/osm_cache so a
    network connection is only needed once.
    """
    utils.ensure_dirs(config.OSM_CACHE_DIR)
    cached = config.OSM_CACHE_DIR / "graph.graphml"
    if cached.exists() and not config.REFRESH_OSM_CACHE:
        print(f"[osm] using cached OSM graph -> {cached}")
        return load_graphml(cached)
    try:
        import osmnx
    except ImportError as exc:
        print(f"[osm] osmnx not available ({exc}). Falling back to DEMO grid graph.")
        return synthetic_grid_graph()
    try:
        print(f"[osm] downloading road network around {config.OSM_CENTER} (+/-{config.OSM_DIST} m, "
              f"{config.OSM_NETWORK_TYPE}) ...")
        mdg = osmnx.graph_from_point(config.OSM_CENTER, dist=config.OSM_DIST,
                                     network_type=config.OSM_NETWORK_TYPE)
        G = _osm_to_undirected_shared(mdg)
        print(f"[osm] OSM graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        save_graph(G, nodes_path=config.OSM_CACHE_DIR / "nodes.json",
                   edges_path=config.OSM_CACHE_DIR / "edges.json",
                   graphml_path=cached)
        return G
    except Exception as exc:
        print(f"[osm] OSM download failed ({exc}). Falling back to DEMO grid graph "
              f"(offline-safe).")
        return synthetic_grid_graph()


def overpass_call(fn, what: str = "query"):
    """Run an osmnx download, rotating Overpass mirrors with backoff.

    The primary overpass-api.de endpoint rate-limits aggressively; a single
    reset during a nine-area build used to drop that area entirely.  Each retry
    moves to the next mirror and waits a little longer.
    """
    import time
    mirrors = config.OSM_OVERPASS_MIRRORS or ["https://overpass-api.de/api"]
    tries = max(1, config.OSM_OVERPASS_TRIES)
    last = None
    for i in range(tries):
        url = mirrors[i % len(mirrors)]
        try:
            import osmnx
            osmnx.settings.overpass_url = url
            osmnx.settings.log_console = False
            # Several public mirrors (notably overpass.kumi.systems) reject
            # requests that carry no meaningful User-Agent, answering HTTP 429.
            osmnx.settings.http_user_agent = config.OSM_HTTP_USER_AGENT
            osmnx.settings.requests_timeout = config.OSM_HTTP_TIMEOUT_S
            return fn()
        except Exception as exc:
            last = exc
            print(f"[osm] {what}: {url.split('/')[2]} failed "
                  f"({type(exc).__name__}) - attempt {i + 1}/{tries}")
            if i + 1 < tries:
                time.sleep(config.OSM_OVERPASS_BACKOFF_S * (i + 1))
    raise last


def _way_is_drivable(tags: dict) -> bool:
    """Mirror of osmnx's `network_type="drive"` Overpass filter, in Python."""
    if tags.get("area") == "yes":
        return False
    if tags.get("access") == "private":
        return False
    if tags.get("motor_vehicle") == "no" or tags.get("motorcar") == "no":
        return False
    if tags.get("highway") in config.OSM_DRIVE_EXCLUDED_HIGHWAY:
        return False
    if tags.get("service") in config.OSM_DRIVE_EXCLUDED_SERVICE:
        return False
    return "highway" in tags


_CORE_XML_CACHE: dict = {}


def core_api_xml(center, dist: int):
    """Fetch + parse OSM XML for a bbox via the core API. Cached per area.

    The core API is a different server from Overpass and stayed responsive
    when every Overpass mirror was timing out or rate-limiting.  Returns
    (root_element, {node_id: (lon, lat)}).
    """
    import math
    import requests
    import xml.etree.ElementTree as ET

    key = (round(float(center[0]), 5), round(float(center[1]), 5), int(dist))
    if key in _CORE_XML_CACHE:
        return _CORE_XML_CACHE[key]

    lat0, lon0 = float(center[0]), float(center[1])

    def _fetch(d: int):
        dlat = d / 111_320.0
        dlon = d / (111_320.0 * max(math.cos(math.radians(lat0)), 1e-6))
        bbox = (lon0 - dlon, lat0 - dlat, lon0 + dlon, lat0 + dlat)
        print(f"[core-api] GET {config.OSM_CORE_API} bbox="
              f"{bbox[0]:.5f},{bbox[1]:.5f},{bbox[2]:.5f},{bbox[3]:.5f}")
        return requests.get(
            config.OSM_CORE_API,
            params={"bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"},
            headers={"User-Agent": config.OSM_HTTP_USER_AGENT},
            timeout=config.OSM_HTTP_TIMEOUT_S)

    # The core API answers HTTP 400 once a bbox holds more than ~50k nodes,
    # which dense city centres like Madurai exceed easily.  Shrinking the box
    # and retrying beats stalling on the rate-limited Overpass mirrors, so take
    # the largest box the server will actually serve.
    d = int(dist)
    resp = None
    for _ in range(4):
        resp = _fetch(d)
        if resp.status_code == 200:
            break
        if resp.status_code == 400 and d > 300:
            d = max(300, d // 2)
            print(f"[core-api] too many nodes for this bbox - retrying at {d} m")
            continue
        print(f"[core-api] HTTP {resp.status_code}")
        return None
    if resp is None or resp.status_code != 200:
        print(f"[core-api] HTTP {getattr(resp, 'status_code', '?')}")
        return None
    if d != int(dist):
        print(f"[core-api] serving a reduced {d} m box instead of {int(dist)} m")
    print(f"[core-api] {len(resp.content) / 1e6:.1f} MB of OSM XML")
    root = ET.fromstring(resp.content)
    nodes: dict = {}
    for el in root.findall("node"):
        try:
            nid = int(el.get("id"))
            nodes[nid] = (float(el.get("lon")), float(el.get("lat")))
        except (TypeError, ValueError):
            continue
    _CORE_XML_CACHE[key] = (root, nodes)
    return _CORE_XML_CACHE[key]


def facilities_from_core_api(center, dist: int):
    """Amenity POIs from the core API, shaped like the Overpass facility rows.

    Returns None when the fetch itself failed (so the caller can try Overpass),
    and an empty list when the fetch worked but the area genuinely has no
    matching amenities.  Those are very different situations and must not be
    confused, or a genuinely unmapped town triggers a pointless retry against
    the rate-limited Overpass mirrors.
    """
    got = core_api_xml(center, dist)
    if got is None:
        return None
    root, nodes = got
    wanted = {ts.partition("=")[2]
              for vals in config.OSM_FACILITY_TAGS.values() for ts in vals}
    out = []

    def _coord(el):
        try:
            return float(el.get("lon")), float(el.get("lat"))
        except (TypeError, ValueError):
            return None

    # Nodes first (the common case: a tagged point on the street).
    for el in root.findall("node"):
        tags = {t.get("k"): t.get("v") for t in el.findall("tag")}
        if tags.get("amenity") not in wanted:
            continue
        c = _coord(el)
        if c:
            out.append({"name": tags.get("name"), "amenity": tags.get("amenity"),
                        "lon": c[0], "lat": c[1]})

    # Then ways.  Hospitals, schools and clinics are frequently mapped as a
    # building polygon rather than a node, and scanning only nodes returned
    # ZERO real facilities for towns like Thoothukudi and Virudhunagar.
    for el in root.findall("way"):
        tags = {t.get("k"): t.get("v") for t in el.findall("tag")}
        if tags.get("amenity") not in wanted:
            continue
        pts = [nodes[int(nd.get("ref"))] for nd in el.findall("nd")
               if nd.get("ref") and int(nd.get("ref")) in nodes]
        if not pts:
            continue
        out.append({"name": tags.get("name"), "amenity": tags.get("amenity"),
                    "lon": sum(p[0] for p in pts) / len(pts),
                    "lat": sum(p[1] for p in pts) / len(pts)})
    return out


def graph_from_core_api(center, dist: int):
    """Build a drive network from the OSM core API instead of Overpass.

    The public Overpass mirrors rate-limit aggressively; the core API is a
    different server and stayed responsive when every Overpass mirror was
    timing out.  Returns an osmnx-style MultiDiGraph (x/y = lon/lat, plus
    lat/lon and length) or None.
    """
    import math

    got = core_api_xml(center, dist)
    if got is None:
        return None
    root, nodes = got

    mdg = nx.MultiDiGraph()
    for nid, (lon, lat) in nodes.items():
        mdg.add_node(nid, x=lon, y=lat, lon=lon, lat=lat)

    def _m(a, b) -> float:
        """Great-circle distance in metres."""
        la1, lo1, la2, lo2 = map(math.radians, (a[1], a[0], b[1], b[0]))
        h = (math.sin((la2 - la1) / 2) ** 2
             + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2)
        return 2 * 6_371_000.0 * math.asin(math.sqrt(h))

    added = 0
    for way in root.findall("way"):
        tags = {t.get("k"): t.get("v") for t in way.findall("tag")}
        if not _way_is_drivable(tags):
            continue
        refs = [int(nd.get("ref")) for nd in way.findall("nd")
                if nd.get("ref") and int(nd.get("ref")) in nodes]
        oneway = tags.get("oneway") in ("yes", "true", "1")
        for u, v in zip(refs, refs[1:]):
            if u == v:
                continue
            length = _m(nodes[u], nodes[v])
            if length <= 0:
                continue
            mdg.add_edge(u, v, length=length, highway=tags.get("highway"),
                         name=tags.get("name"), oneway=oneway)
            added += 1
            if not oneway:
                mdg.add_edge(v, u, length=length, highway=tags.get("highway"),
                             name=tags.get("name"), oneway=False)
    # drop junction nodes that no drivable street touches
    mdg.remove_nodes_from([n for n, d in mdg.degree() if d == 0])
    raw_nodes, raw_edges = mdg.number_of_nodes(), mdg.number_of_edges()
    if mdg.number_of_nodes() == 0:
        return None
    # The core API returns EVERY geometry node on a way; osmnx's Overpass path
    # simplifies those away, leaving only real intersections and dead-ends.
    # Without this, a town reports thousands of fake "junctions", which would
    # inflate the damage counts and skew every connectivity percentage.
    try:
        import osmnx
        mdg = osmnx.simplify_graph(mdg, remove_rings=True)
    except Exception as exc:
        print(f"[core-api] simplification skipped ({exc})")
    print(f"[core-api] {raw_nodes} raw nodes -> {mdg.number_of_nodes()} intersections, "
          f"{mdg.number_of_edges()} directed edges")
    return mdg


def resolve_area_graph(area: dict, refresh: bool = False, tag: str = ""):
    """Load one monitored area, growing the radius when OSM coverage is thin.

    Returns (graph, dist).  `dist` is the radius that actually succeeded, which
    is usually LARGER than the configured one: a 1.2 km box around a small town
    can come back with 40 intersections, which is not enough to simulate on, so
    the box is doubled until the network is usable.  The cache directory name
    carries the radius (`<key>_<dist>`) so the grown graphs live side by side.

    This lives here, not in the build script, because both
    scripts/build_web_dashboard.py and scripts/fetch_basemap.py must resolve an
    area to the SAME graph.  When they disagreed, an area that had grown to a
    6 km radius shipped with a basemap stitched for the original 1.5 km box --
    the map simply stopped part-way across the city.
    """
    key = area["key"]
    dist = int(area["dist"])
    G = None
    for attempt in range(config.OSM_AREA_EXPANSION_TRIES + 1):
        cache_key = key if attempt == 0 else f"{key}_{dist}"
        G = load_area_graph(
            area["center"], dist, config.OSM_AREA_CACHE / cache_key,
            refresh=refresh)
        if G is None or G.number_of_nodes() == 0:
            G = None
            break
        if G.number_of_nodes() >= config.OSM_MIN_AREA_NODES:
            break
        if dist >= config.OSM_MAX_AREA_RADIUS_M:
            break
        grown = min(dist * 2, config.OSM_MAX_AREA_RADIUS_M)
        print(f"[{tag or 'area-graph'}]   {key}: only {G.number_of_nodes()} "
              f"intersections at {dist} m (OSM coverage is thin here) - "
              f"retrying at {grown} m")
        dist = grown
    if G is None:
        return None, int(area["dist"])

    meta = dict(area)
    meta["dist"] = dist
    if dist != area["dist"]:
        meta["label"] = f"{area['label']} ({dist // 1000} km)"
    return G, meta


def load_area_graph(center, dist: int, cache_dir, network_type: str | None = None,
                    refresh: bool = False):
    """Drive network for ONE monitored area, cached in its own directory.

    This is the multi-area counterpart of load_osm_placeholder(): same
    conversion and the same on-disk contract, but the centre, radius and cache
    location are passed in so several towns can live side by side without
    overwriting each other.
    """
    cache_dir = Path(cache_dir)
    network_type = network_type or config.OSM_NETWORK_TYPE
    utils.ensure_dirs(cache_dir)
    cached = cache_dir / "graph.graphml"
    if cached.exists() and not refresh:
        print(f"[area-graph] using cached {cached}")
        return load_graphml(cached)
    try:
        import osmnx
    except ImportError as exc:
        print(f"[area-graph] osmnx not available ({exc}).")
        return None
    try:
        print(f"[area-graph] downloading {network_type} network around "
              f"{center} (+/-{dist} m) ...")
        mdg = None
        core_answered = False
        if config.OSM_PREFER_CORE_API and int(dist) <= config.OSM_MAX_CORE_API_RADIUS_M:
            mdg = graph_from_core_api(center, dist)
            # A successful fetch that found no drivable ways is a real answer:
            # the bbox genuinely has no mapped streets there and Overpass reads
            # the same data, so retrying there would only stall.  Returning None
            # lets the caller grow the radius instead.
            core_answered = mdg is not None or core_api_xml(center, dist) is not None
        try:
            if mdg is None and not core_answered:
                def _go():
                    import osmnx
                    return osmnx.graph_from_point(tuple(center), dist=int(dist),
                                                  network_type=network_type)
                mdg = overpass_call(_go, f"graph @ {center}")
        except Exception as exc:
            if mdg is not None:
                raise
            if not config.OSM_USE_CORE_API_FALLBACK:
                raise
            print(f"[area-graph] Overpass unavailable ({exc}) - "
                  f"falling back to the OSM core API")
            mdg = graph_from_core_api(center, dist)
        if mdg is None:
            if core_answered:
                print("[area-graph] no drivable streets in this bbox "
                      "(OSM coverage gap) - a wider radius may help")
            else:
                print("[area-graph] both the core API and Overpass failed")
            return None
        G = _osm_to_undirected_shared(mdg)
        print(f"[area-graph] {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        save_graph(G, nodes_path=cache_dir / "nodes.json",
                   edges_path=cache_dir / "edges.json", graphml_path=cached)
        return G
    except Exception as exc:
        print(f"[area-graph] download failed ({exc}).")
        return None


def load_graph(source: str | None = None):
    """Dispatch on config.GRAPH_SOURCE (Part C / Part G).

    Returns an undirected nx.Graph or None when the configured source is
    unavailable and no fallback exists.
    """
    source = source or config.GRAPH_SOURCE
    if source == "osm":
        return load_osm_placeholder()
    if source == "teammate_export":
        G = load_teammate_export()
        if G is None:
            print("[graph] teammate export missing -> falling back to OSM placeholder.")
            return load_osm_placeholder()
        return G
    if source == "graphml":
        G = load_graphml(config.TEAMMATE_GRAPHML)
        if G is None:
            print("[graph] graph.graphml missing -> falling back to OSM placeholder.")
            return load_osm_placeholder()
        return G
    print(f"[graph] unknown GRAPH_SOURCE={source!r}. Valid: osm | teammate_export | graphml. "
          f"Falling back to OSM.")
    return load_osm_placeholder()
# ---------------------------------------------------------------------------
# Shared contract serialization (B9)
# ---------------------------------------------------------------------------

def _collect_shared_data(G: nx.Graph):
    """Return ({nodes: [...], edges: [...]}) in the exact shared schema.

    nodes.json entries: {"id": int, "lat": float, "lon": float}
    edges.json entries: {"source": int, "target": int, "weight": float}
    """
    nodes = []
    for nid in G.nodes():
        d = G.nodes[nid]
        nodes.append({
            "id": int(nid),
            "lat": float(d.get("lat", 0.5)),
            "lon": float(d.get("lon", 0.5)),
        })
    edges = []
    for u, v, d in G.edges(data=True):
        edges.append({
            "source": int(u),
            "target": int(v),
            "weight": float(d.get("weight", 1.0)),
        })
    return {"nodes": nodes, "edges": edges}


def save_graph(G: nx.Graph,
               nodes_path=config.TEAMMATE_NODES,
               edges_path=config.TEAMMATE_EDGES,
               graphml_path=config.TEAMMATE_GRAPHML,
               print_summary: bool = True) -> None:
    """Write the graph to nodes.json + edges.json + graph.graphml.

    Node IDs are guaranteed integer (relabeled when necessary) and consistent
    across all three files (spec B9).
    """
    if any(not isinstance(n, int) for n in G.nodes()):
        G = nx.convert_node_labels_to_integers(G, first_label=0)
    data = _collect_shared_data(G)
    for path, payload in ((nodes_path, data["nodes"]), (edges_path, data["edges"])):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2))
    Path(graphml_path).parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(G, str(graphml_path), infer_numeric_types=True)
    if print_summary:
        print(f"[graphio] wrote {len(data['nodes'])} nodes / {len(data['edges'])} edges")
        print(f"      -> {nodes_path.name}, {edges_path.name}, {graphml_path.name}")
        print("      Coordinates are RELATIVE pixel coordinates unless the source was "
              "OSM/GPS.\n      Edge weights are in PIXELS unless a pixel-to-metre scale "
              "is configured.")
    return G