"""
Build the self-contained emergency console (outputs/emergency_dashboard.html).

Everything (graph, metrics, pipeline images) is embedded into a single HTML file
that opens in any browser with no server and no internet.  The JS engine mirrors
src/simulation.py so the hazard sliders respond live.

    python scripts/build_web_dashboard.py
    python scripts/build_web_dashboard.py --source osm     # use the OSM demo graph
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import networkx as nx

import config
from src import graph_io, simulation as sim, emergency_assistant as ea

WEB = ROOT / "web"
OUT = config.OUTPUTS_DIR / "emergency_dashboard.html"


def b64_img(path: Path, max_side: int = 420):
    """Downscale a PNG to a small data: URI so the HTML stays light."""
    if not path or not Path(path).exists():
        return None
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)))
    ok, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode() if ok else None


def area_payload(G, meta: dict) -> dict:
    """Everything the browser needs to simulate ONE area.

    Nodes are emitted in the SAME coordinate space as `extent` -- metres for a
    real GPS graph, native units otherwise.  Raw lat/lon ride along separately
    for provenance.  Mixing degrees here with a metric extent would silently
    break every distance test in the browser.
    """
    nodes = []
    for n, d in G.nodes(data=True):
        gx, gy = sim.node_xy(G, n)
        rec = {"id": int(n), "x": round(float(gx), 1), "y": round(float(gy), 1)}
        if "lat" in d:
            rec["lat"] = round(float(d["lat"]), 6)
        if "lon" in d:
            rec["lon"] = round(float(d["lon"]), 6)
        nodes.append(rec)
    edges = [{"s": int(u), "t": int(v)} for u, v in G.edges()]

    centrality, gatekeepers = {}, []
    if config.RESULTS_JSON.exists() and not meta.get("fresh"):
        res = json.loads(config.RESULTS_JSON.read_text())
        centrality = {e["node_id"]: e["score"] for e in res.get("centrality", [])}
        gatekeepers = res.get("top_gatekeeper_nodes", [])
        if centrality and set(centrality) != set(int(x) for x in G.nodes()):
            centrality, gatekeepers = {}, []   # stale export -> recompute
    if not centrality:
        centrality = {int(k): float(v) for k, v in
                      nx.betweenness_centrality(G, weight="weight").items()}
        gatekeepers = [n for n, _ in sorted(centrality.items(),
                                            key=lambda kv: -kv[1])[:config.TOP_N]]

    x0, y0, x1, y1 = sim.graph_extent(G)
    diag = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    comps = sorted((len(c) for c in nx.connected_components(G)), reverse=True)
    largest = max(nx.connected_components(G), key=len)
    facilities = sim.place_facilities(G, pois=sim.load_real_facilities(G))
    scenarios = sim.build_scenarios(G)

    # The browser sliders work in PERCENT of the graph diagonal so the same
    # control works for any coordinate space.  Derive those percentages from the
    # real metric sizes so Python and JavaScript cannot disagree.
    eq = next((s for s in scenarios if s["kind"] == "disc"), None)
    fl = next((s for s in scenarios if s["kind"] == "band"), None)
    metric = sim.is_gps_graph(G)
    scale = (100.0 * eq["radius"] / diag) if (eq and diag) else 22.0
    band_scale = (100.0 * fl["half_width"] / diag) if (fl and diag) else 16.0
    band_angle = fl["angle"] if fl else 35.0
    size_unit = "m" if metric else "px"

    lats = [d["lat"] for _, d in G.nodes(data=True) if "lat" in d]
    lons = [d["lon"] for _, d in G.nodes(data=True) if "lon" in d]
    centre = [round(sum(lats) / len(lats), 5), round(sum(lons) / len(lons), 5)] \
        if lats and lons else list(meta.get("center", [0.0, 0.0]))

    return {
        "meta": {**meta, "center": centre, "nodes": G.number_of_nodes(),
                 "edges": G.number_of_edges(), "metric": metric},
        "nodes": nodes,
        "edges": edges,
        "centrality": {str(k): v for k, v in centrality.items()},
        "gatekeepers": [int(g) for g in gatekeepers],
        "facilities": facilities,
        "scenarios": scenarios,
        "extent": {"x0": x0, "y0": y0, "x1": x1, "y1": y1,
                   "cx": (x0 + x1) / 2, "cy": (y0 + y1) / 2, "diag": diag},
        "scale": scale,             # blast radius as % of the graph diagonal
        "bandScale": band_scale,    # flood band half-width as % of the diagonal
        "bandAngle": band_angle,
        "sizeUnit": size_unit,
        "realFacilities": bool(facilities) and all("lat" in f for f in facilities),
        "provenance": {
            "metric": metric,
            "graph_source": str(config.GRAPH_SOURCE),
            "place": meta.get("label", ""),
            "radius_m": int(meta.get("dist", 0)),
            "radius_label": (eq or {}).get("radius_label", ""),
            "band_label": (fl or {}).get("half_width_label", ""),
        },
        "baseline": {"nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
                     "components": len(comps), "largest": len(largest),
                     "serviceable": sorted(int(n) for n in largest)},
    }


def load_basemap_manifest() -> dict:
    """Offline basemap images produced by scripts/fetch_basemap.py.

    Missing manifest is NOT an error: the console simply falls back to its
    flat dark canvas background, so a build without network access still
    produces a working file.
    """
    path = config.BASEMAP_DIR / "manifest.json"
    if not path.exists():
        print(f"[web] no basemap manifest at {path} - maps keep the flat background "
              f"(run scripts/fetch_basemap.py to add real map imagery)")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:                            # noqa: BLE001
        print(f"[web] !! basemap manifest unreadable ({exc}) - ignoring it")
        return {}


def basemap_payload(manifest: dict, key: str) -> dict:
    """Inline one area's basemap images as data: URIs plus their geo bounds."""
    entry = (manifest.get("areas") or {}).get(key) or {}
    out: dict = {}
    for style, meta in entry.items():
        img = config.BASEMAP_DIR / meta["file"]
        if not img.exists():
            continue
        raw = base64.b64encode(img.read_bytes()).decode()
        out[style] = {"img": "data:image/jpeg;base64," + raw,
                      "w": meta["w"], "h": meta["h"], "bbox": meta["bbox"],
                      "zoom": meta["zoom"]}
    return out


def region_basemap_payload(manifest: dict) -> dict | None:
    region = manifest.get("region")
    if not region:
        return None
    styles = {}
    for style, meta in (region.get("styles") or {}).items():
        img = config.BASEMAP_DIR / meta["file"]
        if not img.exists():
            continue
        raw = base64.b64encode(img.read_bytes()).decode()
        styles[style] = {"img": "data:image/jpeg;base64," + raw,
                         "w": meta["w"], "h": meta["h"], "bbox": meta["bbox"],
                         "zoom": meta["zoom"]}
    return {"bbox": region.get("bbox"), "styles": styles} if styles else None


def build(source: str) -> Path:
    areas = config.load_areas()
    built: dict = {}
    order: list = []
    basemap = load_basemap_manifest()

    if source in ("osm", "osm_cache"):
        import time
        for i, a in enumerate(areas):
            # The radius may have to grow to get a usable network, so resolve
            # through the shared helper -- that is also what fetch_basemap.py
            # uses, keeping the stitched imagery and the graph in step.
            G, meta = graph_io.resolve_area_graph(
                a, refresh=config.REFRESH_OSM_CACHE and source == "osm", tag="web")
            key = a["key"]
            if G is None:
                print(f"[web] skipping area {key!r} - no graph")
                continue
            built[key] = area_payload(G, meta)
            built[key]["basemap"] = basemap_payload(basemap, key)
            order.append(key)
            if i + 1 < len(areas):
                time.sleep(config.OSM_AREA_PAUSE_S)   # be gentle on the endpoints
    else:
        G = graph_io.load_graph(source)
        if G is None or G.number_of_nodes() == 0:
            print("[web] no graph available - run scripts/run_part_b.py first")
            sys.exit(1)
        meta = {"key": "area", "label": config.OSM_PLACE_LABEL or "Pinned area",
                "center": list(config.OSM_CENTER), "dist": int(config.OSM_DIST),
                "role": "home", "fresh": True}
        built["area"] = area_payload(G, meta)
        order.append("area")

    for k in order:
        # areas built before the manifest was read still need their basemap
        built[k].setdefault("basemap", basemap_payload(basemap, k))

    if not built:
        # Every Overpass mirror refused us.  Rather than leave the console
        # unopenable, fall back to any previously cached OSM graph so there is
        # still a working dashboard -- but label it for what it is, because a
        # cached rehearsal-area map is NOT the configured area list.
        cached = config.OUTPUTS_DIR / "osm_cache" / "graph.graphml"
        G = graph_io.load_graphml(cached) if cached.exists() else None
        if G is None or G.number_of_nodes() == 0:
            print("[web] no areas could be built and no cached graph to fall back on")
            sys.exit(1)
        meta = {"key": "cached",
                "label": "Cached graph (configured areas unavailable)",
                "center": list(config.OSM_CENTER), "dist": int(config.OSM_DIST),
                "role": "home", "fresh": True}
        print("[web] !! falling back to the cached graph - the configured areas "
              "could not be downloaded right now")
        built["cached"] = area_payload(G, meta)
        order.append("cached")
    default = next((k for k in order
                    if built[k]["meta"].get("role") == "home"), order[0])

    pipeline = []
    metric = any(built[k]["meta"].get("metric") for k in order)
    if not metric:
        # Only meaningful when the map actually came from satellite masks.
        for title, path, caption in [
            ("Satellite tile", config.TEST_REGION_DIR / "tile_001.png",
             "Input tile for the segmentation model. NOTE: data/test_region holds "
             "a procedurally generated demo tile, not DeepGlobe imagery."),
            ("Model output", config.MASKS_OUT / "mask_tile_001.png",
             "U-Net + ResNet34 turns the tile into a binary road mask (white = road)."),
            ("Stitched mask", config.MASKS_OUT / "stitched_mask.png",
             "4 adjacent tiles stitched into one region mask, then skeletonized and "
             "healed into the graph shown on the left."),
        ]:
            uri = b64_img(path)
            if uri:
                pipeline.append({"title": title, "img": uri, "caption": caption})
    else:
        # Real OSM areas: show the measured image-to-graph pipeline produced by
        # scripts/build_pipeline_assets.py on a held-out DeepGlobe patch.  This is
        # the MODEL, not the map source -- the maps are real OSM (see `metric`).
        manifest = config.OUTPUTS_DIR / "pipeline_manifest.json"
        if manifest.exists():
            for stage in json.loads(manifest.read_text(encoding="utf-8")):
                uri = b64_img(config.OUTPUTS_DIR / "pipeline" / stage["file"])
                if uri:
                    pipeline.append({"title": stage["title"], "img": uri,
                                     "caption": stage.get("caption", "")})

    base_m = json.loads(config.SEG_METRICS_JSON.read_text()) \
        if config.SEG_METRICS_JSON.exists() else {}
    att_m = json.loads(config.SEG_METRICS_ATTENTION_JSON.read_text()) \
        if config.SEG_METRICS_ATTENTION_JSON.exists() else {}

    # Locator box over every monitored area, so the console can show where they
    # all sit relative to each other.  This is a POSITION map only -- the full
    # Tamil Nadu street network is far too large for Overpass/OSMnx to fetch
    # (the probe failed at ~71x the server's max query area), so no street
    # geometry is invented for the gaps between towns.
    lats = [built[k]["meta"]["center"][0] for k in order
            if built[k]["meta"].get("metric")]
    lons = [built[k]["meta"]["center"][1] for k in order
            if built[k]["meta"].get("metric")]
    pad = 0.25
    region = {
        "label": (built[default]["meta"].get("label") or "").split(",")[-1].strip(),
        "bbox": [min(lats) - pad, min(lons) - pad, max(lats) + pad, max(lons) + pad]
                if lats else None,
        "markers": [{"key": k, "label": built[k]["meta"].get("label", k),
                     "role": built[k]["meta"].get("role", "watch"),
                     "center": built[k]["meta"]["center"]} for k in order],
        "basemap": region_basemap_payload(basemap),
    }

    # Measured research results, straight from the artefacts on disk.  These are
    # REAL held-out numbers produced by scripts/research_metrics.py -- never
    # target values or hand-written placeholders.
    research = {}
    research_path = config.OUTPUTS_DIR / "research_metrics.json"
    if research_path.exists():
        research = json.loads(research_path.read_text(encoding="utf-8"))

    payload = {
        "areas": built,
        "areaOrder": order,
        "defaultArea": default,
        "basemapAttribution": basemap.get("attribution", ""),
        "region": region,
        "pipeline": pipeline,
        "research": research,
        "llm": ea.llm_available() and
               os.environ.get("EMERGENCY_LLM", "0") in ("1", "true", "True"),
        "model": {
            "base": {"iou": base_m.get("test_iou", 0.0),
                     "dice": base_m.get("test_dice", 0.0),
                     "occ": base_m.get("occlusion_recall", 0.0)},
            "att": {"iou": att_m.get("test_iou", 0.0),
                    "dice": att_m.get("test_dice", 0.0),
                    "occ": att_m.get("occlusion_recall", 0.0)},
            "heal": "classical distance+angle healing via Union-Find (MST-style)",
        },
    }

    html = (WEB / "index.html").read_text(encoding="utf-8")
    html = html.replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":")))
    # app2.js is the 7-page console engine; app.js is the legacy single-page one.
    js = (WEB / "app2.js").read_text(encoding="utf-8")
    html = html.replace('<script src="app.js"></script>',
                        "<script>\n" + js + "\n</script>")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"[web] wrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")
    for k in order:
        m = built[k]["meta"]
        print(f"[web]   {k:14s} {m.get('label',''):32s} "
              f"{m['nodes']:5d} junctions / {m['edges']:5d} streets  "
              f"{len(built[k]['facilities'])} facilities")
    return OUT


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="teammate_export",
                    choices=["teammate_export", "graphml", "osm", "osm_cache"],
                    help="'osm' may download; 'osm_cache' reads only the graphs "
                         "already in outputs/area_cache (offline, reproducible)")
    args = ap.parse_args()
    build(args.source)


if __name__ == "__main__":
    main()
