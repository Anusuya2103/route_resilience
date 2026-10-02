"""Build a NETWORK-FREE multi-area fixture for tests/multi_area_smoke.js.

Reuses whichever cached OSM graph is already on disk and presents it as three
distinct areas, so the client-side multi-area engine can be exercised headlessly
without hitting Overpass.  Usage:  python scripts/make_smoke_fixture.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config                                            # noqa: E402
from src import graph_io, simulation as sim              # noqa: E402
from scripts import build_web_dashboard as b             # noqa: E402

# No network: fall back to evenly spread anchor facilities.
sim.load_real_facilities = lambda G: None


def main() -> int:
    cached = [config.OUTPUTS_DIR / "osm_cache" / "graph.graphml"]
    cached += sorted(config.OSM_AREA_CACHE.glob("*/graph.graphml")) \
        if config.OSM_AREA_CACHE.exists() else []
    src = next((p for p in cached if p.exists()), None)
    if src is None:
        print("[fixture] no cached graph found - run a build first")
        return 1
    G = graph_io.load_graphml(src)
    print(f"[fixture] base graph {src}: {G.number_of_nodes()} nodes")

    built, order = {}, []
    for i, (key, label) in enumerate([("a_town", "Town A (home)"),
                                      ("b_town", "Town B"),
                                      ("c_town", "Town C")]):
        meta = {"key": key, "label": label, "center": [9.1 + i * 0.1, 77.8 + i * 0.1],
                "dist": 1200, "role": "home" if i == 0 else "watch"}
        built[key] = b.area_payload(G, meta)
        order.append(key)

    payload = {
        "areas": built, "areaOrder": order, "defaultArea": "a_town",
        "region": {"label": "Test", "bbox": [8.9, 77.7, 9.5, 78.1],
                   "markers": [{"key": k, "label": built[k]["meta"]["label"],
                                "role": built[k]["meta"]["role"],
                                "center": built[k]["meta"]["center"]} for k in order]},
        # app2.js tolerates both being absent, but exercise the real shapes.
        "pipeline": [{"title": f"Stage {i + 1}", "img": "data:image/gif;base64,R0lGOD",
                      "caption": "synthetic fixture stage"} for i in range(7)],
        "research": json.loads(
            (config.OUTPUTS_DIR / "research_metrics.json").read_text(encoding="utf-8")
        ) if (config.OUTPUTS_DIR / "research_metrics.json").exists() else {},
        "llm": False,
        "model": {"base": {"iou": 0.0, "dice": 0.0, "occ": 0.0},
                  "att": {"iou": 0.0, "dice": 0.0, "occ": 0.0}, "heal": "n/a"},
    }
    out = config.OUTPUTS_DIR / "_smoke.html"
    html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    html = html.replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":")))
    # Must match the real build: the 7-page console engine is app2.js.
    html = html.replace('<script src="app.js"></script>',
                        "<script>\n" + (ROOT / "web" / "app2.js").read_text(encoding="utf-8") + "\n</script>")
    out.write_text(html, encoding="utf-8")
    print(f"[fixture] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())