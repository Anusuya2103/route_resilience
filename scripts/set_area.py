"""
Point the whole simulation at a REAL place, by name.

Geocodes the query with Nominatim (via OSMnx), writes the centre / radius /
label into config.py, drops the stale OSM caches, and optionally rebuilds the
dashboard straight away.

    python scripts/set_area.py "Mylapore, Chennai" --dist 1200
    python scripts/set_area.py "Koramangala, Bengaluru" --dist 1500 --build

Needs a one-off internet connection (Overpass + Nominatim).  After the first
run the graph and facilities are cached on disk and rebuilds work offline.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config

CONFIG_FILE = ROOT / "config.py"


def rewrite_config(lat: float, lon: float, dist: int, label: str) -> None:
    """Edit the OSM block in config.py in place, keeping the rest untouched."""
    text = CONFIG_FILE.read_text(encoding="utf-8")
    subs = [
        (r"^OSM_CENTER\s*=.*$",
         f"OSM_CENTER = ({lat:.6f}, {lon:.6f})   # {label}"),
        (r"^OSM_DIST\s*=.*$",
         f"OSM_DIST = {dist}                    # metres around the center point"),
        (r"^OSM_PLACE_LABEL\s*=.*$",
         f'OSM_PLACE_LABEL = "{label}"'),
        # clear_caches() already deletes the cached graph, so the next load
        # re-downloads anyway.  Leaving REFRESH on would re-hit Overpass on
        # EVERY rebuild, not just the first.
        (r"^REFRESH_OSM_CACHE\s*=.*$",
         "REFRESH_OSM_CACHE = False"),
    ]
    for pattern, replacement in subs:
        text, n = re.subn(pattern, replacement, text, count=1, flags=re.M)
        if n == 0:
            print(f"[area] WARNING: could not find {pattern!r} in config.py")
    CONFIG_FILE.write_text(text, encoding="utf-8")
    print(f"[area] config.py updated -> {label} @ ({lat:.5f}, {lon:.5f}), {dist} m")


def clear_caches() -> None:
    """Remove the cached graph and facilities so a new area is really fetched."""
    import shutil
    for path in (config.OSM_CACHE_DIR,):
        if path.exists():
            shutil.rmtree(path)
            print(f"[area] removed {path}")
    if config.OSM_FACILITY_CACHE.exists():
        config.OSM_FACILITY_CACHE.unlink()
        print(f"[area] removed {config.OSM_FACILITY_CACHE}")


def write_areas(areas: list) -> None:
    """Persist the monitored-area list to outputs/areas.json."""
    import json
    config.OSM_AREAS_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.OSM_AREAS_FILE.write_text(json.dumps(areas, indent=2), encoding="utf-8")
    print(f"[area] wrote {config.OSM_AREAS_FILE} with {len(areas)} area(s):")
    for a in areas:
        print(f"[area]   {a['key']:14s} {a['label']:32s} "
              f"{a['center'][0]:.5f},{a['center'][1]:.5f}  r={a['dist']} m  {a['role']}")


def slug(text: str) -> str:
    import re as _re
    return _re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:20] or "area"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("place", nargs="*",
                    help='one or more area names, e.g. "Kovilpatti, Thoothukudi" '
                         '"Madurai" -- the first is the operator\'s home area, the '
                         'rest are monitored from a distance')
    ap.add_argument("--lat", type=float, default=None, help="override: latitude")
    ap.add_argument("--lon", type=float, default=None, help="override: longitude")
    ap.add_argument("--dist", type=int, default=1500,
                    help="radius around each centre in metres (default 1500)")
    ap.add_argument("--region-dist", type=int, default=8000,
                    help="extra wide 'home area' view radius in metres (0 = skip)")
    ap.add_argument("--build", action="store_true",
                    help="rebuild outputs/emergency_dashboard.html afterwards")
    args = ap.parse_args()

    if not args.place and args.lat is None:
        sys.exit("[area] give at least one place name, or --lat/--lon")

    if (args.lat is None) != (args.lon is None):
        sys.exit("[area] give both --lat and --lon, or neither")

    try:
        import osmnx as ox
    except ImportError:
        sys.exit("[area] osmnx is not installed -> pip install osmnx")
    ox.settings.log_console = False

    areas = []
    if args.lat is not None:
        areas.append({"key": "area", "label": "Pinned area",
                      "center": [args.lat, args.lon],
                      "dist": args.dist, "role": "home"})
    else:
        import time
        for i, name in enumerate(args.place):
            print(f"[area] geocoding {name!r} ...")
            try:
                geo = ox.geocode(name)
            except Exception as exc:
                print(f"[area]   lookup failed ({exc}) -- skipping")
                continue
            if geo is None:
                print("[area]   no match -- skipping")
                continue
            lat, lon = float(geo[0]), float(geo[1])
            label = str(geo[2]) if len(geo) > 2 and geo[2] else name
            areas.append({"key": slug(label), "label": label,
                          "center": [round(lat, 6), round(lon, 6)],
                          "dist": args.dist,
                          "role": "home" if i == 0 else "watch"})
            print(f"[area]   matched {label} ({lat:.5f}, {lon:.5f})")
            time.sleep(1.2)          # be polite to Nominatim

    if not areas:
        sys.exit("[area] none of the places could be geocoded")

    if args.region_dist and areas:
        home = dict(areas[0])
        home.update({"key": home["key"] + "_wide",
                     "label": home["label"] + " (wide view)",
                     "dist": args.region_dist, "role": "region"})
        areas.append(home)

    write_areas(areas)
    clear_caches()
    # keep config's single-area fields pointing at the home area
    rewrite_config(areas[0]["center"][0], areas[0]["center"][1],
                   areas[0]["dist"], areas[0]["label"])

    if args.build:
        import subprocess
        cmd = [sys.executable, str(ROOT / "scripts" / "build_web_dashboard.py"),
               "--source", "osm"]
        print("[area] " + " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=str(ROOT))
    else:
        print("[area] done. Now run:  python scripts/build_web_dashboard.py --source osm")


if __name__ == "__main__":
    main()