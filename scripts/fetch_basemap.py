"""
Pre-render real OpenStreetMap raster tiles into offline basemap images.

The console in web/app2.js is a plain 2-D canvas renderer, so it cannot fetch
tiles at runtime without breaking its "open the file and it works, with no
server and no internet" promise.  Instead this script downloads the raster
tiles ONCE at build time, stitches them into one JPEG per (area, style) and
writes the affine transform that lines the image up with the graph's own
coordinate space.  scripts/build_web_dashboard.py then embeds those JPEGs as
data: URIs.

    python scripts/fetch_basemap.py --list          # show what would be fetched
    python scripts/fetch_basemap.py                 # fetch/stitch everything

Tiles are cached on disk under outputs/basemap/tiles/, so a rebuild costs
nothing.  Raw tiles come from CARTO's basemaps (OpenStreetMap data); both
styles require attribution, which the dashboard footer carries.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402

BASE_DIR = config.OUTPUTS_DIR / "basemap"
TILE_DIR = BASE_DIR / "tiles"

# Web-Mercator constants (EPSG:3857).
TILE = 256
ORIGIN_SHIFT = 20037508.342789244      # 20037508.342789244 * 2 = earth circumference
M_PER_DEG_LAT = 111132.92 - 559.82 * math.cos(2 * math.radians(9.5)) \
    + 1.175 * math.cos(4 * math.radians(9.5)) - 0.0023 * math.cos(6 * math.radians(9.5))
_M_PER_DEG_LON_EQUATOR = 111412.84 * math.cos(math.radians(30)) \
    - 93.5 * math.cos(3 * math.radians(30)) + 0.118 * math.cos(5 * math.radians(30))

# Two styles so the console can flip between a dark night map (matches the
# console's own dark chrome) and a classic light map.
#
# CARTO's hosted basemaps were the obvious first choice but now answer every
# tile with an "API KEY REQUIRED" watermark, so these two are used instead:
#   dark  -> Esri "Dark Gray Canvas", which is a BASE layer: the road geometry
#            arrives in `..._Base` and the place labels in `..._Reference`, so
#            the dark style stitches both (base first, labels over it).
#   light -> the standard OpenStreetMap tile layer.
# Both require attribution, which the dashboard footer carries.
STYLES = {
    "dark": {
        "layers": [
            "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/"
            "World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
            "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/"
            "World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}",
        ],
        "bg": (26, 30, 38),
    },
    "light": {
        "layers": [
            "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/"
            "World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
            "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/"
            "World_Light_Gray_Reference/MapServer/tile/{z}/{y}/{x}",
        ],
        "bg": (233, 231, 225),
    },
}

# tile.openstreetmap.org was the obvious light source and it worked -- until it
# did not.  Its usage policy forbids bulk downloading, so after a few hundred
# tiles it started answering 403 with a "Blocked / osm.wiki/Blocked" watermark
# rendered INTO the tile.  Those look like real maps at a glance and would have
# shipped inside the console; `check_stitched` below exists to catch exactly
# that, and the light style now uses the same Esri provider as the dark one.
BLOCKED_WHITE_FRACTION = 0.45

ATTRIBUTION = ("Roads &amp; places: \u00a9 OpenStreetMap contributors. "
               "Basemap: Esri, HERE, Garmin, \u00a9 OpenStreetMap contributors")

MAX_PX = 1000          # stitched image is downscaled to at most this width
TARGET_PX = 1000       # aim for roughly this much detail across the bbox
MIN_Z, MAX_Z = 10, 17
JPEG_QUALITY = 80
# The console fits the graph to the canvas and then draws the basemap at the
# graph's exact scale (so the overlay lands on the real streets).  A tall,
# narrow graph inside a wide canvas therefore needs imagery wider than the
# graph itself, or the flat background shows down the sides.  Widen every bbox
# to at least this canvas aspect ratio before stitching.
VIEWPORT_ASPECT = 2.0
USER_AGENT = ("road-resilience-dashboard/2.0 (academic road-network resilience "
              "research; basemap tiles cached locally for offline use)")
PAUSE_S = 0.25         # be polite to the public tile servers


# --------------------------------------------------------------------------
# Web Mercator
# --------------------------------------------------------------------------
def lonlat_to_world(lon: float, lat: float, z: int) -> tuple[float, float]:
    """(lon, lat) -> global pixel coordinates at zoom z."""
    lat = max(min(lat, 85.05112878), -85.05112878)
    n = TILE * (2 ** z)
    x = (lon + 180.0) / 360.0 * n
    s = math.sin(math.radians(lat))
    y = (0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)) * n
    return x, y


def pick_zoom(lat: float, span_lon: float, span_lat: float, target_px: int = TARGET_PX,
              min_z: int = MIN_Z, max_z: int = MAX_Z) -> int:
    """Highest zoom whose stitched mosaic stays near `target_px` wide/tall.

    `span_lon`/`span_lat` are in DEGREES; they are converted to metres here
    because `res` is metres-per-pixel.  Mixing the two silently produced a
    zoom ~4 levels too high (and then a multi-gigabyte canvas).
    """
    widest_m = max(span_lon * _M_PER_DEG_LON_EQUATOR * math.cos(math.radians(lat)),
                   span_lat * M_PER_DEG_LAT)
    if widest_m <= 0:
        return max_z
    lat_ref = max(min(lat, 60.0), -60.0)
    res = 156543.03392804097 * math.cos(math.radians(lat_ref))   # m/px at z0
    z = math.log2(res * target_px / widest_m)
    return max(min_z, min(max_z, int(round(z))))


def download(url: str, tries: int = 4) -> bytes:
    last: Exception | None = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read()
        except Exception as exc:            # noqa: BLE001 - mirrors are flaky
            last = exc
            time.sleep(0.6 * (i + 1))
    raise RuntimeError(f"could not fetch {url}: {last}")


def tile_image(style: str, z: int, x: int, y: int):
    """One 256px tile for `style`, from disk cache when possible.

    A style may be several layers (the Esri dark basemap ships geometry and
    labels separately); they are composited base-first, labels-last, so the
    result is one ordinary image.
    """
    from io import BytesIO
    from PIL import Image

    cfg = STYLES[style]
    path = TILE_DIR / style / str(z) / str(x) / f"{y}.jpg"
    if path.exists():
        return Image.open(path).convert("RGB")

    out = Image.new("RGB", (TILE, TILE), cfg["bg"])
    for i, template in enumerate(cfg["layers"]):
        raw = download(template.format(z=z, x=x, y=y), tries=4)
        layer = Image.open(BytesIO(raw))
        # The reference/label layer is transparent outside the glyphs, so it
        # must be alpha-composited; only the last layer may be opaque.
        if i < len(cfg["layers"]) - 1 or layer.mode in ("RGBA", "LA", "P"):
            layer = layer.convert("RGBA")
            if layer.getchannel("A").getextrema() == (255, 255):
                out.paste(layer.convert("RGB"), (0, 0))
            else:
                out = Image.alpha_composite(
                    out.convert("RGBA"), layer).convert("RGB")
        else:
            out.paste(layer.convert("RGB"), (0, 0))

    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path, "JPEG", quality=88)
    return out


# --------------------------------------------------------------------------
# Stitching
# --------------------------------------------------------------------------
def mosaic(style: str, bbox: tuple[float, float, float, float],
            target_px: int = TARGET_PX, min_z: int = MIN_Z, max_z: int = MAX_Z):
    """Stitch every tile covering `bbox` into one image.

    Returns (PIL image, meta) where meta carries the geographic bounds and the
    pixel size, which is all the browser needs to align the image with the
    graph (see web/app2.js drawBasemap).  `min_z`/`target_px` are lowered for
    the region-wide locator, whose bbox spans most of Tamil Nadu.
    """
    lat0, lon0, lat1, lon1 = bbox          # south, west, north, east
    lat_c = (lat0 + lat1) / 2.0
    z = pick_zoom(lat_c, lon1 - lon0, lat1 - lat0, target_px, min_z, max_z)

    wx0, wy0 = lonlat_to_world(lon0, lat0, z)
    wx1, wy1 = lonlat_to_world(lon1, lat1, z)
    # lat0 is SOUTH, and in Mercator Y increases southward, so the south edge
    # has the LARGER y.  Order both axes before deriving tile indices, or the
    # row count comes out negative.
    x_min, x_max = int(min(wx0, wx1) // TILE), int(max(wx0, wx1) // TILE)
    y_min, y_max = int(min(wy0, wy1) // TILE), int(max(wy0, wy1) // TILE)

    # A very large bbox (the 8 km "wide" area) would otherwise ask for
    # hundreds of tiles; MAX_Z already bounds this, so this is a safety net.
    if (x_max - x_min + 1) * (y_max - y_min + 1) > 400:
        raise RuntimeError(f"mosaic too large at z{z} for {bbox}")

    cols = x_max - x_min + 1
    rows = y_max - y_min + 1
    from PIL import Image
    canvas = Image.new("RGB", (cols * TILE, rows * TILE), STYLES[style]["bg"])
    for row in range(rows):
        for col in range(cols):
            canvas.paste(tile_image(style, z, x_min + col, y_min + row),
                         (col * TILE, row * TILE))

    # Crop to the bbox itself (tiles are square, the bbox is not).
    # NOTE: lat0 is the SOUTH edge, which is the LARGER Mercator Y, so the
    # north edge (lat1) is the smaller one -- order the corners explicitly
    # rather than assuming lat0 -> top.
    left = int(round(min(wx0, wx1) - x_min * TILE))
    right = int(round(max(wx0, wx1) - x_min * TILE))
    top = int(round(min(wy0, wy1) - y_min * TILE))
    bottom = int(round(max(wy0, wy1) - y_min * TILE))
    canvas = canvas.crop((left, top, right, bottom))

    if canvas.width > MAX_PX:
        scale = MAX_PX / canvas.width
        canvas = canvas.resize((MAX_PX, max(1, int(round(canvas.height * scale)))),
                               Image.LANCZOS)

    meta = {"zoom": z, "w": canvas.width, "h": canvas.height,
            "bbox": [lat0, lon0, lat1, lon1]}
    return canvas, meta


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def area_bbox(G, pad_frac: float = 0.08, aspect: float = VIEWPORT_ASPECT):
    """Geographic bbox of every node in `G`, padded and widened to fill a canvas.

    Two jobs.  The pad keeps roads off the very edge of the image (a road
    clipped at the border looks like a bug).  The widening is what makes the
    console look right: it fits a graph that is usually TALLER than it is wide
    into a canvas that is WIDER than it is tall, and draws the basemap at the
    graph's exact scale so the roads line up.  Without extra width the imagery
    would run out part-way across the canvas and leave flat background showing
    on both sides -- so grow whichever axis is short, symmetrically about the
    centre, never cropping anything the graph covers.
    """
    lats = [d["lat"] for _, d in G.nodes(data=True) if "lat" in d and "lon" in d]
    lons = [d["lon"] for _, d in G.nodes(data=True) if "lat" in d and "lon" in d]
    if not lats or not lons:
        return None
    lat0, lat1 = min(lats), max(lats)
    lon0, lon1 = min(lons), max(lons)
    lat_c, lon_c = (lat0 + lat1) / 2, (lon0 + lon1) / 2
    span_lat = ((lat1 - lat0) or 0.004) * (1 + 2 * pad_frac) * M_PER_DEG_LAT
    span_lon = ((lon1 - lon0) or 0.004) * (1 + 2 * pad_frac) * \
        _M_PER_DEG_LON_EQUATOR * math.cos(math.radians(lat_c))
    # grow each axis until the box is at least `aspect` wide, then re-pad
    span_lon = max(span_lon, span_lat * aspect)
    span_lat = max(span_lat, span_lon / aspect)
    span_lon *= 1.02
    span_lat *= 1.02
    return (lat_c - span_lat / 2 / M_PER_DEG_LAT,
            lon_c - span_lon / 2 / (_M_PER_DEG_LON_EQUATOR
                                    * math.cos(math.radians(lat_c))),
            lat_c + span_lat / 2 / M_PER_DEG_LAT,
            lon_c + span_lon / 2 / (_M_PER_DEG_LON_EQUATOR
                                    * math.cos(math.radians(lat_c))))


def region_bbox(areas, pad: float = 0.25):
    lats = [a["center"][0] for a in areas]
    lons = [a["center"][1] for a in areas]
    if not lats:
        return None
    return (min(lats) - pad, min(lons) - pad, max(lats) + pad, max(lons) + pad)


def check_stitched(name: str, style: str, img) -> bool:
    """Reject a stitched image that is mostly blank paper.

    A blocked tile from a public tile server is not an error page -- it is a
    real map tile with "Blocked / osm.wiki/Blocked" set into it, so every
    other check passes and it silently reaches the dashboard.  Those tiles are
    overwhelmingly near-white; real cartography of a populated Indian town is
    not.  Cheap check, and it is the difference between a working light basemap
    and a console covered in usage-policy warnings.
    """
    small = img.convert("L").resize((160, 160))
    px = list(small.getdata())
    white = sum(1 for p in px if p >= 245) / len(px)
    if white > BLOCKED_WHITE_FRACTION:
        print(f"[basemap]   {name}/{style}: REJECTED - {white:.0%} of the image is "
              f"blank white, which is what a usage-policy 'Blocked' tile looks "
              f"like. Not writing it.")
        return False
    return True


def build(styles: list[str], only: str | None, dry: bool) -> Path:
    from src import graph_io

    areas = config.load_areas()
    entries: dict[str, dict] = {}

    def emit(name: str, bbox, target_px=TARGET_PX, min_z=MIN_Z, max_z=MAX_Z):
        """Stitch + save every requested style for one bbox."""
        if dry:
            return None
        got = {}
        for style in styles:
            try:
                img, meta = mosaic(style, bbox, target_px, min_z, max_z)
            except Exception as exc:                     # noqa: BLE001
                print(f"[basemap]   {style}: FAILED ({exc})")
                continue
            if not check_stitched(name, style, img):
                continue
            out = BASE_DIR / f"{name}_{style}.jpg"
            out.parent.mkdir(parents=True, exist_ok=True)
            img.save(out, "JPEG", quality=JPEG_QUALITY, optimize=True)
            meta["file"] = out.name
            meta["bytes"] = out.stat().st_size
            got[style] = meta
            print(f"[basemap]   {style:5s} z{meta['zoom']} "
                  f"{meta['w']}x{meta['h']}  {meta['bytes'] / 1024:.0f} KB")
        return got or None

    for a in areas:
        key = a["key"]
        if only and key != only:
            continue
        # SAME resolver the dashboard build uses, so an area whose radius had to
        # grow (Thoothukudi 6 km, Virudhunagar 4.8 km) gets imagery stitched for
        # the graph that actually ships, not for the original small box.
        G, meta = graph_io.resolve_area_graph(a, refresh=False, tag="basemap")
        if G is None or G.number_of_nodes() == 0:
            print(f"[basemap] skip {key}: no graph")
            continue
        bbox = area_bbox(G)
        if bbox is None:
            print(f"[basemap] skip {key}: no GPS coordinates")
            continue
        print(f"[basemap] {key:14s} r={meta['dist']:>5d} m  bbox "
              f"{bbox[0]:.4f},{bbox[1]:.4f} .. {bbox[2]:.4f},{bbox[3]:.4f}")
        got = emit(key, bbox)
        if got:
            entries[key] = got

    # The page-01 Regional Locator spans every town at once, so it needs one
    # much wider (and therefore much lower-zoom) mosaic.
    region = None
    if not only:
        rb = region_bbox(areas)
        if rb:
            print(f"[basemap] {'region':14s} bbox {rb[0]:.4f},{rb[1]:.4f} .. "
                  f"{rb[2]:.4f},{rb[3]:.4f}")
            got = emit("region", rb, target_px=900, min_z=6, max_z=12)
            if got:
                region = {"bbox": list(rb), "styles": got}

    manifest = BASE_DIR / "manifest.json"
    if dry:
        print("[basemap] dry run - nothing written")
        return manifest

    data = {"attribution": ATTRIBUTION, "styles": styles,
            "areas": entries, "region": region}
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(data, indent=1), encoding="utf-8")
    total = sum(s["bytes"] for e in entries.values() for s in e.values())
    if region:
        total += sum(s["bytes"] for s in region["styles"].values())
    print(f"[basemap] wrote {manifest}  ({len(entries)} areas, "
          f"{total / 1024:.0f} KB of images)")
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--style", default="dark,light",
                    help="comma-separated subset of: " + ",".join(STYLES))
    ap.add_argument("--area", default=None, help="only this area key")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be fetched, write nothing")
    args = ap.parse_args()
    styles = [s.strip() for s in args.style.split(",") if s.strip()]
    for s in styles:
        if s not in STYLES:
            ap.error(f"unknown style {s!r}; choose from {', '.join(STYLES)}")
    build(styles, args.area, args.dry_run)


if __name__ == "__main__":
    main()
