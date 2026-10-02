"""
PART B -- Graph construction and topological healing.

Turns exported road masks (outputs/masks + outputs/tiles_manifest.json) into a
road-network graph:

    stitch masks -> skeletonize -> extract node/edge skeleton graph
    -> bridge disconnected endpoints (distance + direction scoring, union-find)
    -> export nodes.json / edges.json / graph.graphml
    -> connectivity report

Everything in this module works in PIXEL coordinates.  Weights of every edge and
every healing bridge are lengths in pixels (unless config.METERS_PER_PIXEL is
set, in which case they are converted).  The pixel coordinates are normalized to
relative [0,1] lat/lon when the shared contract files are written (see B9).

Key callables used by main.py / graph_io:
    generate_demo_masks()
    build_graph_pipeline()
    heal_graph()
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import networkx as nx
import numpy as np

from src import utils
import config
# ---------------------------------------------------------------------------
# Demo mask generation (offline graph-stage input)
# ---------------------------------------------------------------------------

def generate_demo_masks() -> None:
    """Build a DEMO tile set: one continuous synthetic road network stitched
    across a 2x2 grid of tiles with a couple of artificially added gaps so the
    topological-healing stage has something meaningful to repair.

    Writes:
        data/test_region/tile_001..004.png   (synthetic satellite tiles)
        outputs/masks/mask_tile_001..004.png  (0 = background, 255 = road)
        outputs/tiles_manifest.json
    """
    utils.ensure_dirs(config.MASKS_OUT, config.TEST_REGION_DIR)
    rng = np.random.default_rng(config.SEED + 7)
    s = config.IMAGE_SIZE
    big = utils.draw_synthetic_road_network(2 * s, rng=rng)

    # randomly punch 1-2 gaps in the network to create disconnected components
    ys, xs = np.nonzero(big)
    for _ in range(2):
        if len(ys):
            k = int(rng.integers(len(ys)))
            radius = int(rng.integers(4, 9))
            yy, xx = np.ogrid[:big.shape[0], :big.shape[1]]
            big[((yy - ys[k]) ** 2 + (xx - xs[k]) ** 2) < radius ** 2] = 0

    manifest = []
    for i, (r, c) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        tm = big[r * s:(r + 1) * s, c * s:(c + 1) * s]
        img = utils.make_synthetic_satellite_image(tm, rng=rng, occlude=True)
        tile_path = config.TEST_REGION_DIR / f"tile_{i + 1:03d}.png"
        mask_path = config.MASKS_OUT / f"mask_tile_{i + 1:03d}.png"
        cv2.imwrite(str(tile_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(mask_path), (tm * 255).astype(np.uint8))
        manifest.append({
            "tile_id": f"tile_{i + 1:03d}",
            "source_image": str(tile_path.relative_to(config.ROOT_DIR)).replace("\\", "/"),
            "width": s, "height": s,
            "mask_path": str(mask_path.relative_to(config.ROOT_DIR)).replace("\\", "/"),
            "bounds": None,
        })
    config.TILES_MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"[graph] generated DEMO mask set: {len(manifest)} tiles (synthetic roads + gaps)")

    # also save the stitched ground-truth for inspection
    cv2.imwrite(str(config.MASKS_OUT / "stitched_mask.png"), (big * 255).astype(np.uint8))


def load_masks_from_manifest(manifest_path: Path | None = None) -> list:
    """Load tile masks listed in tiles_manifest.json -> [(tile_id, binary_mask)]."""
    manifest_path = config.TILES_MANIFEST if manifest_path is None else manifest_path
    if not manifest_path.exists():
        print(f"[graph] tiles manifest not found ({manifest_path}); generating demo masks")
        generate_demo_masks()
    entries = json.loads(config.TILES_MANIFEST.read_text())
    masks = []
    for entry in entries:
        mp = Path(str(entry["mask_path"]))
        if not mp.is_absolute():
            mp = config.ROOT_DIR / mp
        if not mp.exists():
            print(f"[graph] missing mask {mp} -- skipping tile {entry['tile_id']}")
            continue
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if m is None:
            print(f"[graph] unreadable mask {mp} -- skipping tile {entry['tile_id']}")
            continue
        masks.append((entry["tile_id"], (m > 127).astype(np.uint8)))
    if not masks:
        raise RuntimeError("No masks available to build a graph. Run "
                           "`python main.py --stage extraction` first (or use demo mode).")
    return masks


# ---------------------------------------------------------------------------
# Stitching (B1)
# ---------------------------------------------------------------------------

def stitch_masks(masks: list) -> tuple:
    """Stitch tile masks into one large binary mask in row-major grid order.

    Returns (stitched_mask, grid_info).  Tile positions come from the manifest
    ordering (no GIS library required for the MVP).
    """
    n = len(masks)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    t_h, t_w = masks[0][1].shape
    stitched = np.zeros((rows * t_h, cols * t_w), dtype=np.uint8)
    for k, (tile_id, m) in enumerate(masks):
        r, c = divmod(k, cols)
        stitched[r * t_h:(r + 1) * t_h, c * t_w:(c + 1) * t_w] = m
    info = {"rows": rows, "cols": cols, "tile_h": t_h, "tile_w": t_w}
    return stitched, info


# ---------------------------------------------------------------------------
# Skeletonization (B2) + noise removal (B4)
# ---------------------------------------------------------------------------

def remove_small_components(bin_mask: np.ndarray, min_size: int = config.MIN_COMPONENT_SIZE):
    """Drop tiny connected components from the mask before skeletonization so
    speckle noise never becomes fake roads."""
    num, labels = cv2.connectedComponents(bin_mask, connectivity=8)
    for lbl in range(1, num):
        if int((labels == lbl).sum()) < min_size:
            bin_mask[labels == lbl] = 0
    return bin_mask


def skeletonize_mask(bin_mask: np.ndarray) -> np.ndarray:
    """One-pixel-wide centerline of the road mask (B2)."""
    from skimage.morphology import skeletonize
    return skeletonize(bin_mask > 0)


def _neighbor_counts(skel: np.ndarray) -> np.ndarray:
    """8-connected neighbor count for every skeleton pixel (B3)."""
    from scipy.ndimage import convolve
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    return convolve(skel.astype(np.uint8), kernel, mode="constant", cval=0)


# ---------------------------------------------------------------------------
# Skeleton graph extraction (B3)
# ---------------------------------------------------------------------------

def _walk_chain(skel, node_ids, start, first):
    """Follow the one-pixel-wide skeleton from `start` through `first` until the
    next node pixel.  Returns the list of pixels or None."""
    prev, cur = start, first
    path = [start, first]
    guard, max_len = 0, skel.shape[0] * skel.shape[1]
    while cur not in node_ids:
        guard += 1
        if guard > max_len:
            return None
        nxt = None
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                yy, xx = cur[0] + dy, cur[1] + dx
                if 0 <= yy < skel.shape[0] and 0 <= xx < skel.shape[1] and skel[yy, xx]:
                    if (yy, xx) != prev:
                        nxt = (yy, xx)
                        break
            if nxt is not None:
                break
        if nxt is None:
            return None
        prev, cur = cur, nxt
        path.append(cur)
    return path


def extract_graph(bin_mask: np.ndarray, min_size: int = config.MIN_COMPONENT_SIZE):
    """Build a pixel-graph from a binary road mask (B3).

    Node classification (8-connected neighbors on the skeleton):
        1 neighbor  -> endpoint
        2 neighbors -> normal path pixel (not a node)
        3+ neighbors -> intersection

    Edge weight = number of pixels along the walked chain (PIXELS).
    Returns (networkx.Graph, stats_dict).
    """
    mask = remove_small_components(bin_mask.copy(), min_size)
    skel = skeletonize_mask(mask).astype(np.uint8)
    counts = _neighbor_counts(skel)

    node_ids = {}
    next_id = 0
    for y, x in np.argwhere(skel):
        c = int(counts[y, x])
        if c != 2:                       # endpoints (1), junctions (>=3), isolated (0)
            node_ids[(int(y), int(x))] = next_id
            next_id += 1

    G = nx.Graph()
    for (y, x), nid in node_ids.items():
        G.add_node(nid, x=int(x), y=int(y))

    seen = set()
    for (ny, nx_), nid in node_ids.items():
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                yy, xx = ny + dy, nx_ + dx
                if not (0 <= yy < skel.shape[0] and 0 <= xx < skel.shape[1]):
                    continue
                if not skel[yy, xx]:
                    continue
                if (yy, xx) in node_ids:                 # direct node-node adjacency
                    end_id = node_ids[(yy, xx)]
                    key = tuple(sorted((nid, end_id)))
                    if key not in seen:
                        seen.add(key)
                        G.add_edge(nid, end_id, weight=1.0)
                    continue
                path = _walk_chain(skel, node_ids, (ny, nx_), (yy, xx))
                if path is None:
                    continue
                end_id = node_ids[path[-1]]
                key = tuple(sorted((nid, end_id)))
                if key in seen:
                    continue
                seen.add(key)
                w = float(len(path) - 1)                  # path length in pixels
                if G.has_edge(nid, end_id):
                    G[nid][end_id]["weight"] = min(G[nid][end_id]["weight"], w)
                else:
                    G.add_edge(nid, end_id, weight=w)

    stats = {"nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
             "components": nx.number_connected_components(G)}
    return G, stats
# ---------------------------------------------------------------------------
# Topological healing (B5, B6, B7) + connectivity report (B8)
# ---------------------------------------------------------------------------

class _UnionFind:
    """Disjoint-set structure used to add only MST-style healing bridges."""

    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        path = []
        while self.parent[x] != x:
            path.append(x)
            x = self.parent[x]
        for p in path:                       # path compression
            self.parent[p] = x
        return x

    def union(self, a, b) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[rb] = ra
        return True


def _positions_m_geo(G: nx.Graph) -> dict:
    """Local equirectangular projection of lat/lon nodes into a metre frame."""
    lats = [float(G.nodes[n]["lat"]) for n in G.nodes()]
    lons = [float(G.nodes[n]["lon"]) for n in G.nodes()]
    lat0 = sum(lats) / len(lats)
    lon0 = sum(lons) / len(lons)
    out = {}
    for n in G.nodes():
        dlat = math.radians(float(G.nodes[n]["lat"]) - lat0)
        dlon = math.radians(float(G.nodes[n]["lon"]) - lon0)
        out[n] = (dlon * 6371000.0 * math.cos(math.radians(lat0)), dlat * 6371000.0)
    return out


def node_positions(G: nx.Graph) -> dict:
    """Node -> (x, y) position used for bridge geometry.

    Pixel-space graphs (Part B pipeline) carry x/y directly.  Geo graphs
    (OSM, Part D) carry lat/lon, which are projected to a local metre frame so
    distances and MAX_GAP_DISTANCE are directly comparable.
    """
    has_xy = all("x" in d and "y" in d for _, d in G.nodes(data=True))
    if has_xy:
        return {n: (float(G.nodes[n]["x"]), float(G.nodes[n]["y"])) for n in G.nodes()}
    return _positions_m_geo(G)


def _bridging_cost(a, b, G, max_gap, pos: dict | None = None):
    """Cost of bridging endpoint a to endpoint b (B6) -- the CLASSICAL scorer.

    cost = DISTANCE_WEIGHT * normalized_distance + ANGLE_WEIGHT * normalized_angle
    where distances are relative to MAX_GAP_DISTANCE and the angle term prefers
    bridges that continue each road's local direction.  Lower is better.
    """
    if pos is None:
        pos = node_positions(G)
    xa, ya = pos[a]
    xb, yb = pos[b]
    dist = math.hypot(xa - xb, ya - yb)
    if dist > max_gap:
        return None

    nb_a = next(iter(G.neighbors(a)))
    nb_b = next(iter(G.neighbors(b)))
    dir_a = np.array([xa - pos[nb_a][0], ya - pos[nb_a][1]], dtype=float)
    dir_b = np.array([xb - pos[nb_b][0], yb - pos[nb_b][1]], dtype=float)
    if np.linalg.norm(dir_a) < 1e-6 or np.linalg.norm(dir_b) < 1e-6:
        return None
    dir_a /= np.linalg.norm(dir_a)
    dir_b /= np.linalg.norm(dir_b)

    vec_ab = np.array([xb - xa, yb - ya], dtype=float)
    vec_ab /= np.linalg.norm(vec_ab)
    ang_a = np.degrees(np.arccos(np.clip(np.dot(dir_a, vec_ab), -1.0, 1.0)))
    ang_b = np.degrees(np.arccos(np.clip(np.dot(dir_b, -vec_ab), -1.0, 1.0)))
    norm_angle = (ang_a + ang_b) / 360.0     # 0..1
    norm_dist = dist / max_gap               # 0..1
    return config.DISTANCE_WEIGHT * norm_dist + config.ANGLE_WEIGHT * norm_angle


def heal_graph(G: nx.Graph, scorer=None, pos: dict | None = None,
               max_gap: float | None = None):
    """Bridge disconnected components at nearby, well-aligned endpoints.

    B7 algorithm:
      1. find components
      2. generate candidate endpoint bridges (distance <= MAX_GAP_DISTANCE)
      3. sort bridges by cost
      4. add a bridge only when the two endpoints are still in different
         components (union-find), stopping when everything mergeable is merged
      5. respect MAX_GAP_DISTANCE -- never invent long/unrealistic edges

    scorer: optional callable scorer(a, b, G) -> cost (lower is better) or
      None to forbid the bridge.  Defaults to the classical distance+angle
      cost.  Part D swaps ONLY this scoring function (GNN link prediction);
      the sorting + Union-Find logic below is identical for both methods.
    pos: optional node->(x,y) positions (defaults to node_positions(G)).
    max_gap: bridge threshold (defaults to config.MAX_GAP_DISTANCE).  When
      benchmarking the two scorers we pass the SAME max_gap to both so they
      rank an identical candidate set.

    Returns (G, stats).  Edge weights stay in pixels (or metres for geo graphs).
    """
    if pos is None:
        pos = node_positions(G)
    if max_gap is None:
        max_gap = config.MAX_GAP_DISTANCE
    if scorer is None:
        def scorer(a, b, G):        # classical Part B scorer
            return _bridging_cost(a, b, G, max_gap, pos)
    comps = list(nx.connected_components(G))
    largest_before = max((len(c) for c in comps), default=0)
    comp_of = {n: i for i, c in enumerate(comps) for n in c}
    endpoints = sorted(n for n, deg in dict(G.degree()).items() if deg == 1)

    candidates = []
    for ai, a in enumerate(endpoints):
        if G.degree(a) != 1:
            continue
        for b in endpoints[ai + 1:]:
            if comp_of[a] == comp_of[b]:
                continue
            cost = scorer(a, b, G)
            if cost is not None:
                dist = math.hypot(pos[a][0] - pos[b][0], pos[a][1] - pos[b][1])
                candidates.append((cost, a, b, dist))
    candidates.sort(key=lambda t: t[0])

    uf = _UnionFind()
    bridges = []
    for cost, a, b, dist in candidates:
        if uf.find(a) != uf.find(b):         # only join separate components
            if uf.union(a, b):
                weight = dist if config.METERS_PER_PIXEL is None else \
                    dist * config.METERS_PER_PIXEL
                G.add_edge(a, b, weight=float(weight))
                bridges.append((int(a), int(b), round(float(weight), 2), round(cost, 4)))

    comps_after = list(nx.connected_components(G))
    largest_after = max((len(c) for c in comps_after), default=0)
    stats = {
        "components_before": len(comps),
        "components_after": len(comps_after),
        "largest_before": largest_before,
        "largest_after": largest_after,
        "bridges_added": len(bridges),
        "bridges": bridges,
        "connectivity_ratio": (100.0 * largest_after / largest_before) if largest_before else 0.0,
    }
    return G, stats


def _print_connectivity_report(stats: dict) -> None:
    """Part B8 terminal block."""
    print("========== GRAPH CONNECTIVITY ==========")
    print(f"Components Before: {stats['components_before']}")
    print(f"Components After: {stats['components_after']}")
    print(f"Largest Component Before: {stats['largest_before']} nodes")
    print(f"Largest Component After: {stats['largest_after']} nodes")
    print(f"Connectivity Ratio: {stats['connectivity_ratio']:.2f}%")
    print(f"Bridges Added: {stats['bridges_added']}")
    for b in stats["bridges"]:
        print(f"  bridge {b[0]} -> {b[1]}  distance={b[2]} px  cost={b[3]:.4f}")
    if stats["components_after"] > 1:
        print(f"Remaining disconnected components: {stats['components_after']} "
              f"(endpoints farther than MAX_GAP_DISTANCE={config.MAX_GAP_DISTANCE} px; "
              f"no unrealistic edges were invented)")
    print("=========================================")


def graph_to_shared_coords(G: nx.Graph, height: int, width: int) -> None:
    """Convert pixel (x, y) node attributes to relative (lat, lon) in place (B9).

    lat = 1 - y/height  (north up),  lon = x/width.
    These are RELATIVE PIXEL COORDINATES, not geographic GPS coordinates.
    """
    h, w = max(height - 1, 1), max(width - 1, 1)
    for _nid, d in G.nodes(data=True):
        x, y = float(d.get("x", 0)), float(d.get("y", 0))
        d["lat"] = float(1.0 - y / h)
        d["lon"] = float(x / w)
    G.graph["coordinate_mode"] = "relative"
    G.graph["height"] = int(height)
    G.graph["width"] = int(width)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_graph_pipeline(use_demo: bool = True):
    """Stitch -> skeletonize -> extract graph -> heal -> report.

    Returns (G, stats) with pixel coordinates still in node attrs x/y; shared
    outputs (nodes.json / edges.json / graph.graphml) are written by graph_io.
    """
    utils.set_seed(config.SEED)
    if not config.TILES_MANIFEST.exists() or \
            not list(config.MASKS_OUT.glob("mask_tile_*.png")):
        if use_demo:
            print("[graph] no segmentation masks found -- generating a DEMO tile set "
                  "(synthetic roads + gaps)")
            generate_demo_masks()
        else:
            print("[graph] no exported masks found. Run first:")
            print("    python main.py --stage extraction")
            sys.exit(1)

    masks = load_masks_from_manifest()
    stitched, grid = stitch_masks(masks)
    cv2.imwrite(str(config.MASKS_OUT / "stitched_mask.png"),
                (stitched * 255).astype(np.uint8))
    print(f"[graph] stitched {len(masks)} tiles -> {stitched.shape[0]}x{stitched.shape[1]} "
          f"mask (stitched_mask.png)")

    G, stats = extract_graph(stitched)
    print(f"[graph] skeleton graph: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges, {stats['components']} components")
    # healing scorer: classical (default) or the learned GNN link predictor
    scorer = None
    if config.HEALING_METHOD == "gnn":
        from src import gnn_healing
        try:
            model = gnn_healing.load_link_predictor()
            scorer = gnn_healing.GNNBridgingScorer(model, config.MAX_GAP_DISTANCE)
            print(f"[graph] healing scorer: GNN link prediction ({config.HEALING_METHOD})")
        except Exception as exc:
            print(f"[graph] GNN scorer unavailable ({exc}); falling back to CLASSICAL healing")
            scorer = None
    else:
        print(f"[graph] healing scorer: classical distance+angle ({config.HEALING_METHOD})")
    G, heal_stats = heal_graph(G, scorer=scorer)
    stats.update(heal_stats)
    graph_to_shared_coords(G, stitched.shape[0], stitched.shape[1])
    _print_connectivity_report(heal_stats)
    return G, stats