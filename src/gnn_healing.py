"""
PART D -- Learned Graph Healing via GNN Link Prediction (Novelty 2).

Problem framing:
    Gap-healing == edge/link prediction.  Every pair of disconnected road
    endpoints is a *candidate* edge; the task is to predict whether it should
    exist.  This mirrors the classical heuristic of Part B (which scores
    candidates by distance + angular alignment) but replaces the hand-tuned
    formula with a learned, graph-aware probability.

Data construction (graph-level analogue of Part A's pixel occlusion):
    * ground-truth OSM road graphs (reused from src/graph_io's OSM cache) +
      deterministic synthetic grids for variety;
    * random edges are removed to simulate occlusion gaps;
    * POSITIVES = the removed edges (they must be re-added),
      NEGATIVES  = random non-edges (distant / misaligned endpoints).

Node features (per spec):
    (x,y) position, node degree, distance to nearest endpoint, and the local
    direction vector of the node's existing edge (same angular-alignment idea
    as Part B, but the GNN learns the right weighting instead of a tuned
    closed-form weight).

Integration (deliberately minimal risk):
    * The final graph is STILL built by the same Union-Find procedure as
      Part B (graph_builder.heal_graph).
    * ONLY the scoring function is swapped: heal_graph(G, scorer=GNNScorer)
      replaces heal_graph(G, scorer=_bridging_cost).  Nothing else changes.

Conclusion policy: report honestly. If the classical method is as good and
much faster, we keep it as the default and validate the GNN as an alternative.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config


# ---------------------------------------------------------------------------
# Geometry: positions and candidates in a common metric space
# ---------------------------------------------------------------------------

def positions_m(G: nx.Graph):
    """Map node -> (x_m, y_m) position in a common METRE-like space.

    * OSM graphs carry real lat/lon -> equirectangular projection in metres
      (valid for small neighborhoods).
    * Pixel/synthetic graphs carry x/y -> treated as metres (or scaled by a
      nominal cell size) so distances are comparable.
    """
    pos = {}
    has_lat = all("lat" in d for _, d in G.nodes(data=True))
    if has_lat:
        lats = [float(G.nodes[n]["lat"]) for n in G.nodes()]
        lons = [float(G.nodes[n]["lon"]) for n in G.nodes()]
        lat0 = np.mean(lats)
        lon0 = np.mean(lons)
        for n in G.nodes():
            dlat = math.radians(float(G.nodes[n]["lat"]) - lat0)
            dlon = math.radians(float(G.nodes[n]["lon"]) - lon0)
            pos[n] = (dlon * 6371000.0 * math.cos(math.radians(lat0)),
                      dlat * 6371000.0)
    else:
        # pixel / synthetic graphs: scale cells to metres when the graph declares it
        scale = 1.0
        try:
            scale = float(G.graph.get("cell_meters", 1.0))
        except AttributeError:
            pass
        for n in G.nodes():
            d = G.nodes[n]
            pos[n] = (float(d.get("x", d.get("lon", 0.0))) * scale,
                      float(d.get("y", d.get("lat", 0.0))) * scale)
    return pos


def graph_max_gap(G: nx.Graph) -> float:
    """Per-graph bridge threshold for Part D.

    A fixed MAX_GAP_DISTANCE (tuned for the 512px Part B tiles) is too small for
    real OSM geometry, where a single removed street edge leaves endpoints a
    typical edge-length apart.  We therefore use
        max_gap = max(MAX_GAP_DISTANCE, 1.5 x median edge length)
    so the candidate set actually contains the gaps we are trying to heal.
    """
    pos = positions_m(G)
    lens = [math.hypot(pos[v][0] - pos[u][0], pos[v][1] - pos[u][1])
            for u, v in G.edges()]
    median = float(np.median(lens)) if lens else 0.0
    return float(max(config.MAX_GAP_DISTANCE, 1.5 * median))


def normalize_positions(pos: dict):
    """Return (pos_norm, x_scale, y_scale): positions scaled to the unit box."""
    xs = np.array([v[0] for v in pos.values()])
    ys = np.array([v[1] for v in pos.values()])
    x_scale = float(xs.max() - xs.min()) if xs.max() > xs.min() else 1.0
    y_scale = float(ys.max() - ys.min()) if ys.max() > ys.min() else 1.0
    x0, y0 = float(xs.min()), float(ys.min())
    norm = {n: ((x - x0) / x_scale, (y - y0) / y_scale) for n, (x, y) in pos.items()}
    return norm, x_scale, y_scale


def _haversine_scale(G: nx.Graph) -> float:
    """Metres per unit of networkx distance for pixel-style graphs (nominal)."""
    return float(getattr(G.graph, "get", lambda k, d: d)("cell_meters", 25.0))


def endpoints(G: nx.Graph) -> set:
    return {n for n, deg in dict(G.degree()).items() if deg == 1}


def _direction(pos: dict, node: int) -> tuple:
    """Unit vector along the node's single incident edge, from metric positions.

    This reuses the angular-alignment idea of Part B's classical bridge scoring
    (roads should continue straight), but lets the GNN learn the right weight
    instead of using a closed-form formula.
    """
    # NOTE: direction is normalised later; here we return the raw unit vector.
    return (0.0, 0.0)   # placeholder -- replaced by the pos-aware version below


def direction_at(pos: dict, G: nx.Graph, node: int) -> tuple:
    """Unit vector along the node's incident edge, computed from `pos` (metres)."""
    try:
        nbr = next(G.neighbors(node))
    except StopIteration:
        return (0.0, 0.0)
    ax, ay = pos[node]
    bx, by = pos[nbr]
    d = math.hypot(bx - ax, by - ay)
    if d < 1e-12:
        return (0.0, 0.0)
    return ((bx - ax) / d, (by - ay) / d)


def node_features_matrix(G: nx.Graph, pos: dict):
    """Per-node feature tensor (N x 6).

    Features:
        0..1  normalized (x, y) position
        2     degree
        3     distance to nearest endpoint (normalized)
        4..5  local edge direction unit vector (dirx, diry)
    """
    pos_n, xs_, ys_ = normalize_positions(pos)
    eps = endpoints(G)
    # nearest-endpoint distance for every node
    near_ep = {}
    for n in G.nodes():
        if n in eps:
            near_ep[n] = 0.0
            continue
        px, py = pos[n]
        best = min((math.hypot(px - pos[e][0], py - pos[e][1]) for e in eps), default=0.0)
        near_ep[n] = best
    # diagonal of the position box, as a scale reference
    xs = [v[0] for v in pos.values()]; ys = [v[1] for v in pos.values()]
    diag = math.hypot(max(xs) - min(xs), max(ys) - min(ys)) or 1.0
    deg_max = max((d for _, d in G.degree()), default=1) or 1
    feats = {}
    for n in G.nodes():
        nx_, ny_ = pos_n[n]
        deg = float(G.degree(n)) / deg_max                     # 0..1
        epd = float(near_ep[n]) / max(diag, 1e-9)
        dx, dy = direction_at(pos, G, n)
        feats[n] = [nx_, ny_, deg, epd, dx, dy]
    return pos_n, feats, diag


# ---------------------------------------------------------------------------
# Candidate generation + pair features
# ---------------------------------------------------------------------------

def candidate_pairs(G: nx.Graph, pos: dict, max_gap: float):
    """Endpoint pairs (a,b) in DIFFERENT components of G within max_gap.

    Returns list of (a, b, dist).  This is exactly the candidate set the
    classical Part B healing considers, so both scorers rank the SAME set.
    """
    comp = {}
    for i, c in enumerate(nx.connected_components(G)):
        for n in c:
            comp[n] = i
    eps = sorted(endpoints(G))
    out = []
    for i, a in enumerate(eps):
        ax, ay = pos[a]
        for b in eps[i + 1:]:
            if comp[a] == comp[b]:
                continue
            bx, by = pos[b]
            d = math.hypot(bx - ax, by - ay)
            if d <= max_gap:
                out.append((a, b, d))
    return out


def pair_features(a: int, b: int, G: nx.Graph, pos: dict,
                  node_feats: dict, diag: float, max_gap: float):
    """Feature vector for a candidate edge (a,b).

    Layout (15 dims):
        node_feats[a]  (6): x, y, degree, dist-to-nearest-endpoint, dirx, diry
        node_feats[b]  (6)
        dist / max_gap (1)
        cos(angle between a's direction and the a->b line) (1)
        cos(angle between b's direction and the b->a line) (1)
    """
    ax, ay = pos[a]
    bx, by = pos[b]
    d = math.hypot(bx - ax, by - ay)
    ux, uy = (bx - ax) / d, (by - ay) / d if d > 0 else (0.0, 0.0)
    if d == 0:
        ux, uy = 0.0, 0.0
    dax, day = direction_at(pos, G, a)
    dbx, dby = direction_at(pos, G, b)
    cos_a = dax * ux + day * uy
    cos_b = dbx * (-ux) + dby * (-uy)
    dist_norm = d / max_gap if max_gap > 0 else 0.0
    return np.array(node_feats[a] + node_feats[b] + [dist_norm, cos_a, cos_b],
                    dtype=np.float32)


# ---------------------------------------------------------------------------
# Ground-truth base graphs (OSM cache + synthetic grids)
# ---------------------------------------------------------------------------

def _synthetic_grid(rows: int, cols: int, cell_meters: float = 25.0) -> nx.Graph:
    G = nx.Graph()
    for r in range(rows):
        for c in range(cols):
            G.add_node(r * cols + c, x=float(c), y=float(r))
    for r in range(rows):
        for c in range(cols):
            nid = r * cols + c
            if c + 1 < cols:
                G.add_edge(nid, nid + 1, weight=1.0)
            if r + 1 < rows:
                G.add_edge(nid, nid + cols, weight=1.0)
    G.graph["cell_meters"] = cell_meters
    return G


def build_base_graphs() -> list:
    """Ground-truth road graphs used to synthesise occlusion gaps.

    Priority: the cached OSM neighborhood (reused from the Part C / demo
    pipeline) when present, plus deterministic synthetic grids so Part D
    works fully offline.  Returns a list of nx.Graph.
    """
    from src import graph_io
    graphs = []
    osm_cached = config.OSM_CACHE_DIR / "graph.graphml"
    if osm_cached.exists():
        try:
            G = graph_io.load_graphml(osm_cached)
            if G is not None and G.number_of_nodes() > 0:
                print(f"[gnn] base graph: OSM ({G.number_of_nodes()} nodes)")
                graphs.append(G)
        except Exception as exc:
            print(f"[gnn] could not load OSM base graph ({exc})")
    graphs.append(_synthetic_grid(7, 9))
    graphs.append(_synthetic_grid(11, 7))
    print(f"[gnn] total base graphs: {len(graphs)}")
    return graphs


# ---------------------------------------------------------------------------
# Corruption -> positive / negative pair dataset
# ---------------------------------------------------------------------------

def corrupt_graph(G: nx.Graph, rng, edge_drop: float):
    """Remove random edges to simulate occlusion gaps (graph-level analogue).

    Returns (G_corrupt, removed_edges) where removed_edges is a set of
    frozenset({u,v}).
    """
    H = G.copy()
    edges = list(H.edges())
    k = max(1, int(len(edges) * edge_drop))
    if k > len(edges) - 1:
        k = max(1, len(edges) - 1)
    removed = rng.choice(len(edges), size=k, replace=False)
    removed_edges = set()
    for r in removed:
        u, v = edges[r]
        removed_edges.add(frozenset((u, v)))
        H.remove_edge(u, v)
    return H, removed_edges


def build_pair_samples(G_true: nx.Graph, H: nx.Graph, removed_edges: set,
                       rng, max_gap: float, neg_per_pos: int):
    """Training samples for one corrupted graph.

    Positives = removed edges (should re-connect).  Negatives = endpoint pairs
    that were NOT real edges (random, incl. distant/misaligned ones).
    Returns (X, y): X (M x 15) float32 features, y (M,) float32 labels.
    """
    pos = positions_m(H)
    _, node_feats, diag = node_features_matrix(H, pos)
    X, y = [], []
    # positives
    for e in removed_edges:
        a, b = tuple(e)
        if a not in H or b not in H:
            continue
        X.append(pair_features(a, b, H, pos, node_feats, diag, max_gap))
        y.append(1.0)
    # negatives: random non-edge endpoint pairs, within a slightly wider radius
    cands = candidate_pairs(H, pos, max_gap * 1.5)
    n_pos = int(sum(y))
    n_neg = n_pos * neg_per_pos
    if n_neg > 0 and cands:
        neg_pool = [(a, b) for a, b, _ in cands
                    if frozenset((a, b)) not in removed_edges
                    and not H.has_edge(a, b)]
        if neg_pool:
            n_neg = min(n_neg, len(neg_pool))
            idx = rng.choice(len(neg_pool), size=n_neg, replace=False)
            for j in idx:
                a, b = neg_pool[j]
                X.append(pair_features(a, b, H, pos, node_feats, diag, max_gap))
                y.append(0.0)
    if not X:
        return np.zeros((0, 15), np.float32), np.zeros((0,), np.float32)
    return np.stack(X).astype(np.float32), np.asarray(y, dtype=np.float32)


# ---------------------------------------------------------------------------
# PyTorch Geometric link-prediction model
# ---------------------------------------------------------------------------

def _pyg():
    try:
        import torch_geometric
        from torch_geometric.nn import GCNConv
        return torch_geometric, GCNConv
    except ImportError as exc:
        raise ImportError("Part D needs torch-geometric: pip install torch-geometric") from exc


class GCNLinkPredictor(nn.Module):
    """2-layer GCN encoder + MLP head that scores a candidate edge (a,b).

    A GCN encodes each node together with its neighbourhood (so the score is
    graph-aware, unlike the classical per-pair formula), then the head scores
    the concatenated embeddings of the two endpoints together with their
    explicit pair features (distance + two alignment cosines).
    """

    def __init__(self, node_dim: int = 6, pair_dim: int = 15, hidden: int = 64,
                 layers: int = 2):
        super().__init__()
        _pyg_tg, GCNConv = _pyg()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(node_dim, hidden))
        for _ in range(max(0, layers - 1)):
            self.convs.append(GCNConv(hidden, hidden))
        self.head = nn.Sequential(
            nn.Linear(2 * hidden + pair_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, max(8, hidden // 2)),
            nn.ReLU(),
            nn.Linear(max(8, hidden // 2), 1),
        )

    def node_embeddings(self, x, edge_index):
        h = x
        for conv in self.convs:
            h = F.relu(conv(h, edge_index))
        return h

    def forward(self, x, edge_index, pair_x, pair_ab):
        # x (N,node_dim); pair_x (M,pair_dim); pair_ab (M,2) local node indices
        z = self.node_embeddings(x, edge_index)
        inp = torch.cat([z[pair_ab[:, 0]], z[pair_ab[:, 1]], pair_x], dim=1)
        return self.head(inp).squeeze(-1)


def _corrupt_to_data(G_true: nx.Graph, H: nx.Graph, removed: set, rng,
                     max_gap: float, neg_per_pos: int):
    """Build a PyG Data sample for one corrupted graph (or None if degenerate)."""
    from torch_geometric.data import Data
    pos = positions_m(H)
    nodes = list(H.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    _, feats, _ = node_features_matrix(H, pos)
    x = np.array([feats[n] for n in nodes], dtype=np.float32)
    ei = []
    for u, v in H.edges():
        ei.append([idx[u], idx[v]]); ei.append([idx[v], idx[u]])
    edge_index = np.array(ei, dtype=np.int64).T if ei else np.zeros((2, 0), np.int64)
    cands = candidate_pairs(H, pos, max_gap)
    pos_pairs = [(a, b) for a, b, _ in cands if frozenset((a, b)) in removed]
    neg_pairs = [(a, b) for a, b, _ in cands
                 if frozenset((a, b)) not in removed and not H.has_edge(a, b)]
    n_pos = len(pos_pairs)
    if n_pos == 0 or not neg_pairs:
        return None
    n_neg = min(n_pos * neg_per_pos, len(neg_pairs))
    sel = rng.choice(len(neg_pairs), size=n_neg, replace=False)
    neg_pairs = [neg_pairs[j] for j in sel]
    all_pairs = pos_pairs + neg_pairs
    labels = [1.0] * n_pos + [0.0] * n_neg
    PX, PAB = [], []
    for a, b in all_pairs:
        PX.append(pair_features(a, b, H, pos, feats, 1.0, max_gap))
        PAB.append([idx[a], idx[b]])
    return Data(x=torch.from_numpy(x),
                edge_index=torch.from_numpy(edge_index),
                pair_x=torch.from_numpy(np.stack(PX).astype(np.float32)),
                pair_ab=torch.from_numpy(np.array(PAB, dtype=np.int64)),
                y=torch.tensor(labels, dtype=torch.float32))


def train_link_predictor(base_graphs: list, epochs: int | None = None,
                         lr: float | None = None, max_gap: float | None = None):
    """Train the GCN link predictor on synthetic occlusions of the base graphs.

    max_gap=None (default) derives the threshold per graph via graph_max_gap().
    """
    epochs = epochs or config.GNN_EPOCHS
    lr = lr or config.GNN_LR
    torch.manual_seed(config.SEED)
    model = GCNLinkPredictor(node_dim=6, pair_dim=15, hidden=config.GNN_HIDDEN,
                             layers=config.GNN_LAYERS)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(config.SEED)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[gnn] GCN link predictor: {n_params} params | {epochs} epochs | "
          f"hidden={config.GNN_HIDDEN} layers={config.GNN_LAYERS} lr={lr}")
    model.train()
    for ep in range(1, epochs + 1):
        total, nbatch, npos = 0.0, 0, 0
        for _ in range(config.GNN_TRAIN_CORRUPTIONS):
            G_true = base_graphs[int(rng.integers(len(base_graphs)))]
            H, removed = corrupt_graph(G_true, rng, config.GNN_EDGE_DROP)
            gap = max_gap if max_gap else graph_max_gap(H)
            data = _corrupt_to_data(G_true, H, removed, rng, gap,
                                    config.GNN_NEG_PER_POS)
            if data is None or data.y.numel() == 0:
                continue
            opt.zero_grad()
            logits = model(data.x, data.edge_index, data.pair_x, data.pair_ab)
            loss = F.binary_cross_entropy_with_logits(logits, data.y)
            loss.backward()
            opt.step()
            total += loss.item(); nbatch += 1
            npos += int(data.y.sum().item())
        print(f"[gnn] epoch {ep}/{epochs} loss={total / max(nbatch, 1):.4f} "
              f"(batches={nbatch}, positives={npos})")
    return model


class GNNBridgingScorer:
    """Drop-in scorer for graph_builder.heal_graph (replaces _bridging_cost).

    heal_graph calls scorer(a, b, G) per candidate and expects a float COST
    (lower = better bridge) or None to forbid it.  We return cost = -logit, so a
    higher connect-probability becomes a lower cost and the SAME sorting +
    Union-Find logic runs unchanged.  The max_gap hard threshold is kept so both
    methods bridge exactly the same candidate set.
    """

    def __init__(self, model, max_gap: float):
        self.model = model.eval()
        self.max_gap = max_gap
        self._cache_key = None
        self._cache = None

    def _features_for(self, G: nx.Graph):
        key = (id(G), G.number_of_nodes(), G.number_of_edges())
        if self._cache_key == key:
            return self._cache
        pos = positions_m(G)
        nodes = list(G.nodes())
        idx = {n: i for i, n in enumerate(nodes)}
        _, feats, _ = node_features_matrix(G, pos)
        x = np.array([feats[n] for n in nodes], dtype=np.float32)
        ei = []
        for u, v in G.edges():
            ei.append([idx[u], idx[v]]); ei.append([idx[v], idx[u]])
        edge_index = torch.from_numpy(
            np.array(ei, dtype=np.int64).T if ei else np.zeros((2, 0), np.int64))
        self._cache_key = key
        self._cache = (pos, idx, feats, torch.from_numpy(x), edge_index)
        return self._cache

    def __call__(self, a, b, G: nx.Graph):
        pos, idx, feats, x, edge_index = self._features_for(G)
        d = math.hypot(pos[b][0] - pos[a][0], pos[b][1] - pos[a][1])
        if d > self.max_gap:
            return None                                   # same threshold as classical
        px = torch.from_numpy(
            pair_features(a, b, G, pos, feats, 1.0, self.max_gap)[None, :])
        ab = torch.tensor([[idx[a], idx[b]]], dtype=torch.long)
        with torch.no_grad():
            logit = self.model(x, edge_index, px, ab).item()
        return -logit                                     # lower cost = more likely edge


# ---------------------------------------------------------------------------
# Benchmark: classical vs learned on held-out synthetic occlusions (Part D)
# ---------------------------------------------------------------------------

def _false_bridges(stats: dict, removed_edges: set) -> int:
    """Bridges that are NOT among the truly-removed edges -- an automated
    stand-in for the manual 'False Bridges (visually inspected)' column."""
    return sum(1 for b in stats["bridges"]
               if frozenset((b[0], b[1])) not in removed_edges)


def benchmark_healing(base_graphs: list, gnn_model, n_graphs: int | None = None):
    """Heal held-out corrupted graphs with BOTH scorers and compare.

    Returns a results dict (also printed as the Part D table).
    """
    from src import graph_builder
    n_graphs = n_graphs or config.GNN_EVAL_CORRUPTIONS
    rng = np.random.default_rng(config.SEED + 999)
    rows = []
    for gi in range(n_graphs):
        G_true = base_graphs[gi % len(base_graphs)]
        H, removed = corrupt_graph(G_true, rng, config.GNN_EDGE_DROP)
        gap = graph_max_gap(H)
        scorer = GNNBridgingScorer(gnn_model, gap)
        pos = positions_m(H)
        t0 = time.perf_counter()
        _Hc, sc = graph_builder.heal_graph(H.copy(), scorer=None, pos=pos,
                                           max_gap=gap)
        tc = time.perf_counter() - t0
        t0 = time.perf_counter()
        _Hg, sg = graph_builder.heal_graph(H.copy(), scorer=scorer, pos=pos,
                                           max_gap=gap)
        tg = time.perf_counter() - t0
        rows.append({
            "graph": gi,
            "max_gap": gap,
            "removed_edges": len(removed),
            "classical": {
                "connectivity_ratio": sc["connectivity_ratio"],
                "components_before": sc["components_before"],
                "components_after": sc["components_after"],
                "bridges_added": sc["bridges_added"],
                "false_bridges": _false_bridges(sc, removed),
                "runtime_s": tc,
            },
            "gnn": {
                "connectivity_ratio": sg["connectivity_ratio"],
                "components_before": sg["components_before"],
                "components_after": sg["components_after"],
                "bridges_added": sg["bridges_added"],
                "false_bridges": _false_bridges(sg, removed),
                "runtime_s": tg,
            },
        })
    agg = {}
    for method in ("classical", "gnn"):
        agg[method] = {k: float(np.mean([r[method][k] for r in rows]))
                       for k in ("connectivity_ratio", "bridges_added",
                                 "false_bridges", "runtime_s")}
    return {"per_graph": rows, "summary": agg, "n_eval_graphs": n_graphs}


def print_part_d_table(bench: dict):
    """The Part D deliverable table (printed to the console)."""
    s = bench["summary"]
    print("\n" + "=" * 78)
    print("PART D -- LEARNED (GNN) vs CLASSICAL HEALING -- BENCHMARK")
    print("=" * 78)
    print(f"{'Method':<42}{'Connectivity':>13}{'Bridges':>10}"
          f"{'False Br.':>11}{'Runtime(s)':>12}")
    print("-" * 78)
    c, g = s["classical"], s["gnn"]
    print(f"{'Classical (Part B: MST + distance/angle)':<42}"
          f"{c['connectivity_ratio']:>12.1f}%{c['bridges_added']:>10.1f}"
          f"{c['false_bridges']:>11.1f}{c['runtime_s']:>12.3f}")
    print(f"{'Learned (Part D: GNN link prediction)':<42}"
          f"{g['connectivity_ratio']:>12.1f}%{g['bridges_added']:>10.1f}"
          f"{g['false_bridges']:>11.1f}{g['runtime_s']:>12.3f}")
    print("-" * 78)
    print(f"(averaged over {bench['n_eval_graphs']} held-out corrupted graphs)")
    print("=" * 78)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_link_predictor(model, path=config.GNN_MODEL):
    utils_ensure = config.OUTPUTS_DIR
    import os
    os.makedirs(utils_ensure, exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "node_dim": 6, "pair_dim": 15,
                "hidden": config.GNN_HIDDEN, "layers": config.GNN_LAYERS}, path)
    print(f"[gnn] saved link predictor -> {path}")


def load_link_predictor(path=config.GNN_MODEL):
    """Load a trained link predictor (used by the graph stage when
    config.HEALING_METHOD == 'gnn')."""
    import os
    if not os.path.exists(path):
        raise FileNotFoundError(f"no trained GNN link predictor at {path} "
                                f"(train it first: python scripts/run_part_d.py)")
    ck = torch.load(path, map_location="cpu")
    model = GCNLinkPredictor(node_dim=ck.get("node_dim", 6),
                             pair_dim=ck.get("pair_dim", 15),
                             hidden=ck.get("hidden", config.GNN_HIDDEN),
                             layers=ck.get("layers", config.GNN_LAYERS))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model