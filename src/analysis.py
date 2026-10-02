"""
PART D -- Network analysis on the shared road graph.

Input: any nx.Graph loaded through src.graph_io.load_graph() (nodes.json +
edges.json or graph.graphml) -- this module NEVER depends on segmentation
internals.

Steps:
    validate_graph            (D1)
    calculate_centrality      (D2)  betweenness centrality, TOP_N gatekeepers
    baseline_avg_path_length  (D3)  largest connected component
    run_ablation              (D4)  node-removal resilience simulation
    build_results/save        (D5)  outputs/results.json (exact core schema)
    plot_centrality_top10     (D6)  outputs/plots/centrality_top10.png

Terminology: "gatekeeper" is purely the top-N highest-betweenness node in this
project; no subjective/political ranking is implied.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from src import utils
import config


# ---------------------------------------------------------------------------
# D1 -- Graph validation
# ---------------------------------------------------------------------------

def validate_graph(G: nx.Graph) -> bool:
    """Graceful validation with useful errors (never crashes on bad input).

    Returns True when the graph is usable for analysis.
    """
    ok = True
    if G is None:
        print("[analysis] ERROR: no graph loaded (None).")
        return False
    if G.number_of_nodes() == 0:
        print("[analysis] ERROR: graph is empty -- nothing to analyze.")
        return False
    bad = []
    for u, v in G.edges():
        if u not in G or v not in G:
            bad.append((u, v))
    if bad:
        ok = False
        print(f"[analysis] ERROR: {len(bad)} edge(s) reference missing nodes, e.g. {bad[:5]}")
    missing_weight = 0
    for _u, _v, d in G.edges(data=True):
        w = d.get("weight", None)
        if w is None:
            missing_weight += 1
    if missing_weight:
        print(f"[analysis] WARNING: {missing_weight} edge(s) have no weight -> fallback 1.0 "
              f"applied (documented).")
    if nx.number_connected_components(G) == 0:
        print("[analysis] ERROR: graph has no connected component.")
        return False
    n_comp = nx.number_connected_components(G)
    if n_comp > 1:
        print(f"[analysis] NOTE: graph is disconnected ({n_comp} components). "
              f"Path-length metrics use the LARGEST connected component.")
    return ok


# ---------------------------------------------------------------------------
# D2 -- Centrality
# ---------------------------------------------------------------------------

def calculate_centrality(G: nx.Graph) -> dict:
    """Betweenness centrality for every node (weighted by road length)."""
    return nx.betweenness_centrality(G, weight="weight")


def top_n_nodes(G: nx.Graph, centrality: dict, top_n: int = config.TOP_N) -> list:
    """TOP_N gatekeeper candidates, sorted by centrality descending."""
    ranked = sorted(centrality.items(), key=lambda kv: kv[1], reverse=True)
    top = [int(n) for n, _ in ranked[:top_n]]
    print("[analysis] top gatekeeper candidates:", top)
    return top


# ---------------------------------------------------------------------------
# D3 -- Baseline average shortest path length (largest component)
# ---------------------------------------------------------------------------

def largest_component(G: nx.Graph):
    if G.number_of_nodes() == 0:
        return None
    return max(nx.connected_components(G), key=len)


def baseline_avg_path_length(G: nx.Graph):
    """Average shortest path length (weighted) on the largest connected component."""
    nodes = largest_component(G)
    if nodes is None:
        return None
    lcc = G.subgraph(nodes)
    if lcc.number_of_nodes() < 2:
        print("[analysis] NOTE: largest component has < 2 nodes; "
              "average shortest path length is undefined (stored as null).")
        return None
    return float(nx.average_shortest_path_length(lcc, weight="weight"))


# ---------------------------------------------------------------------------
# D4 -- Node ablation / failure simulation
# ---------------------------------------------------------------------------

def run_ablation(G: nx.Graph, top_nodes: list, baseline: float | None) -> list:
    """Remove each top-N node and measure the resilience impact.

    For every node:
        perturbed_avg_path_length   -- on the largest CC after removal
        resilience_index            -- baseline / perturbed (D4)
        path_length_change_percent  -- (perturbed - baseline) / baseline * 100

    When removal empties the graph or yields a trivial component, null/error
    fields are stored instead of crashing.
    """
    runs = []
    baseline_ok = baseline is not None and baseline > 0
    for node in top_nodes:
        if node not in G:
            print(f"[analysis] node {node} not in graph -- skipped in ablation")
            continue
        H = G.copy()
        H.remove_node(node)
        entry = {
            "removed_node": int(node),
            "resilience_index": None,
            "avg_path_length": None,
            "path_length_change_percent": None,
            "note": None,
        }
        if H.number_of_nodes() == 0:
            entry["note"] = "graph became empty after node removal"
            runs.append(entry)
            continue
        nodes = largest_component(H)
        lcc = H.subgraph(nodes)
        if lcc.number_of_nodes() < 2:
            entry["note"] = "largest component has < 2 nodes after removal"
            runs.append(entry)
            continue
        perturbed = float(nx.average_shortest_path_length(lcc, weight="weight"))
        entry["avg_path_length"] = round(perturbed, 4)
        if baseline_ok:
            entry["resilience_index"] = round(baseline / perturbed, 4)
            entry["path_length_change_percent"] = round(
                (perturbed - baseline) / baseline * 100.0, 4)
        else:
            entry["note"] = (entry.get("note") or "") + " baseline path length undefined"
        runs.append(entry)
    return runs


# ---------------------------------------------------------------------------
# D5 -- results.json
# ---------------------------------------------------------------------------

def build_results(G: nx.Graph, centrality: dict, top_nodes: list, baseline,
                  ablation_runs: list) -> dict:
    """Core schema (required fields are never renamed):

        centrality:   [{"node_id": int, "score": float}]
        ablation_runs:[{"removed_node": int, "resilience_index":..,
                        "avg_path_length":..}]
    plus useful extras (baseline_avg_path_length, path_length_change_percent).
    """
    centrality_sorted = sorted(centrality.items(), key=lambda kv: kv[1], reverse=True)
    results = {
        "graph_source": config.GRAPH_SOURCE,
        "baseline_avg_path_length": baseline,
        "graph_summary": {
            "nodes": int(G.number_of_nodes()),
            "edges": int(G.number_of_edges()),
            "components": int(nx.number_connected_components(G)),
        },
        "centrality": [{"node_id": int(n), "score": round(s, 6)}
                       for n, s in centrality_sorted],
        "top_gatekeeper_nodes": top_nodes,
        "ablation_runs": ablation_runs,
    }
    return results


def save_results(results: dict) -> None:
    config.RESULTS_JSON.write_text(json.dumps(results, indent=2))
    print(f"[analysis] results -> {config.RESULTS_JSON.name}")


# ---------------------------------------------------------------------------
# D6 -- Centrality plot
# ---------------------------------------------------------------------------

def plot_centrality_top10(centrality: dict, top_n: int = config.TOP_N,
                          path=config.PLOTS_OUT / "centrality_top10.png") -> None:
    utils.ensure_dirs(config.PLOTS_OUT)
    ranked = sorted(centrality.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    if not ranked:
        print("[plot] no centrality data to plot")
        return
    ids = [int(n) for n, _ in ranked][::-1]
    scores = [s for _, s in ranked][::-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(ids) + 1.2))
    ax.barh([str(i) for i in ids], scores, color="darkorange")
    ax.set_xlabel("Betweenness Centrality (weighted by road length)")
    ax.set_ylabel("Node ID")
    ax.set_title(f"Top-{len(ids)} Gatekeeper Candidates")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] centrality top-{len(ids)} -> {path}")
# ---------------------------------------------------------------------------
# Orchestration + final summary (Part J)
# ---------------------------------------------------------------------------

def analyze_graph(G: nx.Graph) -> dict:
    """Run the full PART D analysis on an already-loaded graph.

    Returns the results dict and leaves outputs/results.json + centrality plot on disk.
    """
    if not validate_graph(G):
        print("[analysis] validation failed; aborting analysis.")
        sys.exit(1)
    centrality = calculate_centrality(G)
    top = top_n_nodes(G, centrality, config.TOP_N)
    baseline = baseline_avg_path_length(G)
    print(f"[analysis] baseline avg shortest path length "
          f"(largest component): {baseline}")
    ablation = run_ablation(G, top, baseline)
    results = build_results(G, centrality, top, baseline, ablation)
    save_results(results)
    plot_centrality_top10(centrality)
    return results


def load_existing_results():
    """Re-read outputs/results.json if present (used by the final summary / dashboard)."""
    if config.RESULTS_JSON.exists():
        try:
            return json.loads(config.RESULTS_JSON.read_text())
        except (json.JSONDecodeError, OSError):
            return None
    return None


def load_segmentation_metrics():
    if config.SEG_METRICS_JSON.exists():
        try:
            return json.loads(config.SEG_METRICS_JSON.read_text())
        except (json.JSONDecodeError, OSError):
            return None
    return None


def print_final_summary(results: dict | None, graph_stats: dict | None = None,
                        seg_metrics: dict | None = None) -> None:
    """Part J -- final terminal output."""
    print("\n==================================================")
    print("ROAD NETWORK MVP -- FINAL RESULTS")
    print("==================================================")

    seg_metrics = seg_metrics or load_segmentation_metrics()
    results = results or load_existing_results()

    print("\nPART A -- SEGMENTATION")
    if seg_metrics:
        print(f"Test IoU              : {seg_metrics.get('test_iou', float('nan')):.3f}")
        print(f"Test Dice             : {seg_metrics.get('test_dice', float('nan')):.3f}")
        print(f"Occlusion-Recall      : {seg_metrics.get('occlusion_recall', float('nan')):.3f}")
    else:
        print("(not run -- no segmentation metrics found)")

    print("\nPART B -- GRAPH")
    if results:
        gs = results.get("graph_summary", {})
        print(f"Nodes                 : {gs.get('nodes', '?')}")
        print(f"Edges                 : {gs.get('edges', '?')}")
    if graph_stats:
        print(f"Components Before     : {graph_stats.get('components_before', '?')}")
        print(f"Components After      : {graph_stats.get('components_after', '?')}")
        print(f"Bridges Added         : {graph_stats.get('bridges_added', '?')}")
        print(f"Connectivity Ratio    : {graph_stats.get('connectivity_ratio', 0.0):.2f}%")

    print("\nPART C -- NETWORK ANALYSIS")
    if results:
        print(f"Baseline Avg Path     : {results.get('baseline_avg_path_length')}")
        print(f"Top Gatekeeper Nodes  : {results.get('top_gatekeeper_nodes')}")

    print("\nPART D -- OUTPUTS")
    print("outputs/")
    print("|-- best_model.pth")
    print("|-- masks/")
    print("|-- tiles_manifest.json")
    print("|-- graph.graphml")
    print("|-- nodes.json")
    print("|-- edges.json")
    print("|-- results.json")
    print("\\-- plots/")
    print("\n==================================================")
    print("MVP COMPLETE")
    print("==================================================\n")