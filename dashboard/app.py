"""
dashboard/app.py -- interactive resilience dashboard (Part E).

Run from the project root:

    streamlit run dashboard/app.py

Shows the road network on a folium map, colors nodes by normalized betweenness
centrality (green -> yellow -> red), lets the user select a gatekeeper node
(via map click or the sidebar) and inspect its node-failure simulation results
from outputs/results.json, and provides a Reset Simulation control.

Coordinate honesty: when the graph uses RELATIVE pixel coordinates (not GPS),
the map maps them into a small labelled demo window around config.OSM_CENTER --
it is clearly annotated and never presented as a geographic map.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import folium
import matplotlib
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import streamlit as st
from streamlit_folium import st_folium

import config
from src import graph_io


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def coords_are_relative(G: nx.Graph) -> bool:
    """True when every lat/lon falls in [0,1] -> relative pixel coordinates."""
    lats = [d.get("lat", 0.0) for _, d in G.nodes(data=True)]
    lons = [d.get("lon", 0.0) for _, d in G.nodes(data=True)]
    return all(abs(v) <= 1.0001 for v in lats + lons)


def node_coords(G: nx.Graph, relative: bool) -> dict:
    """Map node_id -> (lat, lon) for folium.

    Relative coordinates are mapped into a small demo window around
    config.OSM_CENTER so the network shape is visible (NOT real GPS).
    """
    coords: dict = {}
    if relative:
        clat, clon = config.OSM_CENTER
        extent = 0.004                     # ~440 m demo window
        for nid, d in G.nodes(data=True):
            lon = clon + (float(d.get("lon", 0.5)) - 0.5) * 2.0 * extent
            lat = clat + (float(d.get("lat", 0.5)) - 0.5) * 2.0 * extent
            coords[nid] = (float(lat), float(lon))
    else:
        for nid, d in G.nodes(data=True):
            coords[nid] = (float(d.get("lat", 0.0)), float(d.get("lon", 0.0)))
    return coords


def centrality_scores(G: nx.Graph, results: dict | None) -> dict:
    """node_id -> betweenness centrality score (from results.json when available)."""
    if results and results.get("centrality"):
        return {int(e["node_id"]): float(e["score"]) for e in results["centrality"]}
    return nx.betweenness_centrality(G, weight="weight")


def centrality_colors(scores: dict) -> dict:
    """Continuous green->yellow->red hex colors from normalized scores.

    Green = low criticality, red = high criticality.  Visualization only, not
    any kind of political or subjective ranking.
    """
    vals = np.array(list(scores.values()), dtype=float)
    if vals.size and vals.max() > vals.min():
        norm = (vals - vals.min()) / (vals.max() - vals.min())
    else:
        norm = np.zeros(vals.size)
    cmap = matplotlib.colormaps["RdYlGn_r"]
    return {nid: mcolors.to_hex(cmap(v)) for nid, v in zip(scores.keys(), norm)}


def ablation_entry(results: dict | None, node: int):
    if not results:
        return None
    for run in results.get("ablation_runs", []):
        if int(run.get("removed_node")) == int(node):
            return run
    return None


def load_results():
    if config.RESULTS_JSON.exists():
        try:
            return json.loads(config.RESULTS_JSON.read_text())
        except (json.JSONDecodeError, OSError):
            return None
# ---------------------------------------------------------------------------
# Map building
# ---------------------------------------------------------------------------

def build_map(G: nx.Graph, coords: dict, colors: dict, results: dict | None,
              selected: int | None = None, relative: bool = False,
              scores: dict | None = None) -> folium.Map:
    """Folium map: edges as polylines, nodes colored by normalized centrality.

    When `selected` is set, the simulated-failure view is drawn: the removed
    node is marked with a red X and its incident edges are omitted.
    """
    scores = scores or {}
    if relative:
        m = folium.Map(location=list(config.OSM_CENTER), zoom_start=14)
    else:
        lats = [c[0] for c in coords.values()]
        lons = [c[1] for c in coords.values()]
        m = folium.Map(location=[float(np.mean(lats)), float(np.mean(lons))],
                       zoom_start=13)

    # edges (incident edges of the removed node are dropped in failure view)
    for u, v in G.edges():
        if selected is not None and (u == selected or v == selected):
            continue
        pts = [coords[u], coords[v]]
        folium.PolyLine(pts, color="gray", weight=2, opacity=0.7).add_to(m)

    # nodes colored by centrality
    for nid, (lat, lon) in coords.items():
        color = colors.get(nid, "#888888")
        if nid == selected:
            folium.CircleMarker(
                location=(lat, lon), radius=13, color="black",
                fillColor="#ff0000", fill=True, fillOpacity=0.9,
                tooltip="REMOVED node (simulated failure)",
            ).add_to(m)
            continue
        entry = ablation_entry(results, nid)
        if entry is not None and entry.get("note"):
            label = f"<b>Node {nid}</b><br><i>{entry['note']}</i>"
        else:
            ri = entry.get("resilience_index") if entry else None
            sc = scores.get(nid, float("nan"))
            label = f"<b>Node {nid}</b><br>centrality {sc:.4f}" + \
                (f"<br>Resilience: {ri}" if ri is not None else "")
        folium.CircleMarker(
            location=(lat, lon), radius=6, color=color,
            fillColor=color, fill=True, fillOpacity=0.9,
            popup=folium.Popup(label, max_width=280),
        ).add_to(m)
    return m
# ---------------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------------

def render():
    st.set_page_config(page_title="Road Network Resilience", layout="wide")
    st.title("Road Network Resilience")
    st.caption("Occluded satellite imagery -> road extraction -> graph -> "
               "centrality & simulated node-failure analysis")

    # ---------------- sidebar ----------------
    with st.sidebar:
        st.header("Road Network Resilience")
        source = st.selectbox(
            "Graph source",
            ["osm", "teammate_export", "graphml"],
            index=["osm", "teammate_export", "graphml"].index(config.GRAPH_SOURCE)
            if config.GRAPH_SOURCE in ["osm", "teammate_export", "graphml"] else 0,
        )
        top_n = st.slider("Top-N nodes", 1, 50, int(config.TOP_N))
        if st.button("Reset Simulation"):
            st.session_state.pop("selected_node", None)
            st.rerun()

    # ---------------- data ----------------
    results = load_results()
    G = graph_io.load_graph(source)
    if G is None or G.number_of_nodes() == 0:
        st.error("No graph available. Run `python main.py --stage demo` "
                 "(or `--stage all` / `--stage graph` + `--stage analysis`) first.")
        st.stop()
    if not results:
        st.info("outputs/results.json not found -- centrality is computed live; "
                "run `python main.py --stage analysis` for the full ablation results.")

    relative = coords_are_relative(G)
    scores = centrality_scores(G, results)
    colors = centrality_colors(scores)
    coords = node_coords(G, relative)
    top_nodes = [int(n) for n, _ in sorted(scores.items(), key=lambda kv: kv[1],
                                            reverse=True)[:top_n]]

    # gatekeeper ids listed in the sidebar
    with st.sidebar:
        st.markdown("**Top-N gatekeeper nodes:**")
        st.write(top_nodes)

    # ---------------- selection (click or dropdown) ----------------
    node_options = {int(n): f"Node {n}" for n in G.nodes()}
    if "selected_node" not in st.session_state:
        st.session_state.selected_node = top_nodes[0] if top_nodes else None
    with st.sidebar:
        sel = st.selectbox(
            "Selected node",
            list(node_options.keys()),
            index=list(node_options.keys()).index(st.session_state.selected_node)
            if st.session_state.selected_node in node_options else 0,
            format_func=lambda n: node_options[n],
        )
    st.session_state.selected_node = sel
    selected = st.session_state.selected_node

    # ---------------- map ----------------
    m = build_map(G, coords, colors, results, selected=selected,
                  relative=relative, scores=scores)
    if relative:
        st.caption("Map viewer: coordinates are RELATIVE pixel coordinates mapped "
                   "into a small labelled demo window around the demo center -- "
                   "this is NOT a geographic/GPS map.")
    click_info = st_folium(m, height=520, use_container_width=True,
                           returned_objects=["last_object_clicked"],
                           key=f"roadmap_{source}_{selected or 'none'}")
    clicked = click_info.get("last_object_clicked") if click_info else None
    if isinstance(clicked, str):
        mth = re.search(r"Node\s+(\d+)", clicked)
        if mth and int(mth.group(1)) in G:
            st.session_state.selected_node = int(mth.group(1))
            selected = st.session_state.selected_node
            st.rerun()

    # ---------------- failure-simulation stats ----------------
    entry = ablation_entry(results, selected)
    st.markdown("---")
    st.subheader(f"Selected Node: {selected}")
    if not results:
        st.info("No ablation results available (results.json missing).")
    elif entry is None:
        st.info(f"Node {selected} has no ablation entry (may not be in the top-N set).")
    else:
        c1, c2, c3, c4 = st.columns(4)
        base = results.get("baseline_avg_path_length")
        c1.metric("Baseline Avg Path Length",
                  f"{base:.2f}" if base is not None else "n/a")
        c2.metric("After Removal",
                  f"{entry['avg_path_length']:.2f}" if entry.get("avg_path_length") else "n/a")
        c3.metric("Resilience Index",
                  f"{entry['resilience_index']:.4f}" if entry.get("resilience_index") else "n/a")
        chg = entry.get("path_length_change_percent")
        c4.metric("Path Length Change",
                  f"{chg:.2f}%" if chg is not None else "n/a")
        if entry.get("note"):
            st.warning(entry["note"])

    st.markdown("**Colors** (visualization only, not a ranking): "
                "low centrality -> green, medium -> yellow, high -> red.  "
                "Selecting a node shows the simulated-failure view (removed node "
                "highlighted, incident edges omitted).")

    st.caption("Full route-by-route rerouting is a documented future enhancement; "
               "the MVP shows the before/after statistics and removed-node view.")


if __name__ == "__main__":
    render()