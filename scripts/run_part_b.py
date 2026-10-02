"""
PART B driver -- Graph skeletonization + CLASSICAL topological healing.

    python scripts/run_part_b.py               # classical healing (baseline, default)
    python scripts/run_part_b.py --method gnn  # use the Part D learned scorer instead

Pipeline:
  1. stitch tile masks (outputs/masks + outputs/tiles_manifest.json)
  2. skeletonize  (skimage.morphology.skeletonize)
  3. node & edge extraction (endpoints / intersections / chain walks, weight=pixels)
  4. classical healing: candidate endpoint pairs scored by
     cost = DISTANCE_WEIGHT*normalized_distance + ANGLE_WEIGHT*normalized_angle,
     selected with Union-Find (MST-style) within MAX_GAP_DISTANCE
  5. shared coordinate export (relative normalized lat/lon) and
     graph.graphml + nodes.json + edges.json  <- the teammate's input contract

Classical healing is the BASELINE for Part D's comparison and remains the
default; --method gnn only swaps the scoring function.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from src import utils, graph_builder, graph_io


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=["classical", "gnn"], default=None,
                    help="healing scorer (default: config.HEALING_METHOD)")
    ap.add_argument("--no-demo-masks", action="store_true",
                    help="fail instead of generating demo masks when none exist")
    args = ap.parse_args()

    if args.method is not None:
        config.HEALING_METHOD = args.method
    utils.set_seed(config.SEED)
    utils.print_banner(f"PART B -- GRAPH SKELETONIZATION + HEALING ({config.HEALING_METHOD})")

    G, stats = graph_builder.build_graph_pipeline(use_demo=not args.no_demo_masks)
    graph_io.save_graph(G)
    print(f"\n[partB] final graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print("[partB] contract written: graph.graphml, nodes.json, edges.json "
          "(integer node IDs, pixel weights)")
    return G, stats


if __name__ == "__main__":
    main()