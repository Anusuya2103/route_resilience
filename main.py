"""
main.py -- one-command MVP pipeline (Part F).

    python main.py                 # == --stage all
    python main.py --stage all
    python main.py --stage demo          # OSM -> graph -> analysis (offline-safe)
    python main.py --stage extraction    # train / evaluate / export masks
    python main.py --stage graph         # masks -> skeleton -> heal -> graph files
    python main.py --stage analysis      # centrality -> ablation -> results.json

Extra switches:
    --graph-from {auto,masks,graph_source}  how the graph stage picks its input
    --source {osm,teammate_export,graphml}  override config.GRAPH_SOURCE
    --samples N                             override config.MAX_TRAIN_SAMPLES
    --epochs N                              override config.EPOCHS
    --use-synthetic                         force the procedural dataset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from src import utils, extraction, graph_builder, graph_io, analysis


def _banner(text: str) -> None:
    utils.print_banner(f"STAGE: {text}")
    utils.ensure_dirs(config.OUTPUTS_DIR, config.MASKS_OUT, config.PLOTS_OUT)


def _apply_overrides(args: argparse.Namespace) -> None:
    if args.samples is not None:
        config.MAX_TRAIN_SAMPLES = int(args.samples)
        config.SYNTHETIC_FALLBACK = True       # always allow the sized synthetic set
    if args.epochs is not None:
        config.EPOCHS = int(args.epochs)
    if args.source is not None:
        config.GRAPH_SOURCE = args.source
        print(f"[cli] GRAPH_SOURCE -> {config.GRAPH_SOURCE}")
    if args.use_synthetic:
        config.SYNTHETIC_FALLBACK = True
    if args.model is not None:
        config.MODEL_CHOICE = args.model
        print(f"[cli] MODEL_CHOICE -> {config.MODEL_CHOICE}")
    if args.healing is not None:
        config.HEALING_METHOD = args.healing
        print(f"[cli] HEALING_METHOD -> {config.HEALING_METHOD}")
    print(f"[cli] GRAPH_SOURCE={config.GRAPH_SOURCE}  DEMO_MODE={config.DEMO_MODE}  "
          f"EPOCHS={config.EPOCHS}  MAX_TRAIN_SAMPLES={config.MAX_TRAIN_SAMPLES}")
    print(f"[cli] MODEL_CHOICE={config.MODEL_CHOICE}  HEALING_METHOD={config.HEALING_METHOD}")


# ---------------------------------------------------------------------------
# Stage implementations
# ---------------------------------------------------------------------------

def run_extraction_stage() -> dict:
    _banner("EXTRACTION (segmentation training + evaluation + mask export)")
    if config.MODEL_CHOICE == "attention":
        # Part C model: trained separately by scripts/run_part_c.py
        if not config.ATTENTION_MODEL.exists():
            print("[cli] attention checkpoint missing -- train it first:")
            print("    python scripts/run_part_c.py")
            sys.exit(1)
        seg = extraction.evaluate_segmentation_model()
    else:
        extraction.train_segmentation_model(use_synthetic=None)
        seg = extraction.evaluate_segmentation_model()
    extraction.export_masks(model_choice=config.MODEL_CHOICE)
    return seg


def run_graph_stage() -> tuple:
    _banner(f"GRAPH (stitch -> skeleton -> graph -> heal [{config.HEALING_METHOD}] -> export)")
    G, stats = graph_builder.build_graph_pipeline(use_demo=True)
    graph_io.save_graph(G)
    print(f"[graph] final graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return stats


def run_analysis_stage() -> dict:
    _banner("ANALYSIS (centrality + node-failure ablation)")
    G = graph_io.load_graph(config.GRAPH_SOURCE)
    if G is None:
        print("[analysis] no graph could be loaded -- aborting.")
        sys.exit(1)
    graph_io.save_graph(G)
    return analysis.analyze_graph(G)


def run_demo_stage() -> dict:
    _banner("DEMO (OSM/graph-source -> graph -> centrality -> ablation -> dashboard data)")
    G = graph_io.load_graph(config.GRAPH_SOURCE)
    if G is None:
        print("[demo] no graph could be loaded -- aborting.")
        sys.exit(1)
    graph_io.save_graph(G)
    utils.print_section("Demo graph source ready")
    return analysis.analyze_graph(G)


def run_all_stage(args: argparse.Namespace) -> dict:
    _banner("ALL")
    seg_metrics = None
    graph_stats = None

    # 1) segmentation half -- only when explicitly requested (not pure demo mode)
    if (not config.DEMO_MODE) or args.use_synthetic:
        seg_metrics = run_extraction_stage()

    # 2) graph half -- where does the graph come from?
    masks_present = config.TILES_MANIFEST.exists() and \
        bool(list(config.MASKS_OUT.glob("mask_tile_*.png")))
    use_masks = args.graph_from == "masks" or \
        (args.graph_from == "auto" and masks_present and not config.DEMO_MODE)

    if use_masks:
        G, graph_stats = graph_builder.build_graph_pipeline(use_demo=not masks_present)
        graph_io.save_graph(G)
        utils.print_section("Re-loading freshly exported graph through the shared contract "
                            "(teammate_export mode) to validate the interface")
        results = analysis.analyze_graph(graph_io.load_graph("teammate_export"))
    else:
        G = graph_io.load_graph(config.GRAPH_SOURCE)
        if G is None:
            print("[all] no graph could be loaded -- aborting.")
            sys.exit(1)
        graph_io.save_graph(G)
        utils.print_section("Analyzing graph loaded through the shared contract")
        results = analysis.analyze_graph(G)

    # 3) final summary
    analysis.print_final_summary(results, graph_stats=graph_stats, seg_metrics=seg_metrics)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Road Network Reconstruction & Resilience MVP",
    )
    parser.add_argument("--stage", choices=["all", "demo", "extraction", "graph", "analysis"],
                        default="all", help="which part of the pipeline to run")
    parser.add_argument("--graph-from", choices=["auto", "masks", "graph_source"],
                        default="auto",
                        help="input for the graph stage when running --stage all")
    parser.add_argument("--source", choices=["osm", "teammate_export", "graphml"],
                        default=None, help="override config.GRAPH_SOURCE")
    parser.add_argument("--samples", type=int, default=None,
                        help="override config.MAX_TRAIN_SAMPLES")
    parser.add_argument("--epochs", type=int, default=None,
                        help="override config.EPOCHS")
    parser.add_argument("--use-synthetic", action="store_true",
                        help="force the procedurally generated dataset in the extraction stage")
    parser.add_argument("--model", choices=["baseline", "attention"], default=None,
                        help="which segmentation model exports the masks "
                             "(baseline=Part A, attention=Part C)")
    parser.add_argument("--healing", choices=["classical", "gnn"], default=None,
                        help="graph-healing scorer (classical=Part B, gnn=Part D)")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    utils.set_seed(config.SEED)
    _apply_overrides(args)

    if args.stage == "extraction":
        run_extraction_stage()
    elif args.stage == "graph":
        run_graph_stage()
    elif args.stage == "analysis":
        run_analysis_stage()
    elif args.stage == "demo":
        run_demo_stage()
    else:
        run_all_stage(args)


if __name__ == "__main__":
    main()