"""
config.py -- single source of truth for the Road Network Reconstruction &
Resilience MVP.

Every tunable knob of the pipeline lives here.  All paths are *relative* to the
project root so the repository can be cloned and run anywhere (there are no
hardcoded absolute Windows paths anywhere in the codebase).
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent

DATA_DIR = ROOT_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"              # raw satellite images (real dataset)
MASKS_DIR = DATA_DIR / "masks"                # raw ground-truth masks (real dataset)
TEST_REGION_DIR = DATA_DIR / "test_region"    # adjacent tiles used for mask export
PATCHES_DIR = DATA_DIR / "patches"            # 256x256 patches built by preprocessing

OUTPUTS_DIR = ROOT_DIR / "outputs"
MASKS_OUT = OUTPUTS_DIR / "masks"
PLOTS_OUT = OUTPUTS_DIR / "plots"
OSM_CACHE_DIR = OUTPUTS_DIR / "osm_cache"

# Shared graph contract files (the ONLY interface between extraction & analysis)
TEAMMATE_NODES = OUTPUTS_DIR / "nodes.json"
TEAMMATE_EDGES = OUTPUTS_DIR / "edges.json"
TEAMMATE_GRAPHML = OUTPUTS_DIR / "graph.graphml"

RESULTS_JSON = OUTPUTS_DIR / "results.json"
TILES_MANIFEST = OUTPUTS_DIR / "tiles_manifest.json"
BEST_MODEL = OUTPUTS_DIR / "best_model.pth"
SPLIT_JSON = OUTPUTS_DIR / "dataset_split.json"
SEG_METRICS_JSON = OUTPUTS_DIR / "segmentation_metrics.json"

# --- Novelty / research-extension outputs (Parts C & D) ----------------------
ATTENTION_MODEL = OUTPUTS_DIR / "best_attention_model.pth"
SEG_METRICS_ATTENTION_JSON = OUTPUTS_DIR / "segmentation_metrics_attention.json"
PART_C_TABLE_JSON = OUTPUTS_DIR / "partC_comparison.json"
PART_C_TABLE_PNG = PLOTS_OUT / "partC_comparison.png"
GNN_MODEL = OUTPUTS_DIR / "gnn_link_predictor.pt"
GNN_PAIRS_TRAIN = OUTPUTS_DIR / "gnn_pairs_train.pt"
GNN_PAIRS_VAL = OUTPUTS_DIR / "gnn_pairs_val.pt"
PART_D_TABLE_JSON = OUTPUTS_DIR / "partD_comparison.json"
PART_D_TABLE_PNG = PLOTS_OUT / "partD_comparison.png"
PART_D_FALSE_BRIDGES_PNG = PLOTS_OUT / "healing_comparison_grid.png"
GNNCACHE_DIR = OSM_CACHE_DIR  # reuse the OSM cache for Part D base graphs
BASEMAP_DIR = OUTPUTS_DIR / "basemap"   # offline raster basemap images

# ---------------------------------------------------------------------------
# Pipeline control
# ---------------------------------------------------------------------------
# Graph source used by the analysis / dashboard stages:
#   "osm"              -> download a small real neighbourhood via OSMnx (demo/dev);
#                         falls back to a built-in grid graph when offline
#   "teammate_export"  -> nodes.json + edges.json produced by the extraction pipeline
#   "graphml"          -> graph.graphml
GRAPH_SOURCE = "osm"

# In demo mode, `python main.py --stage all` skips segmentation training and
# builds the graph straight from GRAPH_SOURCE, demonstrating the analysis +
# dashboard half immediately (the demo-first requirement).
DEMO_MODE = True

# If no real dataset is found at data/images + data/masks, the extraction stage
# can fall back to a small procedurally generated road dataset so the whole
# pipeline remains runnable offline.  Pass --use-synthetic to force it.
SYNTHETIC_FALLBACK = True

# ---------------------------------------------------------------------------
# Dataset / preprocessing
# ---------------------------------------------------------------------------
IMAGE_SIZE = 256
MAX_TRAIN_SAMPLES = 200        # cap on (image, mask) pairs used for training
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
# TEST_FRACTION = 1.0 - TRAIN_FRACTION - VAL_FRACTION

# Standard ImageNet normalization used by the ResNet encoder family.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Synthetic occlusion augmentation -- only applied to a portion of training data.
OCCLUSION_PROB = 0.35          # fraction of training samples that get occluded
OCCLUSION_DARK_PROB = 0.6      # dark-rectangle effect probability (when occluding)
OCCLUSION_BLUR_PROB = 0.6      # gaussian-blur-patch effect probability
OCCLUSION_SHADOW_PROB = 0.5    # soft shadow-gradient effect probability

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
EPOCHS = 10
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_WORKERS = 0                # 0 = safest on Windows / CPU-only machines

# ---------------------------------------------------------------------------
# Graph construction / topological healing (weights are in PIXELS)
# ---------------------------------------------------------------------------
MIN_COMPONENT_SIZE = 20        # tiny mask components removed before skeletonization
MAX_GAP_DISTANCE = 30          # max bridge length (pixels) for topological healing
DISTANCE_WEIGHT = 0.7          # bridge-scoring weight for normalized distance
ANGLE_WEIGHT = 0.3             # bridge-scoring weight for normalized angle
METERS_PER_PIXEL = None        # leave None -> weight unit stays "pixels"

# ---------------------------------------------------------------------------
# Network analysis
# ---------------------------------------------------------------------------
TOP_N = 10                     # gatekeeper candidates (top betweenness centrality)

# ---------------------------------------------------------------------------
# OSM placeholder graph (Part C, mode 3)
# ---------------------------------------------------------------------------
OSM_CENTER = (13.031647, 80.270017)   # Mylapore, Chennai
OSM_DIST = 1200                    # metres around the center point
OSM_NETWORK_TYPE = "drive"
REFRESH_OSM_CACHE = False

# Overpass mirrors, tried in order.  The main overpass-api.de endpoint rate-limits
# hard and starts resetting connections during a multi-area build, so mirrors are
# rotated with backoff between attempts instead of failing the area outright.
OSM_OVERPASS_MIRRORS = [
    "https://overpass.kumi.systems/api",
    "https://overpass-api.de/api",
    "https://overpass.osm.ch/api",
]
OSM_OVERPASS_TRIES = 4
OSM_OVERPASS_BACKOFF_S = 10
# Courtesy pause between monitored areas.  Nine consecutive Overpass queries
# will trip the public rate limiter, which resets the connection mid-download.
OSM_AREA_PAUSE_S = 8
# Public mirrors reject anonymous/default agents with HTTP 429.
OSM_HTTP_USER_AGENT = ("road-resilience-dashboard/1.0 (academic road-network "
                       "resilience study; non-commercial)")
OSM_HTTP_TIMEOUT_S = 90

# Place label reported on the dashboard (e.g. "Mylapore, Chennai").
OSM_PLACE_LABEL = "Mylapore, Chennai"

# ---------------------------------------------------------------------------
# Areas to monitor
# ---------------------------------------------------------------------------
# The console is multi-area: one operator watches a home town AND every other
# locality on the list, under the SAME running scenario, so the effects can be
# compared side by side.  Each entry is built into its own cached graph.
#
#   key     short id used in the URL/selector
#   label   human name shown on the dashboard
#   center  (lat, lon) -- filled in by scripts/set_area.py via Nominatim
#   dist    radius in metres for the detailed street network
#   role    "home" (highlighted) or "watch" (monitored from a distance)
# OSM core API fallback -- a completely different server from Overpass, so it
# still works when the public Overpass mirrors are rate-limiting or timing out.
OSM_CORE_API = "https://api.openstreetmap.org/api/0.6/map"
OSM_USE_CORE_API_FALLBACK = True
# The core API returned a full 1.5 km town in ~2 s while every Overpass mirror
# was timing out, so it is tried FIRST and Overpass is kept as the backup.  The
# core API caps a bbox at ~0.25 square degrees, which comfortably covers the
# per-town radii this console uses (and the 8 km "wide" view).
OSM_PREFER_CORE_API = True
OSM_MAX_CORE_API_RADIUS_M = 12000
# OSM coverage varies a lot between towns: the point Nominatim matched for
# Thoothukudi sits in a sparsely-mapped pocket with only ~23 highway ways in
# 1.5 km, which is far too small to simulate anything meaningful.  When an area
# comes back with fewer than this many intersections, the radius is doubled and
# re-fetched (up to OSM_AREA_EXPANSION_TRIES times).
OSM_MIN_AREA_NODES = 150
OSM_AREA_EXPANSION_TRIES = 3
OSM_MAX_AREA_RADIUS_M = 12000
# Translated from osmnx's own drive filter (ox._overpass._get_network_filter
# ("drive")) so the fallback yields the same street set as the Overpass path.
OSM_DRIVE_EXCLUDED_HIGHWAY = {
    "abandoned", "bridleway", "bus_guideway", "construction", "corridor",
    "cycleway", "elevator", "escalator", "footway", "no", "path", "pedestrian",
    "planned", "platform", "proposed", "raceway", "razed", "rest_area",
    "service", "services", "steps", "track",
}
OSM_DRIVE_EXCLUDED_SERVICE = {
    "alley", "driveway", "emergency_access", "parking", "parking_aisle", "private",
}
OSM_AREAS_FILE = OUTPUTS_DIR / "areas.json"
OSM_AREA_CACHE = OUTPUTS_DIR / "area_cache"


def load_areas() -> list:
    """Read the monitored-area list, falling back to the single OSM_CENTER."""
    import json
    if OSM_AREAS_FILE.exists():
        try:
            areas = json.loads(OSM_AREAS_FILE.read_text(encoding="utf-8"))
            if areas:
                return areas
        except (json.JSONDecodeError, OSError):
            print(f"[areas] {OSM_AREAS_FILE} unreadable - using OSM_CENTER")
    return [{"key": "area", "label": OSM_PLACE_LABEL or "Pinned area",
             "center": [float(OSM_CENTER[0]), float(OSM_CENTER[1])],
             "dist": int(OSM_DIST), "role": "home"}]


# Real emergency facilities pulled from OSM POIs (amenity/shop tags).
# When the graph is a real GPS graph these are looked up live and cached;
# otherwise the simulation falls back to evenly spread anchor points.
OSM_USE_REAL_FACILITIES = True
OSM_FACILITY_TAGS = {
    "hospital":     ["amenity=hospital", "amenity=clinic"],
    "fire_station": ["amenity=fire_station"],
    "police":       ["amenity=police"],
    "shelter":      ["amenity=community_centre", "amenity=school",
                     "amenity=place_of_worship", "amenity=townhall"],
}
OSM_FACILITY_CACHE = OUTPUTS_DIR / "osm_facilities.json"
# Facility search scales with each area's own radius; this is only a ceiling.
# Too low and towns whose radius had to be auto-expanded report zero real
# facilities and silently fall back to synthetic anchor points.
OSM_FACILITY_SEARCH_MAX_M = 8000

# ---------------------------------------------------------------------------
# Hazard geometry
# ---------------------------------------------------------------------------
# For a REAL (GPS) graph, hazards are sized in METRES so the numbers mean
# something physical ("a 400 m blast radius") and are independent of how much
# of the city happened to get downloaded.  For the relative/pixel demo graph
# the fractions of the graph diagonal are used instead.
USE_METRIC_HAZARDS = True
HAZARD_EARTHQUAKE_RADIUS_M = 400.0    # metres -- epicentre disc
HAZARD_FLOOD_HALF_WIDTH_M = 250.0     # metres -- half-width of the flood corridor
HAZARD_FLOOD_ANGLE_DEG = 35.0         # compass direction of the river band
HAZARD_BLACKOUT_NODES = 3             # gatekeeper junctions simply closed
# Landslide: a narrower, steeper corridor than the flood band, run along a
# different compass bearing so the two hazards cut across the town differently.
HAZARD_LANDSLIDE_HALF_WIDTH_M = 150.0
HAZARD_LANDSLIDE_ANGLE_DEG = 115.0

# Fraction-of-diagonal fallbacks, used only when the graph is NOT a GPS graph.
HAZARD_EARTHQUAKE_FRACTION = 0.22
HAZARD_FLOOD_HALF_WIDTH_FRACTION = 0.16
HAZARD_LANDSLIDE_HALF_WIDTH_FRACTION = 0.10

# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Research extensions -- independent toggles (Parts C & D)
# ---------------------------------------------------------------------------
# Segmentation model used for inference / mask export:
#   "baseline"  -> smp U-Net + ResNet34 (Part A), checkpoint outputs/best_model.pth
#   "attention" -> Occlusion-Aware Attention U-Net (Part C), outputs/best_attention_model.pth
MODEL_CHOICE = "baseline"

# Graph healing scorer used by the graph stage:
#   "classical" -> distance + angle formula, configured by DISTANCE_WEIGHT/ANGLE_WEIGHT
#   "gnn"       -> gained link-prediction scorer (Part D); classical stays the default
HEALING_METHOD = "classical"

# --- Part C: Occlusion-Aware Attention model ---------------------------------
ATTENTION_BOTTLENECK_HIDDEN = 128   # projected Q/K/V channels at the bottleneck self-attention
ATTENTION_SE_REDUCTION = 8          # squeeze-and-excitation reduction ratio in the decoder
ATTENTION_DECODER_BASE = 64         # base channel width of the first decoder block
# Occlusion-conditioned loss: occ-masked pixels get an extra (1 + OCC_MULT * occ) weight
OCCLOSS_MIX = 1.0                   # scalar weighting whole occlusion-weighted BCE term
OCCLOSS_MULT = 5.0                  # additional weight multiplier inside occluded regions
EPOCHS_ATTENTION = 10               # epochs for the Part C model (same split as baseline)

# --- Part D: GNN link-prediction healing -------------------------------------
GNN_HIDDEN = 64
GNN_LAYERS = 2
GNN_EPOCHS = 60
GNN_LR = 1e-3
GNN_EDGE_DROP = 0.20                # fraction of edges removed to synth. gaps in training data
GNN_NEG_PER_POS = 2                 # negative pairs sampled per positive (balanced)
GNN_TRAIN_CORRUPTIONS = 60          # corrupted-graph variants generated per epoch
GNN_PAIR_LIMIT = 4000               # cap on pairs per corrupted graph (memory guard)
GNN_EVAL_CORRUPTIONS = 12           # hold-out corrupted graphs used in the benchmark table