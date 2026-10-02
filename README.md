# Road Network Reconstruction & Resilience Analysis MVP

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Repo](https://img.shields.io/badge/repo-Anusuya2103/route_resilience-24292f.svg)](https://github.com/Anusuya2103/route_resilience)

**Source code:** <https://github.com/Anusuya2103/route_resilience>

End-to-end pipeline for **road extraction from (potentially occluded) satellite
images** followed by **road-network graph construction, topological healing, and
resilience analysis** under node failures, with an interactive Streamlit +
Folium dashboard and a fully offline single-file web console.

```text
SATELLITE IMAGES
      |
      v
PART A - ROAD EXTRACTION (U-Net/ResNet34, occlusion-robust)
      |
      v                    binary road masks
      |
      v
skeletonization + graph construction  ->  topological healing
      |
      v                  nodes.json + edges.json + graph.graphml
      |
      v
PART B - NETWORK ANALYSIS (betweenness centrality + node-failure ablation)
      |
      v                        results.json
      |
      v
PART C - INTERACTIVE DASHBOARD (Folium + Streamlit)
```

> **Demo-first:** the entire analysis + dashboard half works immediately without
> any satellite dataset or trained model via `python main.py --stage demo`
> (OSM graph -> analysis -> results -> dashboard).

---

## 1. Project Objective

Reconstruct a road network from satellite imagery and quantify how resilient
that network is to individual junction/node failures. The system handles
partial occlusions (trees, shadows, clouds, dark objects) at the segmentation
level, heals small topological gaps in the extracted graph, ranks junctions by
betweenness centrality, and simulates node removal to compute a
**Resilience Index** for every critical junction.

---

## 1a. Repository contents (source code only)

This repository is **source code only**. Datasets, model checkpoints, build
artifacts and screenshots are all reproducible from the code, so none of them
are committed — that keeps the clone small and avoids redistributing
third-party satellite imagery.

**Committed (36 files):**

| Path | Contents |
|------|----------|
| `main.py`, `config.py` | CLI orchestrator and all tunables |
| `src/` | extraction, graph building, healing, analysis, simulation, assistant |
| `dashboard/` | Streamlit + Folium dashboard |
| `web/` | `index.html`, `app.js`, `app2.js` — the offline web console |
| `scripts/` | pipeline stages, dataset/basemap fetchers, dashboard builders |
| `tests/` | Node harnesses for the web console |
| `requirements.txt`, `TRAIN_COMMANDS.txt`, `README.md` | docs & deps |

**Ignored (regenerate locally):**

| Path | How to regenerate |
|------|-------------------|
| `outputs/` | `python main.py --stage all` / the build scripts |
| `data/images`, `data/masks`, `data/patches`, `data/test_region` | `python scripts/download_dataset.py --dataset deepglobe` |
| `archive/` | DeepGlobe archives, same command |
| `outputs/basemap/` | `python scripts/fetch_basemap.py` (needs network once) |
| `*.pth`, `*.pt` | model checkpoints, `python scripts/run_part_*.py` |
| `shots/` | web-console screenshots |
| `road_resilience_code.zip` | `python scripts/make_code_zip.py` |

The pipeline stays runnable offline: if no real dataset is present,
`SYNTHETIC_FALLBACK` in `config.py` generates a small procedural road dataset,
and the web console falls back to flat backgrounds if no basemap manifest is
present.

---

## 2. Architecture

The pipeline is deliberately modular. The two halves communicate **only**
through the shared graph contract:

| Module | Responsibility | Spec section |
|--------|----------------|--------------|
| `main.py` | CLI orchestrator (`--stage all/demo/extraction/graph/analysis`) | F |
| `config.py` | every tunable constant | 3 |
| `src/extraction.py` | dataset, patches, occlusion augmentation, U-Net training, evaluation, mask export | A |
| `src/graph_builder.py` | stitching, skeletonization, graph extraction, topological healing | B |
| `src/graph_io.py` | `load_graph()` abstraction, shared-contract save/load, OSM placeholder | C |
| `src/analysis.py` | graph validation, centrality, node ablation, results.json | D |
| `src/gnn_healing.py` | learned graph healing via GCN link prediction | D |
| `src/simulation.py` | earthquake / flood / landslide / blackout hazard geometry | — |
| `src/emergency_assistant.py` | rule-based assistant over live simulation state | — |
| `dashboard/app.py` | Streamlit + Folium interactive dashboard | E |
| `scripts/build_web_dashboard.py` | builds the single-file offline web console | — |
| `scripts/fetch_basemap.py` | offline light/dark raster basemaps (with blank-image guard) | — |
| `scripts/download_dataset.py` | optional real-dataset downloader/patcher | A1 |
| `web/` | the offline console: `index.html`, `app2.js` | — |
| `tests/` | Node harnesses that exercise the console headlessly | — |

```
project/
|-- main.py
|-- config.py
|-- requirements.txt
|-- TRAIN_COMMANDS.txt
|-- data/                    # NOT committed - see section 1a
|   |-- images/              # real dataset images (optional)
|   |-- masks/               # real dataset ground-truth masks (optional)
|   `-- test_region/         # adjacent tiles used for mask export
|-- outputs/                 # NOT committed - all generated artifacts
|   |-- masks/               # mask_tile_XXX.png (0/255) + stitched_mask.png
|   |-- plots/               # training_curve.png, predictions.png, centrality_top10.png
|   |-- osm_cache/           # cached OSM demo graph
|   |-- basemap/             # offline light/dark basemap images + manifest.json
|   |-- emergency_dashboard.html   # the built single-file web console
|   |-- tiles_manifest.json
|   |-- graph.graphml
|   |-- nodes.json
|   |-- edges.json
|   `-- results.json
|-- src/
|   |-- extraction.py
|   |-- attention_model.py
|   |-- graph_builder.py
|   |-- graph_io.py
|   |-- gnn_healing.py
|   |-- analysis.py
|   |-- simulation.py
|   |-- emergency_assistant.py
|   `-- utils.py
|-- scripts/
|   |-- run_part_a.py ... run_part_d.py
|   |-- fetch_basemap.py
|   |-- build_web_dashboard.py
|   `-- download_dataset.py
|-- web/
|   |-- index.html
|   `-- app2.js              # console source inlined at build time
|-- tests/
|   |-- console_harness.js
|   `-- multi_area_smoke.js
`-- dashboard/
    `-- app.py
```

---

## 3. Installation

Python 3.10+ is recommended (tested on 3.11, Windows CPU).

```bash
pip install -r requirements.txt
```

The key dependencies are `torch`, `torchvision`, `segmentation-models-pytorch`,
`albumentations`, `opencv-python`, `scikit-image`, `networkx`, `osmnx`,
`streamlit`, `streamlit-folium`, `folium`, `numpy`, `matplotlib`, `pandas`,
`Pillow`.

GPU is used automatically when `torch.cuda.is_available()`; everything runs on
plain CPU otherwise.

---

## 4. Dataset Setup

The pipeline works out of the box with a built-in **procedurally generated**
synthetic road dataset (see `config.SYNTHETIC_FALLBACK`), which is ideal for
developing and smoke-testing every stage offline.

To train on real imagery, place files into:

```text
data/images/   <satellite images>
data/masks/    <ground-truth binary masks>
```

Both common layouts are auto-detected by `src/extraction.py`:

* **Massachusetts Roads** — `data/images/<name>.png` and `data/masks/<name>.png`
  (same stem).
* **DeepGlobe** — `data/images/<name>_sat.jpg` and `data/masks/<name>_mask.png`.

Raw images of any size are center-cropped and tiled into `256x256` patches
(only patches containing at least one road pixel are kept), then split
**70% train / 15% validation / 15% test** with a fixed seed. The split is
cached in `outputs/dataset_split.json`.

`MAX_TRAIN_SAMPLES = 200` caps the number of samples so training completes in
roughly 30 minutes on modest hardware.

```bash
# optional helper for obtaining a real dataset (prints manual instructions too)
python scripts/download_dataset.py --dataset massachusetts
python scripts/download_dataset.py --dataset deepglobe
```

If no dataset is present, the extraction stage prints:

```text
Dataset not found.
Please place the dataset in:
    data/images  (satellite images)
    data/masks   (ground-truth road masks)
...
```

and suggests `--use-synthetic` to train on the built-in synthetic dataset.

---

## 5. Training

```bash
python main.py --stage extraction
# smaller/faster smoke test on CPU:
python main.py --stage extraction --use-synthetic --samples 24 --epochs 1
```

The model is a **U-Net with a ResNet34 encoder**
(`segmentation_models_pytorch.Unet`, ImageNet weights when downloadable) trained
with **BCE + Dice** loss. Synthetic occlusion augmentation is applied to
**~35%** of training images (`config.OCCLUSION_PROB`):

* random dark rectangle,
* random strong-Gaussian-blur region,
* soft shadow-like diagonal gradient,

which mimics trees / shadows / clouds / dark objects hiding the roads. Masks are
never occluded, so the model learns to predict the road *under* the occlusion.
Imagenet normalization (`mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`) is
applied.

Per-epoch console output:

```text
Epoch 1/10
Train Loss: 0.621
Val Loss: 0.548
Val IoU: 0.61
```

The best epoch (highest validation IoU) is saved to `outputs/best_model.pth`;
training curves are saved to `outputs/plots/training_curve.png`.

---

## 6. Evaluation

```bash
python main.py --stage extraction   # includes evaluation + mask export
```

Prints on the held-out test split:

```text
========== TEST RESULTS ==========
IoU: 0.XX
Dice: 0.XX
===================================
```

**Occlusion-Recall** is defined in this project as the **IoU measured on
synthetically occluded versions of the test images** (every test image is
occluded deterministically). It is *not* conventional classification recall:

```text
Occlusion-Recall: 0.XX
```

`outputs/plots/predictions.png` shows 3-4 example rows of
Original | Ground Truth | Predicted Mask (the last row is an occluded input).

---

## 7. Graph Generation

```bash
python main.py --stage graph
```

Inputs: `outputs/masks/mask_tile_*.png` + `outputs/tiles_manifest.json`.
Outputs: `outputs/{nodes.json, edges.json, graph.graphml}` plus the
connectivity report.

Stage flow (`src/graph_builder.py`):

1. **Stitch** -- tile masks are assembled into one large mask in row-major grid
   order (no GIS libraries needed).
2. **Skeletonize** -- `skimage.morphology.skeletonize` produces one-pixel-wide
   centerlines.
3. **Extract graph** -- skeleton pixels are classified by their 8-connected
   neighbor count: `1` = endpoint, `2` = path pixel, `3+` = intersection.
   Endpoints/junctions become nodes; chains between nodes become edges with
   `weight` = path length in **pixels**.
4. **Remove noise** -- connected components smaller than
   `MIN_COMPONENT_SIZE = 20` pixels are dropped first.
5. **Heal** -- see next section.
6. **Export** -- shared contract files are written by `src/graph_io.py`.

When no masks exist yet (no trained model), the graph stage generates a labelled
DEMO mask set (synthetic roads + artificial gaps) so the whole stage can be
exercised standalone.

---

## 8. Topological Healing

Disconnected components whose endpoints lie within `MAX_GAP_DISTANCE = 30`
pixels of each other are candidates for bridging. Every candidate pair is
scored (lower is better):

```text
cost = DISTANCE_WEIGHT * (distance / MAX_GAP_DISTANCE)
     + ANGLE_WEIGHT   * normalized_angle
```

`normalized_angle` prefers bridges that continue each road's local direction
(angle between the road direction at the endpoint and the straight line to the
other endpoint, normalized to [0,1]). Defaults:
`DISTANCE_WEIGHT = 0.7`, `ANGLE_WEIGHT = 0.3`.

Bridges are added with a **Union-Find (MST-style)** procedure: candidates are
sorted by cost and a bridge is added **only** when it joins two currently
separate components. This connects components without over-connecting every
nearby endpoint. Components that are too far apart are left alone and reported
(no unrealistic long edges).

Connectivity report:

```text
========== GRAPH CONNECTIVITY ==========
Components Before: 3
Components After: 1
Largest Component Before: 39 nodes
Largest Component After: 67 nodes
Connectivity Ratio: 171.79%
Bridges Added: 6
...
=========================================
```
---

## 9. OSM Demo Mode

The **demo-first** requirement means the analysis + dashboard half works even
before the segmentation model is trained:

```bash
python main.py --stage demo
```

`src/graph_io.load_osm_placeholder()` downloads a small, real road network
around `config.OSM_CENTER = (12.9716, 80.2209)` (Chennai, ~700 m radius, drive
network) with `osmnx.graph_from_point()`, converts it to the shared contract
(sequential integer node IDs, edge `weight` = OSM `length` in meters) and caches
everything under `outputs/osm_cache/`. If a network connection is unavailable
the demo falls back to a clearly-labelled built-in grid graph (relative pixel
coordinates) so it never crashes offline.

Set `GRAPH_SOURCE` in `config.py` to choose where every analysis/dashboard run
reads the graph from:

```python
GRAPH_SOURCE = "osm"               # demo / development (cached OSM or grid fallback)
GRAPH_SOURCE = "teammate_export"   # nodes.json + edges.json (extraction half)
GRAPH_SOURCE = "graphml"           # graph.graphml
```

The same switch works per-invocation:

```bash
python main.py --stage analysis --source teammate_export
python main.py --stage analysis --source graphml
```

---

## 10. Network Analysis

```bash
python main.py --stage analysis
```

Steps in `src/analysis.py`:

1. **Validate** -- nodes exist, edges reference real nodes (invalid edges are
   reported), weights are numeric (fallback `weight = 1.0`, printed), graph is
   non-empty and has >= 1 connected component. Errors are graceful, never crashes.
2. **Centrality** -- `networkx.betweenness_centrality(G, weight="weight")` for
   all nodes; the TOP_N = 10 highest become **gatekeeper candidates**.
3. **Baseline** -- average shortest path length (weighted) of the **largest
   connected component** (disconnected graphs handled automatically).
4. **Node ablation** -- for each top-N node the graph is copied with the node
   removed; the average shortest path length of the largest remaining component
   is recomputed and

   ```text
   Resilience Index = baseline / perturbed
   Path Length Change (%) = (perturbed - baseline) / baseline * 100
   ```

   Empty/trivial results are stored as `null` with an explanatory `note`.

Results go to `outputs/results.json` (schema in Section 12) and the top-10
centrality chart to `outputs/plots/centrality_top10.png`.

---

## 11. Run the Streamlit Dashboard

```bash
streamlit run dashboard/app.py
```

Sidebar -- **Road Network Resilience**:

* Graph source: `osm` / `teammate_export` / `graphml` (defaults to config.GRAPH_SOURCE)
* Top-N slider
* Selected-node dropdown (guaranteed interaction path)
* Reset Simulation button (returns to the baseline view)
* Top-N gatekeeper node IDs listed for reference

The folium map draws each edge as a polyline and each node colored by normalized
betweenness centrality with a continuous **green -> yellow -> red** scale and a
legend ("visualization only, not a ranking"). Clicking a marker (or selecting in
the sidebar) shows:

```text
Selected Node: XXXX
Baseline Avg Path Length: XXX
After Removal: XXX
Resilience Index: X.XX
Path Length Change: XX%
```

and the failure view (removed node marked, incident edges omitted). Full
route-by-route rerouting is a documented future enhancement.
---

## 12. Shared Graph Contract

The ONLY interface between the extraction and analysis halves:

`outputs/nodes.json`:

```json
[ { "id": 1, "lat": 0.421, "lon": 0.732 } ]
```

`outputs/edges.json`:

```json
[ { "source": 1, "target": 2, "weight": 84.5 } ]
```

`outputs/graph.graphml` -- the same graph via `networkx.write_graphml()`.

* Node IDs are **integers**, consistent across all three files. (GraphML stores
  XML string IDs by spec; `src/graph_io.load_graphml()` casts them back to int.)
* `weight` = road-segment length: **pixels** for the mask/skeleton path, OSM
  `length` in **meters** for the OSM demo path. The unit is always documented in
  the console output. `config.METERS_PER_PIXEL` converts pixels to meters when
  a calibration is known.
* Missing weights fall back to `1.0` with a printed note.

---

## 13. Coordinate Assumptions

* **Extraction pipeline**: no georeferencing in the MVP, so pixel coordinates
  are normalized to relative [0,1] coordinates --

  ```text
  lat = 1 - y / height     (north-up)
  lon = x / width
  ```

  These are stored in the `lat`/`lon` JSON fields but are explicitly **relative
  image coordinates, NOT GPS** (printed at save time and annotated in the
  dashboard).
* **OSM demo**: real latitude/longitude from OpenStreetMap.
* The dashboard auto-detects [0,1] relative coordinates and maps them into a
  small labelled demo window around the demo center rather than pretending they
  are real geography.

---

## 14. Known Limitations

* Baseline U-Net only -- no transformer/attention heads, connectivity losses or
  in-model graph healing (a deliberate MVP trade-off).
* The synthetic dataset is a proxy for real satellite imagery; real
  Massachusetts Roads / DeepGlobe data improves realism (Section 4).
* No georeferencing: mask stitching uses row-major tile order, relative
  coordinates, and pixel units.
* Topological healing is geometric (distance + direction) -- it does not use
  semantic/surface cues to decide where broken roads continue.
* Analysis uses betweenness centrality and single-node ablation on the largest
  connected component; multi-failure cascades are out of scope.
* ImageNet encoder weights are downloaded on first use when the network allows;
  they fall back to random initialization otherwise (train longer in that case).
* Full CPU training (200 samples x 10 epochs) takes ~20-45 minutes on modest
  hardware; use `--samples`/`--epochs` for faster experiments.

---

## 15. Future Improvements

* Connectivity-aware / topology-preserving segmentation losses and stronger
  encoders.
* Georeferenced stitching (tile bounds), a meters-per-pixel calibration, and
  real GPS output from the extraction half.
* Dynamic route-by-route rerouting, alternate-path and disconnected-component
  overlays in the dashboard.
* Cascading-failure and disruption-recovery simulations, budgeted hardening.
* Fine-tuning and evaluation on Massachusetts Roads / DeepGlobe.

---

## 16. Research Extensions (Parts C & D) — the Extraction Track's novelties

Parts A and B above are the **baselines**. Two additional, independently
runnable extensions compare learned alternatives against them. Each part has its
own script; each can be run on its own:

```bash
python scripts/run_part_a.py        # baseline segmentation  (train/eval/export masks)
python scripts/run_part_b.py        # skeleton + classical healing -> graph contract
python scripts/run_part_c.py        # attention model, same split, comparison table
python scripts/run_part_d.py        # GNN link-prediction healing, benchmark table
```

Both novelties are **toggles** so the final export can mix and match:

```python
# config.py
MODEL_CHOICE  = "baseline"   # "baseline" (Part A)  | "attention" (Part C)
HEALING_METHOD = "classical" # "classical" (Part B) | "gnn" (Part D)
```

or per-run:

```bash
python main.py --stage extraction --model attention    # export masks from the attention model
python scripts/run_part_b.py --method gnn               # heal with the learned scorer
```

### Part C — Occlusion-Aware Attention model (`src/attention_model.py`)

* **Same U-Net + ResNet34 backbone as Part A** (so the comparison isolates the
  attention machinery, not the encoder).
* **Spatial self-attention at the bottleneck**: an occluded road can only be
  reconstructed by referencing *distant visible* road pixels, so a non-local
  attention block sits between encoder and decoder, giving every position a
  whole-tile receptive field exactly where that cross-reference is cheapest.
* **Channel attention (squeeze-and-excitation) in every decoder block**: after
  up-sampling+skip-concat the channels carry both road-continuity evidence and
  occlusion texture; SE re-weights them where they compete.
* **Occlusion-conditioned loss**: the per-pixel BCE is weighted by
  `1 + OCCLOSS_MULT` inside the occlusion maps produced in Part A step 2, so
  "be robust exactly where occlusion happened" is an explicit training signal.

`run_part_c.py` trains on the **same split** as the baseline, reports the same
three metrics, and prints the deliverable table:

| Model | IoU | Dice | Occlusion-Recall |
|---|---|---|---|
| U-Net baseline (Part A) | | | |
| Attention model (Part C) | | | |

It also prints a **novelty check**: the claim only holds if Occlusion-Recall
improves *more* than overall IoU (evidence the attention targets occlusion
specifically, not segmentation in general). This is reported honestly either way.
Results are saved to `outputs/partC_comparison.json` and
`outputs/plots/partC_comparison.png`.

### Part D — Learned graph healing (GNN link prediction) (`src/gnn_healing.py`)

* **Framing**: every disconnected endpoint pair is a candidate edge; the task is
  link prediction ("should this edge exist?").
* **Training data**: ground-truth OSM road graphs (reused from the OSM cache) plus
  deterministic synthetic grids. Random edges are removed to simulate occlusion
  gaps (the graph-level analogue of Part A's pixel occlusion). Positives = the
  removed (should-reconnect) edges; negatives = random non-edges.
* **Node features**: `(x,y)` position, degree, distance to nearest endpoint, and
  the local direction vector of the node's existing edge (Part B's angular
  idea, but the GNN learns the weighting).
* **Model**: a 2-layer **GCN** (PyTorch Geometric) encoder + MLP head that outputs
  a should-connect probability per candidate pair.
* **Minimal-risk integration**: the final graph is still built by the same
  Union-Find procedure; **only the scoring function is swapped**
  (`heal_graph(G, scorer=GNNScorer)`). Both scorers rank the identical candidate
  set (same `max_gap`), so the comparison is apples-to-apples.

`run_part_d.py` trains the GNN, heals held-out corrupted graphs with both
scorers, and prints the deliverable table:

| Method | Connectivity Ratio | Bridges Added | False Bridges | Runtime |
|---|---|---|---|---|
| Classical (Part B: MST + distance/angle) | | | | fast |
| Learned (Part D: GNN link prediction) | | | | slower |

"False bridges" are automated against the known removed-edge ground truth (a
reproducible stand-in for manual visual inspection). The conclusion is printed
plainly from the data — if classical is as good and much faster it stays the
default, and the GNN is documented as a validated alternative. Results go to
`outputs/partD_comparison.json` and `outputs/plots/partD_comparison.png`.

---

## 17. Offline Web Console

`scripts/build_web_dashboard.py` produces **`outputs/emergency_dashboard.html`**:
a single, self-contained 7-page emergency-operations console. Everything —
graph data, basemap imagery, CSS and JS — is inlined, so the file works from
`file://` with no server and no network.

```bash
# 1. fetch offline basemaps (only needs network once; cached under outputs/basemap)
python scripts/fetch_basemap.py

# 2. build the single-file console
python scripts/build_web_dashboard.py --source osm_cache
```

| Page | Contents |
|------|----------|
| 01 Operations Overview | live status, scenario controls, monitored-areas table |
| 02 Disaster Simulation | earthquake / flood / landslide / blackout, animated |
| 03 Road Reconstruction | stitch → skeleton → heal, step by step |
| 04 Network Analytics | betweenness, articulation points, component analysis |
| 05 Facility & Route Impact | accessibility and alternate routes |
| 06 Recovery Planning | crew allocation, staged restoration |
| 07 Model & Research Results | segmentation + healing benchmark tables |

**Basemaps and the style toggle.** Areas are drawn over pre-stitched offline
raster basemaps in two styles (Esri dark canvas and light gray base + reference
labels). The basemap is placed with the *same exact graph affine transform* used
for the overlay, so the network always sits on the right streets — no separate
cover scaling. A single **Map** button per panel flips the style, and flipping
it re-renders every map on every page. If the manifest is missing or invalid the
console falls back to flat backgrounds.

Because OSM's public bulk tile service returns HTTP 403 placeholder tiles that
*look* like valid imagery, `fetch_basemap.py` validates every stitched image
with `check_stitched()` and refuses to write suspiciously blank/white results.

**Tests.** The console has two Node harnesses that run the real page under a
minimal DOM/canvas stub — no browser needed:

```bash
node --check web/app2.js     # syntax
node tests/console_harness.js        # page switches, hazards, assistant, recovery
python scripts/make_smoke_fixture.py  # build outputs/_smoke.html (multi-area)
node tests/multi_area_smoke.js        # per-area hazards + assistant answers
```

---

## 18. Quick Reference

```bash
pip install -r requirements.txt

python main.py --stage all          # full pipeline; DEMO_MODE skips training, prints FINAL RESULTS
python main.py --stage demo         # OSM -> graph -> centrality -> ablation -> results.json
python main.py --stage extraction   # train / evaluate / export masks (add --samples/--epochs)
python main.py --stage graph        # masks -> stitch -> skeleton -> heal -> graph files
python main.py --stage analysis     # centrality -> ablation -> results.json + plots
streamlit run dashboard/app.py      # interactive dashboard

# Extraction-Track parts (independently runnable)
python scripts/run_part_a.py        # Part A baseline segmentation
python scripts/run_part_b.py        # Part B skeleton + classical healing
python scripts/run_part_c.py        # Part C attention model + comparison table
python scripts/run_part_d.py        # Part D GNN healing + benchmark table

# real dataset helper (optional)
python scripts/download_dataset.py --dataset massachusetts
python scripts/download_dataset.py --dataset deepglobe

# offline web console (see section 17)
python scripts/fetch_basemap.py               # once; needs network
python scripts/build_web_dashboard.py --source osm_cache

# web console tests
node --check web/app2.js
node tests/console_harness.js
node tests/multi_area_smoke.js
```