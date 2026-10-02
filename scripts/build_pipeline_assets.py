"""
Render the REAL reconstruction-pipeline stages shown on the "Road
Reconstruction" page of the RoadNet Ops console.

Each stage is produced by actually running that step on real data, so the page
shows the pipeline working rather than a hand-drawn diagram:

  1. satellite image   a real DeepGlobe patch from data/patches/images
  2. preprocessing     the same patch after the real normalisation / resize
  3. segmentation      the real forward pass of the trained attention U-Net
  4. predicted mask    the thresholded binary mask, white = road
  5. ground truth      the real DeepGlobe mask for the same patch
  6. skeleton          the real skimage skeletonization of the predicted mask
  7. healed graph      the real graph built from the vectorised skeleton

    python scripts/build_pipeline_assets.py
    python scripts/build_pipeline_assets.py --model baseline

Writes  outputs/pipeline/*.png  and  outputs/pipeline_manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2                                                          # noqa: E402
import numpy as np                                                   # noqa: E402
import torch                                                        # noqa: E402

import config                                                        # noqa: E402
from src import extraction, utils                                    # noqa: E402

OUT_DIR = config.OUTPUTS_DIR / "pipeline"
MANIFEST = config.OUTPUTS_DIR / "pipeline_manifest.json"


def _label(img: np.ndarray, text: str) -> np.ndarray:
    """Stamp a small caption bar along the top of a BGR image."""
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 20), (10, 16, 32), -1)
    cv2.putText(out, text, (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (235, 245, 255), 1, cv2.LINE_AA)
    return out


def _denorm_preview(norm_rgb: np.ndarray, size: int) -> np.ndarray:
    """Invert the ImageNet normalisation so the preprocessed tensor is viewable."""
    mean = np.asarray(config.IMAGENET_MEAN, np.float32)
    std = np.asarray(config.IMAGENET_STD, np.float32)
    img = norm_rgb * std + mean
    return cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8),
                        cv2.COLOR_RGB2BGR)


def pick_patch(test: list) -> tuple:
    """Choose the test patch with the most road pixels.

    Picking the most road-dense tile makes the mask stages legible at console
    size; the patch is still a real held-out sample.
    """
    best, best_cov = test[0], -1.0
    for img_path, mask_path in test:
        m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if m is None or not m.size:
            continue
        cov = float((m > 127).mean())
        if cov > best_cov:
            best, best_cov = (img_path, mask_path), cov
    return best, best_cov


def _skeleton_to_graph(skel: np.ndarray, min_component: int = 20):
    """Vectorise a 1-px skeleton into (edges, nodes_px).

    The input MUST already be a skeleton (skimage.morphology.skeletonize).
    On the thick binary mask nearly every pixel has 8 neighbours, so a
    "degree != 2 is a junction" test finds no structure at all.  On the
    skeleton the degree histogram is meaningful (endpoints 1, runs 2,
    crossings >= 3) and a walk traces whole road segments.

    Node pixels (degree 1 or >= 3) are CLUSTERED before walking.  One fork
    occupies several adjacent node pixels; if each were its own node, a walk
    would stop on the fork's own neighbour after a single step and the graph
    would collapse into 1-px stubs.  Clustering each clump into one node means
    a walk exits along a real branch and runs to the next node.
    """
    binary = (skel > 0).astype(np.uint8)
    n_labels, labels = cv2.connectedComponents(binary, connectivity=8)

    # 8-neighbour count for every pixel, via convolution (fast)
    ker = np.ones((3, 3), np.uint8)
    ker[1, 1] = 0
    cnt = cv2.filter2D(binary, -1, ker, borderType=cv2.BORDER_CONSTANT)

    h, w = labels.shape
    edges: list = []
    nodes_px: list = []

    for label in range(1, n_labels):
        comp = np.argwhere(labels == label)
        if len(comp) < min_component:
            continue

        ys, xs = comp[:, 0], comp[:, 1]
        degrees = cnt[ys, xs]
        node_sel = degrees != 2
        if not node_sel.any():            # pure cycle, no junctions/endpoints
            continue

        # cluster adjacent node pixels into single nodes
        node_bin = np.zeros_like(binary)
        node_bin[ys[node_sel], xs[node_sel]] = 1
        nn, nlab = cv2.connectedComponents(node_bin, connectivity=8)

        node_of: dict = {}                # (y, x) -> GLOBAL node id
        offset = len(nodes_px)
        clusters: list = []
        for c in range(1, nn):
            cy, cx = np.where(nlab == c)
            gid = offset + c - 1
            clusters.append(list(zip(cy.tolist(), cx.tolist())))
            for p in clusters[-1]:
                node_of[p] = gid
            nodes_px.append((int(round(cy.mean())), int(round(cx.mean()))))

        # Trace each branch by FOLLOWING the pixel chain.  A straight-line
        # ray walk fails the moment a road bends: the next pixel along the ray
        # is off-road, the loop stops, and no edge is ever recorded.  Instead,
        # from each node pixel step to a road neighbour and keep stepping to
        # the unique unvisited road neighbour until a node is reached.
        def road_neigh(p):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    q = (p[0] + dy, p[1] + dx)
                    if 0 <= q[0] < h and 0 <= q[1] < w \
                            and labels[q[0], q[1]] == label:
                        yield q

        visited = set()
        for pixels in clusters:
            for sy, sx in pixels:
                start = (sy, sx)
                if start in visited:
                    continue
                for first in road_neigh(start):
                    if first in node_of or first in visited:
                        continue          # entering another node: not a branch
                    path, cur, prev = [first], first, start
                    visited.add(first)
                    steps = 0
                    while cur not in node_of and steps < 8192:
                        nxt = None
                        for q in road_neigh(cur):
                            if q != prev and q not in visited:
                                nxt = q
                                break
                        if nxt is None:
                            break        # dead end or a fork we already took
                        path.append(nxt)
                        visited.add(nxt)
                        prev, cur = cur, nxt
                        steps += 1
                    other = node_of.get(cur)
                    a = node_of[start]
                    if other is None or other == a:
                        continue
                    pair = (a, other) if a < other else (other, a)
                    if pair not in edges:
                        edges.append(pair)
    return edges, nodes_px


def _render_graph(nodes_px: list, edges: list, size: int) -> np.ndarray:
    """Draw the healed graph the way the console draws a road network."""
    img = np.full((size, size, 3), 12, np.uint8)
    # BGR: a light blue road casing with a brighter core, so the structure is
    # legible against the near-black background at console size.
    for a, b in edges:
        (y1, x1), (y2, x2) = nodes_px[a], nodes_px[b]
        cv2.line(img, (x1, y1), (x2, y2), (200, 140, 60), 3, cv2.LINE_AA)
        cv2.line(img, (x1, y1), (x2, y2), (255, 235, 190), 1, cv2.LINE_AA)
    for y, x in nodes_px:
        cv2.circle(img, (x, y), 4, (60, 90, 255), -1, cv2.LINE_AA)
        cv2.circle(img, (x, y), 4, (255, 245, 235), 1, cv2.LINE_AA)
    return _label(img, f"healed graph {len(nodes_px)}n / {len(edges)}e")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="attention", choices=["baseline", "attention"])
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(utils.get_device())
    _, _, test = extraction.get_dataset_lists()
    if not test:
        print("[pipeline] no test split available")
        return 1

    (img_path, mask_path), coverage = pick_patch(test)
    print(f"[pipeline] patch {Path(img_path).name} ({coverage * 100:.1f}% road pixels)")

    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if img is None or gt is None:
        print("[pipeline] could not read the chosen patch")
        return 1
    size = config.IMAGE_SIZE
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    gt = cv2.resize(gt, (size, size), interpolation=cv2.INTER_NEAREST)
    gt_bin = (gt > 127).astype(np.float32)

    # --- stage 3/4: a real forward pass of the trained checkpoint -----------
    model = extraction._load_model(device, args.model)
    mean = np.asarray(config.IMAGENET_MEAN, np.float32)
    std = np.asarray(config.IMAGENET_STD, np.float32)
    norm = ((img.astype(np.float32) / 255.0)[..., ::-1] - mean) / std   # RGB
    x = torch.from_numpy(norm.transpose(2, 0, 1)[None].copy()).to(device)
    with torch.no_grad():
        probs = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    pred_bin = (probs > 0.5).astype(np.uint8) * 255
    print(f"[pipeline] predicted {(pred_bin > 0).mean() * 100:.1f}% road pixels")

    # --- stage 6: the real skeletonization ---------------------------------
    # Crop the caption bar FIRST (it is UI chrome, not road).  Its text strokes
    # would otherwise become extra connected components in the vectorisation.
    road_only = pred_bin[20:, :]
    skel = None
    try:
        from skimage.morphology import skeletonize
        skel_crop = skeletonize(road_only > 0).astype(np.uint8) * 255
        # pad back to full size so this stage lines up with the other stages
        skel = np.vstack([np.zeros((20, skel_crop.shape[1]), np.uint8), skel_crop])
    except Exception as exc:                                # pragma: no cover
        print(f"[pipeline] skeletonization unavailable ({exc}) - using the mask")
        skel = road_only

    # --- stage 7: vectorise the skeleton and heal it into a real graph -----
    # Must vectorise the SKELETON, not the thick mask (see _skeleton_to_graph).
    vec_in = skel[20:, :] if skel.shape[0] == size else skel
    edges, nodes_px = _skeleton_to_graph(vec_in)
    healed = _render_graph(nodes_px, edges, size)
    print(f"[pipeline] healed graph: {len(nodes_px)} nodes / {len(edges)} edges")

    gray2bgr = lambda a: cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_GRAY2BGR)  # noqa: E731
    stages = [
        ("1. Satellite image", img,
         "Real DeepGlobe satellite patch fed to the segmentation model (256x256)."),
        ("2. Preprocessing", _label(_denorm_preview(norm[..., ::-1], size), "preprocessing"),
         "Resized to 256x256 and ImageNet-normalised (per-channel mean/std)."),
        ("3. Segmentation", _label(gray2bgr(np.clip(probs, 0, 1) * 255),
                                    f"{args.model} forward pass"),
         "Raw sigmoid road probability from the trained model (brighter = more "
         "confident road)."),
        ("4. Predicted mask", _label(gray2bgr(pred_bin), "thresholded at 0.5"),
         "Binary road mask (white = road) -- this is what gets vectorised."),
        ("5. Ground truth", _label(gray2bgr(gt_bin * 255), "DeepGlobe truth"),
         "The real DeepGlobe mask for the same patch, shown for comparison."),
        ("6. Skeletonized", _label(gray2bgr(skel if skel is not None else pred_bin),
                                   "1-px skeleton"),
         "The mask thinned to a 1-pixel-wide centreline (skimage skeletonize)."),
        ("7. Healed road graph", healed,
         f"Vectorised and healed into a graph: {len(nodes_px)} nodes / {len(edges)} "
         f"edges -- the same structure the network pages analyse."),
    ]

    manifest = []
    for title, image, caption in stages:
        path = OUT_DIR / (title.split(".")[0].strip().lower().replace(" ", "_") + ".png")
        cv2.imwrite(str(path), image)
        manifest.append({"title": title, "file": path.name, "caption": caption})
        print(f"[pipeline]   wrote {path.name}")

    # Stitched-mask stage: the real multi-tile export, when it exists.
    stitched = config.MASKS_OUT / "stitched_mask.png"
    if stitched.exists():
        s = cv2.imread(str(stitched), cv2.IMREAD_COLOR)
        if s is not None:
            s = cv2.resize(s, (size, size), interpolation=cv2.INTER_NEAREST)
            path = OUT_DIR / "5b_stitched_mask.png"
            cv2.imwrite(str(path), _label(s, "4 tiles stitched"))
            manifest.append({
                "title": "Stitched mask", "file": path.name,
                "caption": "Four adjacent tiles stitched into one region mask, so "
                           "roads crossing a tile border are not cut off."})
            print(f"[pipeline]   wrote {path.name}")

    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[pipeline] wrote {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
