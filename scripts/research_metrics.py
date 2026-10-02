"""
Compute the REAL model / dataset numbers shown on the "Model & Research Results"
page of the RoadNet Ops console.

The existing `outputs/segmentation_metrics*.json` files only carry IoU, Dice and
occlusion-recall.  The console also has to show precision, recall, F1, mean
prediction confidence, per-image spread and inference time, so this script runs
both trained checkpoints over the held-out test split and writes them out.

Nothing here is invented: every number is measured on the real DeepGlobe
patches in `data/patches` using the real checkpoints in `outputs/`.

    python scripts/research_metrics.py            # both models, CPU
    python scripts/research_metrics.py --models attention

Writes  outputs/research_metrics.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np                                                   # noqa: E402
import torch                                                        # noqa: E402

import config                                                        # noqa: E402
from src import extraction, utils                                    # noqa: E402

OUT_JSON = config.OUTPUTS_DIR / "research_metrics.json"

MODEL_INFO = {
    "baseline": {"label": "Part A - U-Net + ResNet34 (baseline)",
                 "ckpt": config.BEST_MODEL,
                 "role": "Baseline segmenter trained with BCE + Dice loss."},
    "attention": {"label": "Part C - Occlusion-attention U-Net",
                  "ckpt": config.ATTENTION_MODEL,
                  "role": "Adds spatial + channel attention and an "
                          "occlusion-conditioned loss."},
}


def _confusion(pred: torch.Tensor, target: torch.Tensor, thr: float = 0.5):
    """Pixel counts for TP/FP/FN/TN at a 0.5 threshold."""
    p = (pred > thr).float()
    t = (target > 0.5).float()
    tp = float((p * t).sum())
    fp = float((p * (1 - t)).sum())
    fn = float(((1 - p) * t).sum())
    tn = float(((1 - p) * (1 - t)).sum())
    return tp, fp, fn, tn


def _rates(tp, fp, fn, tn) -> dict:
    """Precision / recall / F1 / IoU / Dice / accuracy from pixel counts."""
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    acc = (tp + tn) / max(tp + fp + fn + tn, 1.0)
    return {"precision": prec, "recall": rec, "f1": f1, "iou": iou,
            "dice": dice, "accuracy": acc}


def evaluate_model(model, loader, device) -> dict:
    """Aggregate + per-image metrics over one dataloader.

    `mean_confidence` is the mean sigmoid probability the model assigns to the
    pixels it calls road -- a calibration-flavoured number that pairs naturally
    with precision/recall in the results table.
    """
    totals = np.zeros(4, dtype=np.float64)
    per_image, road_conf, times = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            t0 = time.perf_counter()
            logits = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            times.append(dt / max(x.shape[0], 1))
            probs = torch.sigmoid(logits)

            for i in range(x.shape[0]):
                c = _confusion(probs[i, 0], y[i, 0])
                totals += np.asarray(c, dtype=np.float64)
                per_image.append(_rates(*c))
                p = probs[i, 0]
                road_conf.append(float(p[p > 0.5].mean()) if bool((p > 0.5).any()) else 0.0)

    agg = _rates(*[float(v) for v in totals])
    agg["mean_confidence"] = float(np.mean(road_conf)) if road_conf else 0.0
    # spread across images -> shows the model is not carried by a few tiles
    agg["iou_std"] = float(statistics.pstdev([p["iou"] for p in per_image])) \
        if len(per_image) > 1 else 0.0
    agg["iou_min"] = float(min((p["iou"] for p in per_image), default=0.0))
    agg["iou_max"] = float(max((p["iou"] for p in per_image), default=0.0))
    agg["inference_ms_per_image"] = float(np.mean(times)) * 1000.0 if times else 0.0
    agg["n_images"] = len(per_image)
    return agg


def dataset_stats() -> dict:
    """Real counts of what the pipeline trained and tested on."""
    import cv2

    split_path = config.SPLIT_JSON
    split = json.loads(split_path.read_text()) if split_path.exists() else {}

    def _count_dir(d: Path, exts) -> int:
        if not d.is_dir():
            return 0
        return sum(1 for p in d.iterdir() if p.suffix.lower() in exts)

    img_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    # Average road coverage of the real patches -- explains why the absolute
    # IoU sits near 0.4 even for a well-trained model: roads are a small
    # minority of the pixels, and empty space counts as a correct negative.
    coverages = []
    patch_dir = config.PATCHES_DIR / "masks"
    if patch_dir.is_dir():
        for p in sorted(patch_dir.iterdir())[:60]:
            m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if m is not None and m.size:
                coverages.append(float((m > 127).mean()))

    return {
        "deepglobe_images": _count_dir(config.IMAGES_DIR, img_exts),
        "deepglobe_masks": _count_dir(config.MASKS_DIR, {".png"}),
        "patches": int(split.get("samples", 0) or _count_dir(patch_dir, {".png"})),
        "n_train": int(split.get("n_train", 0)),
        "n_val": int(split.get("n_val", 0)),
        "n_test": int(split.get("n_test", 0)),
        "patch_size": int(config.IMAGE_SIZE),
        "mean_road_coverage": float(np.mean(coverages)) if coverages else 0.0,
        "sampled_coverage_n": len(coverages),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=["baseline", "attention"],
                    choices=list(MODEL_INFO))
    args = ap.parse_args()

    from torch.utils.data import DataLoader

    device = torch.device(utils.get_device())
    print(f"[research] device={device}")
    _, _, test = extraction.get_dataset_lists()
    if not test:
        print("[research] empty test split - run the extraction stage first")
        return 1
    print(f"[research] test split: {len(test)} real patches")

    clean = DataLoader(extraction.PatchDataset(test, occlusion="none"),
                       batch_size=config.BATCH_SIZE, shuffle=False,
                       num_workers=config.NUM_WORKERS)
    occluded = DataLoader(extraction.PatchDataset(test, occlusion="always"),
                          batch_size=config.BATCH_SIZE, shuffle=False,
                          num_workers=config.NUM_WORKERS)

    results: dict = {}
    for name in args.models:
        info = MODEL_INFO[name]
        if not info["ckpt"].exists():
            print(f"[research] skipping {name}: {info['ckpt'].name} not found")
            continue
        print(f"[research] evaluating {name} ...")
        model = extraction._load_model(device, name)
        clean_m = evaluate_model(model, clean, device)
        occ_m = evaluate_model(model, occluded, device)
        results[name] = {
            "label": info["label"], "role": info["role"],
            "params": int(sum(p.numel() for p in model.parameters())),
            "clean": clean_m, "occluded": occ_m,
            # Occlusion-Recall keeps the project's own definition: IoU measured
            # on synthetically occluded imagery.
            "occlusion_recall": occ_m["iou"],
        }
        print(f"[research]   {name}: IoU={clean_m['iou']:.3f} Dice={clean_m['dice']:.3f} "
              f"P={clean_m['precision']:.3f} R={clean_m['recall']:.3f} "
              f"conf={clean_m['mean_confidence']:.3f} "
              f"{clean_m['inference_ms_per_image']:.0f} ms/img")

    payload = {
        "models": results,
        "dataset": dataset_stats(),
        "device": str(device),
        "note": "Measured on the held-out real DeepGlobe test split with the "
                "checkpoints in outputs/.",
    }
    if "baseline" in results and "attention" in results:
        b, a = results["baseline"], results["attention"]
        payload["delta"] = {k: a["clean"][k] - b["clean"][k]
                            for k in ("iou", "dice", "precision", "recall", "f1")}
        payload["delta"]["occlusion_recall"] = a["occlusion_recall"] - b["occlusion_recall"]

    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[research] wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
