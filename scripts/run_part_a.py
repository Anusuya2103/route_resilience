"""
PART A driver -- Baseline U-Net road segmentation.

    python scripts/run_part_a.py                 # train (if needed) + evaluate + export masks
    python scripts/run_part_a.py --epochs 10      # explicit epoch count
    python scripts/run_part_a.py --eval-only      # evaluate an existing checkpoint
    python scripts/run_part_a.py --export-only    # only re-run inference on the test region

Pipeline:
  1. dataset check / 256x256 patch building (real DeepGlobe/Massachusetts or synthetic)
  2. occlusion augmentation (~35% of samples) -- the occlusion MAP is also produced
     here (apply_occlusion_with_mask) for Part C's occlusion-conditioned loss
  3. U-Net (ResNet34) + BCE/Dice training
  4. evaluation: IoU, Dice, Occlusion-Recall  (saved to outputs/segmentation_metrics.json)
  5. inference export: mask_<tile_id>.png (0/255) + tiles_manifest.json

This is the BASELINE that Part C's attention model is compared against; the
metrics written here are exactly the three numbers the Part C table needs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from src import utils, extraction


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=None, help="override EPOCHS")
    ap.add_argument("--samples", type=int, default=None, help="override MAX_TRAIN_SAMPLES")
    ap.add_argument("--use-synthetic", action="store_true",
                    help="train on the procedural dataset instead of a real one")
    ap.add_argument("--eval-only", action="store_true", help="skip training")
    ap.add_argument("--export-only", action="store_true", help="only export masks")
    args = ap.parse_args()

    if args.samples is not None:
        config.MAX_TRAIN_SAMPLES = int(args.samples)
    utils.set_seed(config.SEED)
    utils.print_banner("PART A -- BASELINE U-NET ROAD SEGMENTATION")

    extraction.check_dataset_available()
    # --eval-only / --export-only are independent switches:
    #   eval-only   -> skip training, still evaluate
    #   export-only -> skip training and evaluation, still export masks
    do_train = not (args.eval_only or args.export_only)
    do_eval = not args.export_only
    do_export = not args.eval_only

    if do_train:
        extraction.train_segmentation_model(use_synthetic=args.use_synthetic or None,
                                           epochs=args.epochs)
    else:
        print("[partA] skipping training")
    if do_eval:
        metrics = extraction.evaluate_segmentation_model(model_choice="baseline")
        print(f"\n[partA] IoU              : {metrics['test_iou']:.3f}")
        print(f"[partA] Dice             : {metrics['test_dice']:.3f}")
        print(f"[partA] Occlusion-Recall : {metrics['occlusion_recall']:.3f}")
    if do_export:
        extraction.export_masks(model_choice="baseline")
    print(f"[partA] done. metrics -> {config.SEG_METRICS_JSON.name}")


if __name__ == "__main__":
    main()