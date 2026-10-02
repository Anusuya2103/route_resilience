"""
PART C driver -- Occlusion-Aware Attention model (Novelty 1) vs Part A baseline.

    python scripts/run_part_c.py                    # train + evaluate + comparison table
    python scripts/run_part_c.py --epochs 4          # quick run
    python scripts/run_part_c.py --export-masks      # also write masks with this model

Trains the attention U-Net (spatial self-attention bottleneck + SE decoder) with
the occlusion-conditioned loss on the SAME split as the Part A baseline, reports
IoU / Dice / Occlusion-Recall for BOTH models, and prints the Part C comparison
table -- including whether Occlusion-Recall improved MORE than overall IoU, which
is the actual evidence that the attention machinery targets occlusion specifically.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

import config
from src import utils, extraction
from src.attention_model import OcclusionAttentionUNet, attention_loss


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_attention(epochs: int | None = None):
    """Train the attention U-Net with the occlusion-conditioned loss."""
    utils.set_seed(config.SEED)
    device = torch.device(utils.get_device())
    train_s, val_s, _ = extraction.get_dataset_lists()
    if len(train_s) > config.MAX_TRAIN_SAMPLES:
        train_s = train_s[:config.MAX_TRAIN_SAMPLES]
    epochs = int(epochs or config.EPOCHS_ATTENTION)

    # return_occlusion=True yields the per-pixel occlusion map (Part A step 2)
    train_loader = DataLoader(
        extraction.PatchDataset(train_s, occlusion="random", return_occlusion=True),
        batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(
        extraction.PatchDataset(val_s, occlusion="random", return_occlusion=True),
        batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

    model = OcclusionAttentionUNet().to(device)
    opt = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[partC] device={device} | {n_params / 1e6:.2f}M params | {epochs} epochs | "
          f"{len(train_s)} train / {len(val_s)} val")
    print(f"[partC] occlusion-conditioned loss: mix={config.OCCLOSS_MIX}, "
          f"mult={config.OCCLOSS_MULT} (pixels under occlusion weigh "
          f"{1 + config.OCCLOSS_MULT:.0f}x)")

    best_iou, best_epoch = -1.0, -1
    for ep in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for x, y, occ in train_loader:
            x, y, occ = x.to(device), y.to(device), occ.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = attention_loss(logits, y, occ)
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)
            n += x.size(0)
        train_loss = total / max(n, 1)
        val_loss, val_iou = _validate_attention(model, val_loader, device)
        print(f"Epoch {ep}/{epochs}")
        print(f"Train Loss: {train_loss:.3f}")
        print(f"Val Loss: {val_loss:.3f}")
        print(f"Val IoU: {val_iou:.3f}")
        if val_iou > best_iou:
            best_iou, best_epoch = val_iou, ep
            torch.save(model.state_dict(), config.ATTENTION_MODEL)
    print(f"[partC] best val IoU {best_iou:.3f} at epoch {best_epoch} -> "
          f"{config.ATTENTION_MODEL.name}")
    return best_iou


@torch.no_grad()
def _validate_attention(model, loader, device):
    model.eval()
    losses, ious = [], []
    for x, y, occ in loader:
        x, y, occ = x.to(device), y.to(device), occ.to(device)
        logits = model(x)
        losses.append(attention_loss(logits, y, occ).item())
        ious.append(extraction._iou_dice_batch(logits, y)[0])
    return float(np.mean(losses)), float(np.mean(ious))


@torch.no_grad()
def evaluate_attention() -> dict:
    """Same three metrics as the Part A baseline, on the same test split."""
    device = torch.device(utils.get_device())
    model = OcclusionAttentionUNet().to(device)
    model.load_state_dict(torch.load(config.ATTENTION_MODEL, map_location=device))
    model.eval()
    _, _, test = extraction.get_dataset_lists()
    clean_loader = DataLoader(extraction.PatchDataset(test, occlusion="none"),
                              batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS)
    occ_loader = DataLoader(extraction.PatchDataset(test, occlusion="always"),
                            batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS)
    iou, dice = extraction._eval_metrics_loader(model, clean_loader, device)
    occ_iou, occ_dice = extraction._eval_metrics_loader(model, occ_loader, device)
    print("========== TEST RESULTS (attention model) ==========")
    print(f"IoU: {iou:.3f}")
    print(f"Dice: {dice:.3f}")
    print("====================================================")
    print("NOTE: Occlusion-Recall = IoU on synthetically occluded test images.")
    print(f"Occlusion-Recall: {occ_iou:.3f}")
    metrics = {"test_iou": round(iou, 4), "test_dice": round(dice, 4),
               "occlusion_recall": round(occ_iou, 4), "occluded_dice": round(occ_dice, 4)}
    config.SEG_METRICS_ATTENTION_JSON.write_text(json.dumps(metrics, indent=2))
    return metrics


# ---------------------------------------------------------------------------
# Comparison table (the Part C deliverable)
# ---------------------------------------------------------------------------

def build_comparison_table(attention: dict) -> dict:
    if not config.SEG_METRICS_JSON.exists():
        print("[partC] baseline metrics not found -- run `python main.py --stage "
              "extraction` (or scripts/run_part_a.py) first.")
        return {}
    base = json.loads(config.SEG_METRICS_JSON.read_text())
    d_iou = attention["test_iou"] - base["test_iou"]
    d_occ = attention["occlusion_recall"] - base["occlusion_recall"]
    d_dice = attention["test_dice"] - base["test_dice"]
    print("\n" + "=" * 62)
    print("PART C -- ATTENTION vs BASELINE (comparison table)")
    print("=" * 62)
    print(f"{'Model':<34}{'IoU':>9}{'Dice':>9}{'Occl-Recall':>14}")
    print("-" * 62)
    print(f"{'U-Net baseline (Part A)':<34}{base['test_iou']:>9.3f}"
          f"{base['test_dice']:>9.3f}{base['occlusion_recall']:>14.3f}")
    print(f"{'Attention model (Part C)':<34}{attention['test_iou']:>9.3f}"
          f"{attention['test_dice']:>9.3f}{attention['occlusion_recall']:>14.3f}")
    print("-" * 62)
    print(f"{'Delta (attention - baseline)':<34}{d_iou:>+9.3f}{d_dice:>+9.3f}{d_occ:>+14.3f}")
    print("=" * 62)
    print("NOVELTY CHECK (claim holds only if Occlusion-Recall improves MORE than IoU):")
    if d_occ > d_iou:
        print(f"  SUPPORTED -- Occlusion-Recall {d_occ:+.3f} > IoU {d_iou:+.3f}: the attention")
        print("  machinery helps specifically under occlusion, not just segmentation in general.")
    else:
        print(f"  NOT SUPPORTED -- Occlusion-Recall {d_occ:+.3f} did not improve more than")
        print(f"  IoU {d_iou:+.3f}; on this data/split the attention block did not isolate an")
        print("  occlusion-specific gain. Reported honestly rather than overclaimed.")
    table = {"baseline": base, "attention": attention,
             "delta": {"iou": round(d_iou, 4), "dice": round(d_dice, 4),
                       "occlusion_recall": round(d_occ, 4)},
             "novelty_claim_supported": bool(d_occ > d_iou)}
    config.PART_C_TABLE_JSON.write_text(json.dumps(table, indent=2))
    return table


def plot_comparison(table: dict, path=config.PART_C_TABLE_PNG):
    if not table:
        return
    utils.ensure_dirs(config.PLOTS_OUT)
    metrics = ["test_iou", "test_dice", "occlusion_recall"]
    labels = ["IoU", "Dice", "Occlusion-Recall"]
    b = [table["baseline"][m] for m in metrics]
    a = [table["attention"][m] for m in metrics]
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(x - 0.19, b, 0.38, label="U-Net baseline (Part A)", color="steelblue")
    ax.bar(x + 0.19, a, 0.38, label="Attention model (Part C)", color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("score")
    ax.set_ylim(0, 1)
    ax.set_title("Part C: baseline vs occlusion-aware attention")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"[plot] Part C comparison -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=None, help="override EPOCHS_ATTENTION")
    ap.add_argument("--eval-only", action="store_true", help="skip training")
    ap.add_argument("--export-masks", action="store_true",
                    help="also export masks produced by the attention model")
    args = ap.parse_args()

    utils.set_seed(config.SEED)
    utils.print_banner("PART C -- OCCLUSION-AWARE ATTENTION MODEL")
    if not args.eval_only:
        train_attention(epochs=args.epochs)
    metrics = evaluate_attention()
    table = build_comparison_table(metrics)
    plot_comparison(table)
    if args.export_masks:
        print("\n[partC] exporting masks with the attention model ...")
        extraction.export_masks(model_choice="attention")
    print(f"\n[partC] comparison JSON -> {config.PART_C_TABLE_JSON.name}")


if __name__ == "__main__":
    main()