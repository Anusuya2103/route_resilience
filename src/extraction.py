"""
PART A -- Occlusion-robust road segmentation.

Converts satellite imagery into binary road masks using a deliberately simple
baseline: U-Net with a ResNet34 encoder (segmentation_models_pytorch) and a
BCE + Dice loss.  No transformers, no attention, no connectivity loss and no
in-graph healing -- those are the responsibility of the graph stage.

Every function in this module is callable directly and used by main.py / the
CLI stages:

    check_dataset_available()
    get_dataset_lists()
    train_segmentation_model()
    evaluate_segmentation_model()
    export_masks()
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset

from src import utils
import config

# matplotlib needs the non-interactive backend before pyplot is imported.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Dataset discovery / preprocessing (A1, A2)
# ---------------------------------------------------------------------------

def check_dataset_available() -> bool:
    """True when real satellite images and masks exist under data/images & data/masks."""
    if not config.IMAGES_DIR.is_dir() or not config.MASKS_DIR.is_dir():
        return False
    img_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    n_imgs = sum(1 for p in config.IMAGES_DIR.iterdir() if p.suffix.lower() in img_exts)
    n_masks = sum(1 for _ in config.MASKS_DIR.iterdir())
    return n_imgs > 0 and n_masks > 0


def _find_mask_for_image(img_path: Path):
    """Match an image to its mask for common Massachusetts/DeepGlobe layouts."""
    stem = img_path.stem
    stems = {stem}
    if "_sat" in stem:                     # DeepGlobe: *_sat.jpg -> *_mask.png
        stems.add(stem.replace("_sat", ""))
    for s in stems:
        for suffix in (".png", ".tif", ".jpg"):
            cand = config.MASKS_DIR / f"{s}_mask{suffix}"
            if cand.exists():
                return cand
        cand = config.MASKS_DIR / f"{s}.png"   # Massachusetts: same stem
        if cand.exists():
            return cand
    # Last resort: match by unique stem prefix
    for m in config.MASKS_DIR.iterdir():
        if m.stem.startswith(stem) or stem.startswith(m.stem):
            return m
    return None


def _tile_pair(img: np.ndarray, mask: np.ndarray, tile: int):
    """Center-crop to a tile-aligned extent, then yield non-overlapping (img, mask) tiles."""
    h, w = img.shape[:2]
    hc = max(tile, (h // tile) * tile)
    wc = max(tile, (w // tile) * tile)
    y0, x0 = (h - hc) // 2, (w - wc) // 2
    img = img[y0:y0 + hc, x0:x0 + wc]
    mask = mask[y0:y0 + hc, x0:x0 + wc]
    for r in range(0, hc, tile):
        for c in range(0, wc, tile):
            it, mt = img[r:r + tile, c:c + tile], mask[r:r + tile, c:c + tile]
            if it.shape[0] == tile and it.shape[1] == tile:
                yield it, mt


def build_patches_from_real_data(max_samples: int | None = None) -> list:
    """Tile real (image, mask) pairs into IMAGE_SIZE patches saved under
    data/patches.  Only patches containing at least one road pixel are kept.

    Performance notes for large archives (e.g. DeepGlobe):
      * images without a matching mask are skipped in O(1) (no per-image reads),
      * tiles per image are capped so the training pool spans many scenes,
      * scanning stops as soon as `max_samples` patches are collected.

    Returns a list of (image_path, mask_path) tuples.
    """
    max_samples = config.MAX_TRAIN_SAMPLES if max_samples is None else max_samples
    utils.ensure_dirs(config.PATCHES_DIR / "images", config.PATCHES_DIR / "masks")
    rng = np.random.default_rng(config.SEED)
    img_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    imgs = sorted(p for p in config.IMAGES_DIR.iterdir() if p.suffix.lower() in img_exts)
    rng.shuffle(imgs)                      # deterministic random sampling order

    pairs: list = []
    skipped = 0
    used_images = 0
    max_per_image = max(4, max_samples // 20)   # spread tiles across scenes
    for ip in imgs:
        if len(pairs) >= max_samples:
            break
        mp = _find_mask_for_image(ip)
        if mp is None:
            skipped += 1
            continue
        img = cv2.imread(str(ip), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            print(f"[dataset] failed to read {ip.name} / {mp.name} -- skipped")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if mask.shape != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
        tiles_here = 0
        for i, (it, mt) in enumerate(_tile_pair(img, mask, config.IMAGE_SIZE)):
            if len(pairs) >= max_samples or tiles_here >= max_per_image:
                break
            if mt.sum() == 0:
                continue                          # skip background-only tiles
            safe_stem = re.sub(r"[^\w-]", "_", ip.stem)
            stem = f"{safe_stem}_t{i:03d}"
            ip_out = config.PATCHES_DIR / "images" / f"{stem}.png"
            mp_out = config.PATCHES_DIR / "masks" / f"{stem}.png"
            cv2.imwrite(str(ip_out), cv2.cvtColor(it, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(mp_out), (mt * 255).astype(np.uint8))
            pairs.append((ip_out, mp_out))
            tiles_here += 1
        used_images += 1
    pairs = pairs[:max_samples]
    print(f"[dataset] built {len(pairs)} road patches from {used_images} images "
          f"({skipped} images had no matching mask)")
    return pairs

def generate_synthetic_dataset(n: int | None = None) -> list:
    """Procedurally generate (image, mask) pairs for fully-offline training."""
    n = config.MAX_TRAIN_SAMPLES if n is None else n
    utils.ensure_dirs(config.PATCHES_DIR / "images", config.PATCHES_DIR / "masks")
    pairs: list = []
    for i in range(n):
        rng = np.random.default_rng(config.SEED * 1000 + i)
        mask = utils.draw_synthetic_road_network(config.IMAGE_SIZE, rng=rng)
        img = utils.make_synthetic_satellite_image(mask, rng=rng, occlude=True)
        stem = f"synth_{i:04d}"
        ip_out = config.PATCHES_DIR / "images" / f"{stem}.png"
        mp_out = config.PATCHES_DIR / "masks" / f"{stem}.png"
        cv2.imwrite(str(ip_out), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(mp_out), (mask * 255).astype(np.uint8))
        pairs.append((ip_out, mp_out))
    print(f"[dataset] generated {len(pairs)} synthetic (image, mask) pairs")
    return pairs


def _split_lists(pairs: list, train_frac: float, val_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pairs))
    n_tr = int(len(pairs) * train_frac)
    n_va = int(len(pairs) * val_frac)
    return ([pairs[i] for i in idx[:n_tr]],
            [pairs[i] for i in idx[n_tr:n_tr + n_va]],
            [pairs[i] for i in idx[n_tr + n_va:]])


def get_dataset_lists(use_synthetic: bool | None = None):
    """Return (train, val, test) lists of (image_path, mask_path) tuples.

    Priority: cached split -> real dataset -> synthetic fallback.

    A cached SYNTHETIC split is ignored as soon as a real dataset appears in
    data/images + data/masks, so adding real data automatically switches the
    pipeline to it.
    """
    if config.SPLIT_JSON.exists():
        data = json.loads(config.SPLIT_JSON.read_text())
        cached_source = data.get("source", "unknown")
        stale = (cached_source == "synthetic" and check_dataset_available())
        if data.get("samples", 0) > 0 and not stale:
            tr = [tuple(x) for x in data["train"]]
            va = [tuple(x) for x in data["val"]]
            te = [tuple(x) for x in data["test"]]
            print(f"[dataset] cached split (source={cached_source}): "
                  f"train={len(tr)} val={len(va)} test={len(te)}")
            return tr, va, te

    if use_synthetic is None:
        use_synthetic = config.SYNTHETIC_FALLBACK and not check_dataset_available()

    if use_synthetic:
        print("[dataset] no real dataset found -> using the procedurally generated "
              "SYNTHETIC road dataset")
        pairs = generate_synthetic_dataset()
        source = "synthetic"
    elif check_dataset_available():
        print(f"[dataset] building {config.IMAGE_SIZE}x{config.IMAGE_SIZE} patches from "
              f"{config.IMAGES_DIR} + {config.MASKS_DIR}")
        pairs = build_patches_from_real_data()
        source = "real"
    else:
        print("Dataset not found.")
        print("Please place the dataset in:")
        print(f"    {config.IMAGES_DIR}  (satellite images)")
        print(f"    {config.MASKS_DIR}   (ground-truth road masks)")
        print("Then run the pipeline again, or re-run with --use-synthetic to train on")
        print("the built-in synthetic road dataset instead.")
        sys.exit(1)

    if not pairs:
        print("[dataset] no usable patches were produced -- check data/images + data/masks")
        sys.exit(1)

    train, val, test = _split_lists(pairs, config.TRAIN_FRACTION, config.VAL_FRACTION,
                                    config.SEED)
    split = {"samples": len(pairs), "source": source, "n_train": len(train),
             "n_val": len(val), "n_test": len(test),
             "train": [[str(x) for x in p] for p in train],
             "val": [[str(x) for x in p] for p in val],
             "test": [[str(x) for x in p] for p in test]}
    config.SPLIT_JSON.write_text(json.dumps(split, indent=2))
    print(f"[dataset] split (source={source}) -> train={len(train)} val={len(val)} "
          f"test={len(test)}")
    return train, val, test

# ---------------------------------------------------------------------------
# Synthetic occlusion augmentation (A3)
# ---------------------------------------------------------------------------

def apply_occlusion_with_mask(image: np.ndarray, rng=None):
    """Randomly apply one or more occlusion effects to a uint8 RGB image.

    Returns (img_occluded, occ_mask) where:

        occ_mask[y, x] = 1.0  where an occlusion effect was applied
        occ_mask[y, x] = 0.0  elsewhere  (all-zero when the sample was not occluded)

    Effects (each fired independently with its config probability while a
    sample is being occluded):

      1. Dark patch          -- random rectangle with strongly reduced brightness
      2. Gaussian blur patch -- random region blurred with a large kernel
      3. Shadow-like gradient -- soft dark diagonal gradient over part of image

    The ground-truth mask is intentionally NOT modified.  This simulates roads
    that are *hidden* by trees/shadows/clouds while remaining labelable.

    The returned occ_mask feeds the Part C occlusion-conditioned loss, which
    uses it to up-weight prediction errors exactly where occluders were placed.
    """
    rng = rng if rng is not None else np.random.default_rng()
    img = image.copy().astype(np.float32)
    h, w = img.shape[:2]
    occ = np.zeros((h, w), dtype=np.float32)
    if rng.random() > config.OCCLUSION_PROB:
        return np.clip(img, 0, 255).astype(np.uint8), occ   # not occluded
    min_dim = min(h, w)

    if rng.random() < config.OCCLUSION_DARK_PROB:                      # 1) dark patch
        ph = int(min_dim * rng.uniform(0.15, 0.35))
        pw = int(min_dim * rng.uniform(0.15, 0.35))
        y0 = int(rng.integers(0, max(1, h - ph)))
        x0 = int(rng.integers(0, max(1, w - pw)))
        scale = float(rng.uniform(0.15, 0.45))
        img[y0:y0 + ph, x0:x0 + pw] *= scale
        occ[y0:y0 + ph, x0:x0 + pw] = 1.0

    if rng.random() < config.OCCLUSION_BLUR_PROB:                      # 2) blur patch
        bh = int(min_dim * rng.uniform(0.15, 0.30))
        bw = int(min_dim * rng.uniform(0.15, 0.30))
        y0 = int(rng.integers(0, max(1, h - bh)))
        x0 = int(rng.integers(0, max(1, w - bw)))
        ksize = int(rng.integers(15, 31)) | 1
        region = img[y0:y0 + bh, x0:x0 + bw]
        img[y0:y0 + bh, x0:x0 + bw] = cv2.GaussianBlur(region, (ksize, ksize), 0)
        occ[y0:y0 + bh, x0:x0 + bw] = 1.0

    if rng.random() < config.OCCLUSION_SHADOW_PROB:                    # 3) shadow gradient
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        grad = (xx / max(w - 1, 1) + yy / max(h - 1, 1)) / 2.0         # 0..1 diagonal
        th = float(rng.uniform(0.35, 0.60))
        shadow = np.clip((grad - th) * 5.0, 0.0, 1.0)                  # soft falloff
        strength = float(rng.uniform(0.25, 0.55))
        img *= (1.0 - shadow[..., np.newaxis] * strength)
        occ = np.maximum(occ, shadow)                 # soft mask (fractionally 1 near core)

    return np.clip(img, 0, 255).astype(np.uint8), occ


def apply_synthetic_occlusion(image: np.ndarray, rng=None) -> np.ndarray:
    """Backward-compatible wrapper: apply occlusion, return only the image."""
    img, _ = apply_occlusion_with_mask(image, rng)
    return img


# ---------------------------------------------------------------------------
# Dataset class (A2/A4)
# ---------------------------------------------------------------------------

def _normalize_and_tensorize(img: np.ndarray, mask: np.ndarray):
    mean = np.asarray(config.IMAGENET_MEAN, np.float32)
    std = np.asarray(config.IMAGENET_STD, np.float32)
    img = (img.astype(np.float32) / 255.0 - mean) / std
    x = torch.from_numpy(img.transpose(2, 0, 1).copy())
    y = torch.from_numpy(mask[None, ...].copy())
    return x, y


class PatchDataset(Dataset):
    """torch Dataset wrapping (image, mask) pairs.

    occlusion mode:
      "none"   -- images used as-is               (validation / test)
      "random" -- apply synthetic occlusion       (training)
      "always" -- occlude every image             (occlusion-recall evaluation)

    When return_occlusion=True the dataset yields (x, y, occ) where occ is the
    256x256 {0,1} occlusion map for that sample (all-zero when unoccluded) --
    used by the Part C occlusion-conditioned loss.
    """

    def __init__(self, samples, occlusion: str = "none", seed: int = config.SEED,
                 return_occlusion: bool = False):
        self.samples = samples
        self.occlusion = occlusion
        self.seed = seed
        self.size = config.IMAGE_SIZE
        self.return_occlusion = return_occlusion

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else \
            np.zeros((self.size, self.size, 3), np.uint8)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros((self.size, self.size), np.uint8)
        if img.shape[:2] != (self.size, self.size):
            img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        if mask.shape[:2] != (self.size, self.size):
            mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.float32)
        occ = None
        if self.occlusion == "random":
            img, occ = apply_occlusion_with_mask(
                img, rng=np.random.default_rng(self.seed + idx))
        elif self.occlusion == "always":
            img, occ = apply_occlusion_with_mask(
                img, rng=np.random.default_rng(self.seed * 1000 + idx))
        x, y = _normalize_and_tensorize(img, mask)
        if self.return_occlusion:
            occ = np.zeros((self.size, self.size), np.float32) if occ is None else occ
            occ_t = torch.from_numpy(occ[None, ...].copy())
            return x, y, occ_t
        return x, y
# ---------------------------------------------------------------------------
# Model / losses / metrics (A4, A5)
# ---------------------------------------------------------------------------

def build_unet() -> torch.nn.Module:
    """U-Net with ResNet34 encoder; falls back to random init when the ImageNet
    weights cannot be downloaded (offline)."""
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        sys.exit(f"[model] segmentation_models_pytorch is required. Install it with "
                 f"`pip install segmentation-models-pytorch` ({exc})")
    try:
        model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet",
                         in_channels=3, classes=1)
        print("[model] U-Net + ResNet34 (ImageNet pretrained encoder)")
    except Exception as exc:                        # offline / hub unavailable
        print(f"[model] ImageNet encoder weights unavailable ({exc}); random init.")
        model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                         in_channels=3, classes=1)
    return model


def dice_loss(pred_sigmoid: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    smooth = 1.0
    ip = pred_sigmoid.contiguous().view(-1)
    tp = target.contiguous().view(-1)
    inter = (ip * tp).sum()
    return 1.0 - (2.0 * inter + smooth) / (ip.sum() + tp.sum() + smooth)


def combined_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """BCE + Dice (A5)."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    return bce + dice_loss(torch.sigmoid(logits), target)


@torch.no_grad()
def _iou_dice_batch(logits: torch.Tensor, target: torch.Tensor):
    pred = (torch.sigmoid(logits) > 0.5).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = (pred + target).clamp(max=1).sum(dim=(1, 2, 3))
    dice_num = 2.0 * inter
    dice_den = pred.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1e-6
    iou = (inter / (union + 1e-6)).mean().item()
    dice = (dice_num / (dice_den + 1e-6)).mean().item()
    return iou, dice
# ---------------------------------------------------------------------------
# Training (A6) + training curves (A7)
# ---------------------------------------------------------------------------

def _run_epoch(model, loader, optimizer, device, train: bool) -> float:
    model.train() if train else model.eval()
    total, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(x)
            loss = combined_loss(logits, y)
            if train:
                loss.backward()
                optimizer.step()
            total += loss.item() * x.size(0)
            n += x.size(0)
    return total / max(n, 1)


@torch.no_grad()
def _eval_metrics_loader(model, loader, device):
    ious, dices = [], []
    for x, y in loader:
        i, d = _iou_dice_batch(model(x.to(device)), y.to(device))
        ious.append(i)
        dices.append(d)
    return float(np.mean(ious)) if ious else 0.0, float(np.mean(dices)) if dices else 0.0


def train_segmentation_model(use_synthetic: bool | None = None,
                             epochs: int | None = None) -> dict:
    """Full training loop.  Saves outputs/best_model.pth (best val IoU) and
    outputs/plots/training_curve.png.  Prints Epoch / Train Loss / Val Loss /
    Val IoU each epoch (A6)."""
    utils.set_seed(config.SEED)
    device = torch.device(utils.get_device())
    print(f"[train] device = {device}")
    train, val, _ = get_dataset_lists(use_synthetic=use_synthetic)
    if len(train) > config.MAX_TRAIN_SAMPLES:
        train = train[:config.MAX_TRAIN_SAMPLES]
    epochs = int(epochs or config.EPOCHS)
    train_loader = DataLoader(PatchDataset(train, occlusion="random"),
                              batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(PatchDataset(val, occlusion="none"),
                            batch_size=config.BATCH_SIZE, shuffle=False,
                            num_workers=config.NUM_WORKERS)
    print(f"[train] {len(train)} train / {len(val)} val samples, {epochs} epochs, "
          f"batch {config.BATCH_SIZE}, occlusion probability {config.OCCLUSION_PROB}")

    model = build_unet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    history = {"train_loss": [], "val_loss": [], "val_iou": []}
    best_iou, best_epoch = -1.0, -1
    for ep in range(1, epochs + 1):
        t_loss = _run_epoch(model, train_loader, optimizer, device, train=True)
        v_loss = _run_epoch(model, val_loader, optimizer, device, train=False)
        v_iou, _ = _eval_metrics_loader(model, val_loader, device)
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["val_iou"].append(v_iou)
        print(f"Epoch {ep}/{epochs}")
        print(f"Train Loss: {t_loss:.3f}")
        print(f"Val Loss: {v_loss:.3f}")
        print(f"Val IoU: {v_iou:.3f}")
        if v_iou > best_iou:
            best_iou, best_epoch = v_iou, ep
            torch.save(model.state_dict(), config.BEST_MODEL)
    print(f"[train] best val IoU {best_iou:.3f} at epoch {best_epoch} "
          f"-> {config.BEST_MODEL.name} saved")
    plot_training_curve(history)
    return {"best_val_iou": best_iou, "best_epoch": best_epoch, "history": history}


def plot_training_curve(history: dict) -> None:
    """Save outputs/plots/training_curve.png (train loss / val loss / val IoU)."""
    utils.ensure_dirs(config.PLOTS_OUT)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    epochs = list(range(1, len(history["train_loss"]) + 1))
    ax1.plot(epochs, history["train_loss"], "o-", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "s-", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (BCE + Dice)")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.plot(epochs, history["val_iou"], "^-", color="tab:green")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("IoU")
    ax2.set_title("Validation IoU")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    out = config.PLOTS_OUT / "training_curve.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[plot] training curves -> {out}")
# ---------------------------------------------------------------------------
# Evaluation (A8, A9) + visual examples (A10)
# ---------------------------------------------------------------------------

def _load_model(device, model_choice: str | None = None):
    """Load a trained segmentation model.

    model_choice: "baseline"  -> outputs/best_model.pth          (Part A U-Net)
                   "attention" -> outputs/best_attention_model.pth (Part C)
    """
    model_choice = model_choice or config.MODEL_CHOICE
    if model_choice == "attention":
        ckpt = config.ATTENTION_MODEL
        if not ckpt.exists():
            print(f"[eval] attention checkpoint not found ({ckpt}).")
            print("    Train it first:  python scripts/run_part_c.py")
            sys.exit(1)
        from src.attention_model import OcclusionAttentionUNet
        model = OcclusionAttentionUNet().to(device)
        print("[model] loaded attention U-Net (Part C)")
    else:
        ckpt = config.BEST_MODEL
        if not ckpt.exists():
            print("[eval] outputs/best_model.pth not found. Train the model first with:")
            print("    python main.py --stage extraction")
            sys.exit(1)
        model = build_unet().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model


def evaluate_segmentation_model(model_choice: str | None = None) -> dict:
    """Test IoU/Dice + Occlusion-Recall on the held-out test split (A8, A9).

    Metrics are written to the file that matches the model so the Part A
    baseline numbers are never overwritten by the Part C attention model
    (the Part C comparison table reads both files).
    """
    model_choice = model_choice or config.MODEL_CHOICE
    device = torch.device(utils.get_device())
    model = _load_model(device, model_choice)
    _, _, test = get_dataset_lists()
    if not test:
        print("[eval] empty test split -- nothing to evaluate")
        return {"test_iou": 0.0, "test_dice": 0.0, "occlusion_recall": 0.0}

    test_loader = DataLoader(PatchDataset(test, occlusion="none"),
                             batch_size=config.BATCH_SIZE, shuffle=False,
                             num_workers=config.NUM_WORKERS)
    occ_loader = DataLoader(PatchDataset(test, occlusion="always"),
                            batch_size=config.BATCH_SIZE, shuffle=False,
                            num_workers=config.NUM_WORKERS)
    iou, dice = _eval_metrics_loader(model, test_loader, device)
    print("========== TEST RESULTS ==========")
    print(f"IoU: {iou:.3f}")
    print(f"Dice: {dice:.3f}")
    print("===================================")

    occ_iou, _ = _eval_metrics_loader(model, occ_loader, device)
    print("\nNOTE: Occlusion-Recall is defined in this project as the IoU measured on")
    print("the synthetically occluded test images (NOT conventional classification recall).")
    print(f"Occlusion-Recall: {occ_iou:.3f}")

    metrics = {"test_iou": iou, "test_dice": dice, "occlusion_recall": occ_iou}
    out_file = (config.SEG_METRICS_ATTENTION_JSON if model_choice == "attention"
                else config.SEG_METRICS_JSON)
    out_file.write_text(json.dumps(metrics, indent=2))
    print(f"[eval] metrics -> {out_file.name}")
    plot_predictions(model, test, device)
    return metrics


def plot_predictions(model, test_samples, device, n_examples: int = 4) -> None:
    """Save outputs/plots/predictions.png -- Original | GT | Predicted rows,
    with the final row synthetically occluded (A10)."""
    utils.ensure_dirs(config.PLOTS_OUT)
    rng = np.random.default_rng(config.SEED)
    count = min(n_examples, len(test_samples))
    idxs = rng.choice(len(test_samples), size=count, replace=False)

    def _predict(img_rgb):
        x, _ = _normalize_and_tensorize(img_rgb,
                                        np.zeros(img_rgb.shape[:2], np.float32))
        with torch.no_grad():
            logits = model(x[None].to(device))
        return ((torch.sigmoid(logits) > 0.5).float().cpu().numpy()[0, 0] * 255).astype(np.uint8)

    rows = []
    for j, k in enumerate(idxs):
        img_path, mask_path = test_samples[k]
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if img.shape[:2] != (config.IMAGE_SIZE, config.IMAGE_SIZE):
            img = cv2.resize(img, (config.IMAGE_SIZE, config.IMAGE_SIZE))
            mask = cv2.resize(mask, (config.IMAGE_SIZE, config.IMAGE_SIZE),
                              interpolation=cv2.INTER_NEAREST)
        occluded = (j == count - 1)                 # last row is an occluded example
        shown = (apply_synthetic_occlusion(img, rng=np.random.default_rng(config.SEED + j))
                 if occluded else img)
        rows.append((shown, mask, _predict(shown), occluded))

    fig, axes = plt.subplots(count, 3, figsize=(13, 4.4 * count))
    if count == 1:
        axes = axes[None, :]
    titles = ["Original Image", "Ground Truth", "Predicted Mask"]
    for j, (img, mask, pred, occ) in enumerate(rows):
        for col, data in enumerate((img, mask, pred)):
            axes[j, col].imshow(data, cmap="gray" if col else None)
            title = titles[col]
            if occ and col == 0:
                title += " (synthetically occluded)"
            axes[j, col].set_title(title)
            axes[j, col].axis("off")
    fig.tight_layout()
    out = config.PLOTS_OUT / "predictions.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] prediction examples -> {out}")
# ---------------------------------------------------------------------------
# Mask export (A11) + tiles manifest (A12)
# ---------------------------------------------------------------------------

def prepare_test_region() -> list:
    """Create 4 adjacent IMAGE_SIZE tiles under data/test_region.

    Real-data path: crop adjacent tiles from one raw image.
    Fallback: procedurally generated satellite-like tiles.
    """
    utils.ensure_dirs(config.TEST_REGION_DIR)
    existing = sorted(config.TEST_REGION_DIR.glob("tile_*.png"))
    if len(existing) >= 3:
        return existing[:4]

    tile = config.IMAGE_SIZE
    rng = np.random.default_rng(config.SEED + 5)
    if check_dataset_available():
        img_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        imgs = sorted(p for p in config.IMAGES_DIR.iterdir() if p.suffix.lower() in img_exts)
        full = cv2.cvtColor(cv2.imread(str(imgs[-1]), cv2.IMREAD_COLOR),
                            cv2.COLOR_BGR2RGB)      # one held-out region image
        if full.shape[0] < 2 * tile or full.shape[1] < 2 * tile:
            full = cv2.resize(full, (2 * tile, 2 * tile))
        for i, (r, c) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
            patch = full[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile]
            cv2.imwrite(str(config.TEST_REGION_DIR / f"tile_{i + 1:03d}.png"),
                        cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
    else:                                           # synthetic fallback tiles
        for i in range(4):
            mask = utils.draw_synthetic_road_network(tile, rng=rng)
            for _ in range(100):                    # keep generating until road-rich
                if mask.sum() > 300:
                    break
                mask = utils.draw_synthetic_road_network(tile, rng=rng)
            img = utils.make_synthetic_satellite_image(mask, rng=rng, occlude=True)
            cv2.imwrite(str(config.TEST_REGION_DIR / f"tile_{i + 1:03d}.png"),
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return sorted(config.TEST_REGION_DIR.glob("tile_*.png"))[:4]


def export_masks(model_choice: str | None = None) -> None:
    """Infer road masks for the adjacent test-region tiles.

    model_choice selects WHICH model produced the masks ("baseline" Part A or
    "attention" Part C) so downstream stages need not know which model ran.

    Saves outputs/masks/mask_tile_XXX.png (0 = background, 255 = road) and
    writes outputs/tiles_manifest.json (A11, A12).
    """
    model_choice = model_choice or config.MODEL_CHOICE
    device = torch.device(utils.get_device())
    print(f"[export-masks] using model: {model_choice}")
    model = _load_model(device, model_choice)
    tiles = prepare_test_region()
    if not tiles:
        print("[export-masks] no test-region tiles available")
        sys.exit(1)
    utils.ensure_dirs(config.MASKS_OUT)
    manifest = []
    for tile_path in tiles:
        img = cv2.cvtColor(cv2.imread(str(tile_path), cv2.IMREAD_COLOR),
                           cv2.COLOR_BGR2RGB)
        if img.shape[:2] != (config.IMAGE_SIZE, config.IMAGE_SIZE):
            img = cv2.resize(img, (config.IMAGE_SIZE, config.IMAGE_SIZE))
        x, _ = _normalize_and_tensorize(img, np.zeros(img.shape[:2], np.float32))
        with torch.no_grad():
            logits = model(x[None].to(device))
        pred = ((torch.sigmoid(logits) > 0.5).float().cpu().numpy()[0, 0] * 255)
        mask8 = pred.astype(np.uint8)
        num = re.search(r"(\d+)", tile_path.stem)
        num = num.group(1) if num else tile_path.stem
        tile_id = f"tile_{num}"
        mask_path = config.MASKS_OUT / f"mask_{tile_id}.png"
        cv2.imwrite(str(mask_path), mask8)
        manifest.append({
            "tile_id": tile_id,
            "source_image": str(tile_path.relative_to(config.ROOT_DIR)).replace("\\", "/"),
            "width": int(img.shape[1]),
            "height": int(img.shape[0]),
            "mask_path": str(mask_path.relative_to(config.ROOT_DIR)).replace("\\", "/"),
            "bounds": None,                 # no georeferencing available in the MVP
        })
    config.TILES_MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"[export-masks] saved {len(manifest)} masks to {config.MASKS_OUT}")
    print(f"[export-masks] manifest -> {config.TILES_MANIFEST.name}")
