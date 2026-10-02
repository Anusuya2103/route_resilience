"""
utils.py -- small shared helpers used across the pipeline.

* deterministic seeding
* device selection (GPU if available, CPU fallback)
* directory creation and console banners
* procedural synthetic road drawing (used by the synthetic dataset generator
  and by the demo mask generator so the whole pipeline runs offline)
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import cv2
import numpy as np

# Make `config` importable no matter what the current working directory is.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def set_seed(seed: int = 42) -> None:
    """Reproducible randomness across python / numpy / torch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device() -> str:
    """Return 'cuda' when a GPU is available, otherwise 'cpu'."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def ensure_dirs(*paths) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def print_banner(text: str, char: str = "=", width: int = 72) -> None:
    print(f"\n{char * width}")
    print(text)
    print(f"{char * width}")


def print_section(text: str, char: str = "-", width: int = 64) -> None:
    print(f"\n{char * width}")
    print(text)
    print(f"{char * width}")


# ---------------------------------------------------------------------------
# Synthetic road drawing (offline demo data)
# ---------------------------------------------------------------------------

def draw_synthetic_road_network(size: int = 256, rng: np.random.Generator | None = None):
    """Draw a small synthetic road network into a binary mask (0/1).

    Roads are random-walk polylines, producing a network with endpoints and
    junctions -- exactly what the graph extraction + healing stage needs.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    mask = np.zeros((size, size), dtype=np.uint8)
    n_roads = int(rng.integers(3, 7))
    for _ in range(n_roads):
        x, y = int(rng.integers(0, size)), int(rng.integers(0, size))
        pts = [(x, y)]
        angle = float(rng.uniform(0.0, 2 * np.pi))
        steps = int(rng.integers(4, 9))
        for _ in range(steps):
            angle += float(rng.uniform(-0.7, 0.7))
            step = float(rng.integers(12, max(size // 3, 13)))
            x = int(np.clip(x + step * np.cos(angle), 0, size - 1))
            y = int(np.clip(y + step * np.sin(angle), 0, size - 1))
            pts.append((x, y))
        thickness = int(rng.integers(2, 4))
        cv2.polylines(mask, [np.asarray(pts, dtype=np.int32)],
                      isClosed=False, color=1, thickness=thickness)
    return mask


def make_synthetic_satellite_image(mask: np.ndarray, rng: np.random.Generator | None = None,
                                   occlude: bool = True) -> np.ndarray:
    """Turn a road mask into a fake 'satellite' RGB image (uint8).

    Roads are drawn as pale gray lanes on randomized terrain.  Optional
    occlusion blobs are painted over some road pixels WITHOUT touching the
    mask, so the segmentation model learns occlusion robustness.
    """
    rng = rng if rng is not None else np.random.default_rng(1)
    h, w = mask.shape[:2]
    base = rng.integers(30, 90, size=(h, w, 3), dtype=np.uint8)      # terrain
    img = base.astype(np.float32)
    img += rng.normal(0, 12, size=(h, w, 1)).astype(np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)

    road = np.stack([(mask > 0).astype(np.uint8)] * 3, axis=-1)
    img = np.where(road > 0, 175, img)                                # pale gray roads

    if occlude:
        # vegetation-like blobs hiding parts of the road network
        for _ in range(int(rng.integers(1, 4))):
            cx = int(rng.integers(0, w))
            cy = int(rng.integers(0, h))
            r = int(rng.integers(8, 25))
            cv2.circle(img, (cx, cy), r,
                       (int(rng.integers(40, 80)), int(rng.integers(90, 140)),
                        int(rng.integers(30, 70))), -1)
    return img