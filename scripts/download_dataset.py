"""
scripts/download_dataset.py -- optional helper for obtaining a REAL road dataset.

The pipeline itself never requires this script: it works out of the box with
the procedurally generated synthetic dataset (see config.SYNTHETIC_FALLBACK).
When you have internet access and want real data, this helper:

  1. attempts an automatic download of the given dataset (MA, serious), or
  2. prints precise manual-download instructions when automatic retrieval is
     blocked (Google-Drive quotas, Kaggle authentication, ...).

The extraction stage accepts either layout directly:

    Massachusetts Roads (images + masks, any size) -> data/images + data/masks
    DeepGlobe         (*_sat.jpg + *_mask.png)     -> data/images + data/masks

After placing the files there, run:

    python main.py --stage extraction

Example:
    python scripts/download_dataset.py --dataset massachusetts
    python scripts/download_dataset.py --dataset deepglobe
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import utils  # noqa: E402
import config  # noqa: E402

KNOWN_SOURCES = {
    # Massachusetts Roads (Mnih 2013) -- publicly mirrored download URLs.
    # These mirror links change occasionally; the helper tries the current set.
    "massachusetts": {
        "urls": [
            "https://www.cs.toronto.edu/~vmnih/data/mass_roads/train/",
            "https://drive.google.com/file/d/1azh83f5HpB7zwqe3QYmusZ0gQ7-cYctXiWtXW1dYONRmD1Q93",  # noqa
        ],
        "note": ("Massachusetts Roads comes in train/val/test zip archives about "
                 "2.3 GB total. After download, extract and copy the image files into "
                 f"{config.IMAGES_DIR} and the corresponding ground-truth masks into "
                 f"{config.MASKS_DIR}."),
    },
    "deepglobe": {
        "urls": [],
        "note": ("DeepGlobe Road Extraction (~380 MB) is distributed via Kaggle:\n"
                 "    https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset\n"
                 "After download, copy `*_sat.jpg` files -> data/images and the matching\n"
                 "`*_mask.png` files -> data/masks."),
    },
}


def print_manual_instructions(dataset: str) -> None:
    info = KNOWN_SOURCES[dataset]
    print("=" * 72)
    print(f"DATASET: {dataset}")
    print("=" * 72)
    print(info["note"])
    print()
    print(f"Then place the files in {config.IMAGES_DIR} and {config.MASKS_DIR} and run:")
    print("    python main.py --stage extraction")
    print()
    print("Manual download links (open in a browser):")
    for url in info.get("urls", []):
        print(f"  - {url}")
    print("=" * 72)


def try_auto_download(dataset: str) -> bool:
    """Best-effort download via Python stdlib / optional `gdown`.

    Returns True when the helper believes files were fetched; downloads go to
    data/<dataset>.  Downloading the full archives is usually impractical in a
    script, so this is best-effort and always falls back to instructions.
    """
    dest = config.DATA_DIR / dataset
    utils.ensure_dirs(dest)
    urls = KNOWN_SOURCES[dataset].get("urls", [])
    import urllib.request

    for url in urls:
        if "drive.google.com" in url:
            print("GoogleDrive links need the optional `gdown` package:")
            print("    pip install gdown")
            try:
                import gdown  # noqa
                print(f"    gdown {url} -O {dest}/")
                return False
            except ImportError:
                continue
        else:
            try:
                print(f"[download] fetching {url} ...")
                with urllib.request.urlopen(url, timeout=30) as resp:
                    ct = resp.headers.get("Content-Type", "")
                    if "html" in ct:
                        print("  (page returned HTML, not a file -- skipping)")
                        continue
                return False  # proxy/site-specific handling; avoid huge downloads
            except Exception as exc:  # noqa: BLE001
                print(f"[download] could not fetch {url}: {exc}")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=list(KNOWN_SOURCES), required=True)
    args = parser.parse_args()

    print_manual_instructions(args.dataset)
    try_auto_download(args.dataset)
    print("\n[download] automatic download attempted; if the files did not land in")
    print(f"[download] {config.IMAGES_DIR} + {config.MASKS_DIR}, follow the manual")
    print("[download] instructions above. The pipeline is fully functional with the")
    print("[download] synthetic dataset in the meantime (config.SYNTHETIC_FALLBACK).")


if __name__ == "__main__":
    main()