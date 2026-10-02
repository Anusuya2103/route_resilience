"""
Package the road_resilience project as a portable CODE-ONLY zip.

Includes: all python sources, config, requirements, docs, plus a GPU_NOTES.txt
and empty data/output folders so the expected structure exists on extraction.

Excludes (transfer separately):
    data/   (~4.1 GB, 15,200 files: satellite images + masks)
    outputs/ (~203 MB: trained checkpoints and generated artefacts)
"""

import os
import zipfile
from pathlib import Path

files = []
for p in ["main.py", "config.py", "requirements.txt", "README.md",
          "TRAIN_COMMANDS.txt", ".gitignore"]:
    if Path(p).exists():
        files.append(p)
for d in ["src", "scripts", "dashboard"]:
    for f in sorted(Path(d).glob("*.py")):
        files.append(str(f).replace(os.sep, "/"))

GPU = """GPU TRAINING NOTES
==================
On a CUDA machine, install a CUDA build of PyTorch FIRST (requirements.txt
would otherwise install the CPU build):

  # check your driver / CUDA version, then pick the matching index, e.g. CUDA 12.1:
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  pip install -r requirements.txt          # everything else

Verify the GPU is visible:

  python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

The code auto-detects the GPU (src/utils.py -> get_device); nothing to change.
Recommended config tweaks in config.py for a GPU:

  BATCH_SIZE = 16            # 8 is conservative; 16-32 uses a modern GPU well
  NUM_WORKERS = 4            # data loading is the usual GPU bottleneck
  EPOCHS = 20-30             # GPU runs are cheap, so train longer
  MAX_TRAIN_SAMPLES = 2000   # far more data is affordable on a GPU

Approximate reference: the CPU run took ~30 min for 10 epochs on 200 samples.
A mid-range GPU should do 10 epochs on 2000 samples in roughly the same wall time.

Reproducibility note: src/utils.py set_seed() seeds python/numpy/torch and calls
torch.cuda.manual_seed_all() when a GPU is present. For bit-exact GPU runs add:

  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

Dataset: copy data/images and data/masks across separately (~4.1 GB,
15,200 files) -- they are NOT inside this code zip.
"""

out = "road_resilience_code.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        z.write(f, f"road_resilience/{f}")
    z.writestr("road_resilience/GPU_NOTES.txt", GPU)
    for d in ["data/images", "data/masks", "data/test_region",
              "outputs/masks", "outputs/plots"]:
        z.writestr(f"road_resilience/{d}/.gitkeep", "")

print(f"created {out} ({os.path.getsize(out) / 1024:.1f} KB, {len(files)} code files)")
with zipfile.ZipFile(out) as z:
    for n in z.namelist():
        print("   ", n)
