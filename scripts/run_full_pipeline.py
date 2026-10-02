"""
Run the FULL Extraction Track end-to-end in order and log everything.

  Part A  evaluate the trained baseline + export masks
  Part C  evaluate the trained attention model + build the comparison table
  Part B  build the graph (classical healing) + shared contract
  Part D  train the GNN link predictor + benchmark classical vs learned
  Final  network analysis on the produced graph

Usage:  python scripts/run_full_pipeline.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STEPS = [
    ("PART A  (baseline eval + mask export)",
     [sys.executable, "scripts/run_part_a.py", "--eval-only"]),
    ("PART C  (attention eval + comparison table)",
     [sys.executable, "scripts/run_part_c.py", "--eval-only"]),
    ("PART B  (skeleton + classical healing + contract)",
     [sys.executable, "scripts/run_part_b.py"]),
    ("PART D  (GNN link-prediction training + benchmark)",
     [sys.executable, "scripts/run_part_d.py", "--epochs", "40", "--bench-graphs", "12"]),
    ("FINAL   (network analysis on the produced graph)",
     [sys.executable, "main.py", "--stage", "analysis", "--source", "teammate_export"]),
]


def main():
    for title, cmd in STEPS:
        print("\n" + "#" * 78)
        print(f"# {title}")
        print(f"# $ {' '.join(cmd[1:])}")
        print("#" * 78, flush=True)
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode != 0:
            print(f"\n[full] STEP FAILED (exit {r.returncode}): {title}")
            sys.exit(r.returncode)
    print("\n[full] ALL STEPS COMPLETED")


if __name__ == "__main__":
    main()
