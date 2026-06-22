from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hw3_act.splits import prepare_splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare A, ABC, and D manifests for CALVIN/LeRobot data.")
    parser.add_argument("--dataset-root", required=True, help="Local dataset root or Hugging Face dataset repo id.")
    parser.add_argument("--output-dir", default="data/splits", help="Directory to write split manifests.")
    args = parser.parse_args()
    prepare_splits(args.dataset_root, args.output_dir)


if __name__ == "__main__":
    main()
