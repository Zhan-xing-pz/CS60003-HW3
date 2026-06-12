from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hw3_act.plotting import plot_runs


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training and zero-shot evaluation metrics.")
    parser.add_argument("--runs", nargs="+", required=True, help="Run output directories.")
    parser.add_argument("--output-dir", default="outputs/figures", help="Directory for generated figures.")
    args = parser.parse_args()
    plot_runs(args.runs, args.output_dir)
    print(f"Saved figures to {args.output_dir}")


if __name__ == "__main__":
    main()
