from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hw3_act.plotting import parse_lerobot_train_log


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse LeRobot train.log into metrics/train_metrics.csv.")
    parser.add_argument("--runs", nargs="+", required=True, help="Run output directories.")
    args = parser.parse_args()
    for run in args.runs:
        df = parse_lerobot_train_log(run)
        print(f"{run}: parsed {len(df)} train metric rows")


if __name__ == "__main__":
    main()
