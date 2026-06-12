from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hw3_act.config import load_config
from hw3_act.train_loop import train_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ACT-style policy for HW3 Task 2.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = train_from_config(cfg)
    print(f"Saved run artifacts to {out_dir}")


if __name__ == "__main__":
    main()
