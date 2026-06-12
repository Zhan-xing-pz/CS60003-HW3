from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hw3_act.config import load_config
from hw3_act.eval import evaluate_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ACT-style policy on environment D.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    args = parser.parse_args()
    out = evaluate_from_config(load_config(args.config), args.checkpoint)
    print(f"Saved evaluation metrics to {out}")


if __name__ == "__main__":
    main()
