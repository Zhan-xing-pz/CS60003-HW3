from __future__ import annotations

import argparse
from pathlib import Path

from run_colmap_pipeline import normalize_3dgs_sparse_layout
from utils import resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize COLMAP dense/sparse layout for 3DGS.")
    parser.add_argument("dense_dir", help="COLMAP dense directory, e.g. outputs/colmap/object_a_singlecam/dense")
    args = parser.parse_args()
    dense_dir = resolve_path(args.dense_dir)
    normalize_3dgs_sparse_layout(dense_dir)
    target = dense_dir / "sparse" / "0"
    if not (target / "images.bin").exists():
        raise FileNotFoundError(f"Could not create 3DGS sparse/0 layout under {dense_dir}")
    print(f"3DGS sparse layout ready: {target}")


if __name__ == "__main__":
    main()
