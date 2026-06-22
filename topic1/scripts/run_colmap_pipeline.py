from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from utils import ensure_dir, load_config, resolve_path


def run(cmd: list[str], dry_run: bool = False) -> None:
    printable = " ".join(f'"{c}"' if " " in c else c for c in cmd)
    print(f"\n$ {printable}")
    if not dry_run:
        subprocess.run(cmd, check=True)


def choose_largest_sparse_model(sparse_dir: Path) -> Path:
    candidates = [p for p in sparse_dir.iterdir() if p.is_dir() and (p / "images.bin").exists()]
    if not candidates:
        raise FileNotFoundError(f"No COLMAP sparse model found under {sparse_dir}")
    return max(candidates, key=lambda p: (p / "images.bin").stat().st_size)


def normalize_3dgs_sparse_layout(dense_dir: Path) -> None:
    sparse = dense_dir / "sparse"
    target = sparse / "0"
    if target.exists() and (target / "images.bin").exists():
        return
    if not (sparse / "images.bin").exists():
        return
    target.mkdir(parents=True, exist_ok=True)
    for name in ["cameras.bin", "images.bin", "points3D.bin", "frames.bin", "rigs.bin"]:
        src = sparse / name
        if src.exists():
            dst = target / name
            dst.write_bytes(src.read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run COLMAP feature extraction, matching, and mapping.")
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--workspace", default=None)
    parser.add_argument("--colmap", default=None)
    parser.add_argument("--camera-model", default="OPENCV")
    parser.add_argument("--separate-cameras", action="store_true", help="Use one camera per image instead of one shared camera.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    colmap = args.colmap or cfg["inputs"]["colmap_executable"]
    image_dir = resolve_path(args.image_dir or cfg["object_a"]["output_dir"])
    workspace = ensure_dir(args.workspace or cfg["object_a"]["colmap_workspace"])
    sparse_dir = ensure_dir(workspace / "sparse")
    dense_dir = ensure_dir(workspace / "dense")
    database = workspace / "database.db"

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

    run(
        [
            colmap,
            "feature_extractor",
            "--database_path",
            str(database),
            "--image_path",
            str(image_dir),
            "--ImageReader.camera_model",
            args.camera_model,
            "--ImageReader.single_camera",
            "0" if args.separate_cameras else "1",
            "--FeatureExtraction.use_gpu",
            "1",
        ],
        args.dry_run,
    )
    run(
        [
            colmap,
            "exhaustive_matcher",
            "--database_path",
            str(database),
            "--FeatureMatching.use_gpu",
            "1",
        ],
        args.dry_run,
    )
    run(
        [
            colmap,
            "mapper",
            "--database_path",
            str(database),
            "--image_path",
            str(image_dir),
            "--output_path",
            str(sparse_dir),
        ],
        args.dry_run,
    )
    best_sparse = sparse_dir / "0" if args.dry_run else choose_largest_sparse_model(sparse_dir)
    print(f"\nSelected sparse model: {best_sparse}")
    run(
        [
            colmap,
            "image_undistorter",
            "--image_path",
            str(image_dir),
            "--input_path",
            str(best_sparse),
            "--output_path",
            str(dense_dir),
            "--output_type",
            "COLMAP",
        ],
        args.dry_run,
    )
    if not args.dry_run:
        normalize_3dgs_sparse_layout(dense_dir)

    print(f"\nCOLMAP workspace: {workspace}")
    print(f"3DGS input path: {dense_dir}")


if __name__ == "__main__":
    main()
