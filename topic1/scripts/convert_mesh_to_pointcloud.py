from __future__ import annotations

import argparse
import math

import numpy as np
import trimesh

from utils import write_ply_xyzrgb


def transform(points: np.ndarray, scale: float, yaw_deg: float, translate: tuple[float, float, float]) -> np.ndarray:
    yaw = math.radians(yaw_deg)
    rot = np.array(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return (points * scale) @ rot.T + np.asarray(translate, dtype=np.float32)


def sample_mesh(mesh_path: str, count: int, color: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if mesh.is_empty:
        raise ValueError(f"Empty mesh: {mesh_path}")
    points, face_idx = trimesh.sample.sample_surface(mesh, count)
    colors = np.tile(np.asarray(color, dtype=np.uint8), (len(points), 1))
    if hasattr(mesh.visual, "face_colors") and len(mesh.visual.face_colors) > 0:
        colors = np.asarray(mesh.visual.face_colors[face_idx])[:, :3].astype(np.uint8)
    return points.astype(np.float32), colors


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample a textured/generated mesh into a colored point cloud for fusion.")
    parser.add_argument("--mesh", required=True, help="Input OBJ/PLY/GLB mesh exported by threestudio/Zero123.")
    parser.add_argument("--output", required=True, help="Output PLY path.")
    parser.add_argument("--points", type=int, default=50000)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--translate", nargs=3, type=float, default=(0.0, 0.0, 0.0))
    parser.add_argument("--fallback-color", nargs=3, type=int, default=(120, 200, 170))
    args = parser.parse_args()

    xyz, rgb = sample_mesh(args.mesh, args.points, tuple(args.fallback_color))
    xyz = transform(xyz, args.scale, args.yaw, tuple(args.translate))
    write_ply_xyzrgb(args.output, xyz, rgb)
    print(f"Wrote {len(xyz)} points to {args.output}")


if __name__ == "__main__":
    main()
