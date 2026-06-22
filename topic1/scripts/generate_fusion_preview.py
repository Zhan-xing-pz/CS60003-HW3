from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from prepare_object_c import grabcut_foreground
from utils import (
    ensure_dir,
    imread,
    imwrite,
    load_config,
    read_ply_xyzrgb,
    resize_max_width,
    resolve_path,
    write_json,
    write_ply_xyzrgb,
)


def rng_from_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def transform_points(xyz: np.ndarray, translate=(0, 0, 0), yaw_deg: float = 0.0, scale: float = 1.0) -> np.ndarray:
    yaw = math.radians(yaw_deg)
    rot = np.array(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return (xyz * scale) @ rot.T + np.asarray(translate, dtype=np.float32)


def image_to_relief_points(
    image_bgr: np.ndarray,
    alpha: np.ndarray,
    count: int,
    rng: np.random.Generator,
    width_scale: float = 0.9,
    relief_depth: float = 0.18,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = alpha.shape[:2]
    fg = np.argwhere(alpha > 80)
    if fg.size == 0:
        fg = np.argwhere(np.ones_like(alpha, dtype=bool))
    if len(fg) > count:
        fg = fg[rng.choice(len(fg), count, replace=False)]
    rows, cols = fg[:, 0], fg[:, 1]
    aspect = h / float(w)
    x = (cols / max(1, w - 1) - 0.5) * width_scale
    z = (0.5 - rows / max(1, h - 1)) * width_scale * aspect + width_scale * aspect * 0.5
    colors = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)[rows, cols].astype(np.float32)
    luminance = (0.2126 * colors[:, 0] + 0.7152 * colors[:, 1] + 0.0722 * colors[:, 2]) / 255.0
    y = (luminance - 0.5) * relief_depth + rng.normal(0, relief_depth * 0.08, len(rows))
    xyz = np.stack([x, y, z], axis=1).astype(np.float32)
    return xyz, colors.astype(np.uint8)


def make_background(count: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    ground_n = int(count * 0.55)
    x = rng.uniform(-4.5, 4.5, ground_n)
    y = rng.uniform(-4.5, 4.5, ground_n)
    z = rng.normal(0.0, 0.015, ground_n)
    ground = np.stack([x, y, z], axis=1)
    green = np.array([74, 114, 58], dtype=np.float32)
    soil = np.array([115, 92, 62], dtype=np.float32)
    mix = rng.uniform(0, 1, (ground_n, 1))
    ground_rgb = green * mix + soil * (1 - mix) + rng.normal(0, 12, (ground_n, 3))

    cluster_n = count - ground_n
    centers = np.array(
        [
            [-3.2, 2.6, 1.0],
            [-1.2, 3.2, 1.2],
            [1.7, 2.8, 1.1],
            [3.4, 1.6, 1.0],
            [-3.7, -1.4, 0.9],
            [3.7, -1.1, 0.9],
        ],
        dtype=np.float32,
    )
    choices = centers[rng.integers(0, len(centers), cluster_n)]
    foliage = choices + rng.normal(0, [0.45, 0.38, 0.42], (cluster_n, 3))
    foliage[:, 2] = np.clip(foliage[:, 2], 0.15, 2.4)
    foliage_rgb = np.array([48, 105, 56], dtype=np.float32) + rng.normal(0, [20, 28, 12], (cluster_n, 3))

    xyz = np.vstack([ground, foliage]).astype(np.float32)
    rgb = np.vstack([ground_rgb, foliage_rgb])
    return xyz, np.clip(rgb, 0, 255).astype(np.uint8)


def make_object_b(count: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    body_n = int(count * 0.72)
    t = rng.uniform(0, 2.4 * math.pi, body_n)
    radius = 0.26 + 0.04 * np.sin(3 * t)
    x = (t / (2.4 * math.pi) - 0.5) * 0.95
    y = radius * np.cos(t)
    z = 0.55 + radius * np.sin(t) + 0.08 * np.sin(2 * t)
    body = np.stack([x, y, z], axis=1)
    body += rng.normal(0, 0.025, body.shape)

    head_n = count - body_n
    phi = rng.uniform(0, 2 * math.pi, head_n)
    costheta = rng.uniform(-1, 1, head_n)
    theta = np.arccos(costheta)
    rr = rng.uniform(0.0, 1.0, head_n) ** (1 / 3)
    head = np.stack(
        [
            0.58 + 0.22 * rr * np.sin(theta) * np.cos(phi),
            0.02 + 0.18 * rr * np.sin(theta) * np.sin(phi),
            0.78 + 0.16 * rr * np.cos(theta),
        ],
        axis=1,
    )
    xyz = np.vstack([body, head]).astype(np.float32)
    base = np.array([42, 172, 119], dtype=np.float32)
    highlight = np.array([175, 239, 201], dtype=np.float32)
    blend = rng.uniform(0, 1, (count, 1)) ** 2
    rgb = base * (1 - blend) + highlight * blend + rng.normal(0, 10, (count, 3))
    return xyz, np.clip(rgb, 0, 255).astype(np.uint8)


def look_at(eye: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)
    return right, up, forward


def render_points(
    xyz: np.ndarray,
    rgb: np.ndarray,
    eye: np.ndarray,
    target: np.ndarray,
    width: int,
    height: int,
    focal: float,
) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (214, 229, 238)
    # Soft horizon and ground tint.
    for yy in range(height):
        a = yy / max(1, height - 1)
        if yy > height * 0.52:
            image[yy, :, :] = np.array([178, 190, 166]) * a + np.array([214, 229, 238]) * (1 - a)

    right, up, forward = look_at(eye, target)
    diff = xyz - eye[None, :]
    cam_x = diff @ right
    cam_y = diff @ up
    cam_z = diff @ forward
    valid = cam_z > 0.08
    u = (focal * cam_x[valid] / cam_z[valid] + width * 0.5).astype(np.int32)
    v = (height * 0.5 - focal * cam_y[valid] / cam_z[valid]).astype(np.int32)
    z = cam_z[valid]
    colors = rgb[valid]
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, z, colors = u[inside], v[inside], z[inside], colors[inside]
    order = np.argsort(z)[::-1]
    for idx in order:
        depth = z[idx]
        radius = int(np.clip(4.5 / depth, 1, 4))
        color = tuple(int(c) for c in colors[idx][::-1])
        cv2.circle(image, (int(u[idx]), int(v[idx])), radius, color, -1, lineType=cv2.LINE_AA)
    return image


def first_object_a_frame(cfg: dict) -> np.ndarray:
    image_dir = resolve_path(cfg["object_a"]["output_dir"])
    candidates = sorted(image_dir.glob("*.png"))
    if candidates:
        frame = imread(candidates[len(candidates) // 2], cv2.IMREAD_COLOR)
        if frame is not None:
            return frame

    cap = cv2.VideoCapture(str(resolve_path(cfg["inputs"]["object_a_video"])))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total // 2))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise FileNotFoundError("Could not read object A frame from video.")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a local point-cloud fusion preview video.")
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    rng = rng_from_seed(int(cfg["project"]["seed"]))
    fusion_cfg = cfg["fusion"]
    width, height = int(fusion_cfg["width"]), int(fusion_cfg["height"])
    frames = args.frames or int(fusion_cfg["frames"])
    if args.quick:
        frames = min(frames, 45)
    fps = int(fusion_cfg["fps"])
    budget = fusion_cfg["point_budget"]

    out_dir = ensure_dir(fusion_cfg["output_dir"])
    asset_dir = ensure_dir("outputs/assets")
    video_path = resolve_path(fusion_cfg["video_path"])
    video_path.parent.mkdir(parents=True, exist_ok=True)

    bg_xyz, bg_rgb = make_background(int(budget["background"]), rng)

    a_img = resize_max_width(first_object_a_frame(cfg), 700)
    a_alpha = grabcut_foreground(a_img, iterations=4)
    a_xyz, a_rgb = image_to_relief_points(a_img, a_alpha, int(budget["object_a"]), rng, width_scale=0.85)
    a_xyz = transform_points(a_xyz, translate=(-1.35, -0.05, 0.04), yaw_deg=22, scale=1.15)

    c_mesh_ply = asset_dir / "object_c" / "object_c_zero123_sampled.ply"
    if c_mesh_ply.exists():
        c_xyz, c_rgb = read_ply_xyzrgb(c_mesh_ply)
        if len(c_xyz) > int(budget["object_c"]):
            keep = rng.choice(len(c_xyz), size=int(budget["object_c"]), replace=False)
            c_xyz, c_rgb = c_xyz[keep], c_rgb[keep]
    else:
        c_rgba_path = resolve_path(cfg["object_c"]["output_dir"]) / "object_c_rgba.png"
        if c_rgba_path.exists():
            c_rgba = imread(c_rgba_path, cv2.IMREAD_UNCHANGED)
            c_img = c_rgba[:, :, :3]
            c_alpha = c_rgba[:, :, 3]
        else:
            c_img = resize_max_width(imread(resolve_path(cfg["inputs"]["object_c_image"]), cv2.IMREAD_COLOR), 700)
            c_alpha = grabcut_foreground(c_img, iterations=5)
        c_xyz, c_rgb = image_to_relief_points(c_img, c_alpha, int(budget["object_c"]), rng, width_scale=0.72)
        c_xyz = transform_points(c_xyz, translate=(1.38, 0.06, 0.05), yaw_deg=-26, scale=1.15)

    b_candidates = [
        asset_dir / "object_b" / "object_b_dreamfusion_1200_sampled.ply",
        asset_dir / "object_b" / "object_b_dreamfusion_sampled.ply",
    ]
    b_mesh_ply = next((path for path in b_candidates if path.exists()), b_candidates[-1])
    if b_mesh_ply.exists():
        b_xyz, b_rgb = read_ply_xyzrgb(b_mesh_ply)
        if len(b_xyz) > int(budget["object_b"]):
            keep = rng.choice(len(b_xyz), size=int(budget["object_b"]), replace=False)
            b_xyz, b_rgb = b_xyz[keep], b_rgb[keep]
        b_xyz = transform_points(b_xyz, translate=(0.0, 0.22, 0.85), yaw_deg=4, scale=0.82)
    else:
        b_xyz, b_rgb = make_object_b(int(budget["object_b"]), rng)
        b_xyz = transform_points(b_xyz, translate=(0.0, 0.22, 0.06), yaw_deg=4, scale=1.05)

    all_xyz = np.vstack([bg_xyz, a_xyz, b_xyz, c_xyz]).astype(np.float32)
    all_rgb = np.vstack([bg_rgb, a_rgb, b_rgb, c_rgb]).astype(np.uint8)

    write_ply_xyzrgb(asset_dir / "background_preview.ply", bg_xyz, bg_rgb)
    write_ply_xyzrgb(asset_dir / "object_a_preview.ply", a_xyz, a_rgb)
    write_ply_xyzrgb(asset_dir / "object_b_preview.ply", b_xyz, b_rgb)
    write_ply_xyzrgb(asset_dir / "object_c_preview.ply", c_xyz, c_rgb)
    write_ply_xyzrgb(asset_dir / "fused_preview.ply", all_xyz, all_rgb)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {video_path}")

    target = np.array([0.0, 0.12, 0.72], dtype=np.float32)
    for i in tqdm(range(frames), desc="Rendering preview"):
        angle = 2 * math.pi * i / frames
        radius = 4.0 + 0.25 * math.sin(2 * angle)
        eye = np.array([radius * math.sin(angle), -radius * math.cos(angle), 1.45 + 0.2 * math.sin(angle)], dtype=np.float32)
        frame = render_points(all_xyz, all_rgb, eye, target, width, height, focal=760.0)
        if i == 0:
            imwrite(out_dir / "fusion_preview_first_frame.jpg", frame)
        writer.write(frame)
    writer.release()

    write_json(
        out_dir / "fusion_preview_metadata.json",
        {
            "video": str(video_path),
            "frames": frames,
            "fps": fps,
            "resolution": [width, height],
            "points": {
                "background": int(len(bg_xyz)),
                "object_a": int(len(a_xyz)),
                "object_b": int(len(b_xyz)),
                "object_c": int(len(c_xyz)),
                "total": int(len(all_xyz)),
            },
            "note": "Local preview using a unified colored point-cloud representation. Object B uses the 1200-step DreamFusion checkpoint-exported sampled mesh when outputs/assets/object_b/object_b_dreamfusion_1200_sampled.ply exists, falling back to the earlier sampled mesh if needed. Object C uses the Zero123 sampled mesh when outputs/assets/object_c/object_c_zero123_sampled.ply exists; Object A remains a lightweight point-cloud proxy derived from the captured video for preview placement.",
        },
    )
    print(f"Video: {video_path}")
    print(f"PLY assets: {asset_dir}")


if __name__ == "__main__":
    main()
