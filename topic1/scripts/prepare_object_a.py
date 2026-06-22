from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from utils import ensure_dir, imwrite, laplacian_sharpness, load_config, make_contact_sheet, resize_max_width, resolve_path, write_json


def sample_indices(total_frames: int, target_count: int) -> np.ndarray:
    if total_frames <= 0:
        raise ValueError("Video reports zero frames.")
    target_count = min(target_count, total_frames)
    # Avoid the first and last second when possible; handheld videos often contain motion start/stop blur.
    margin = min(max(total_frames // 20, 0), 45)
    lo, hi = margin, max(margin + 1, total_frames - margin - 1)
    return np.linspace(lo, hi, target_count, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract uniformly sampled sharp frames from object A video.")
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--video", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--max-width", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    video_path = resolve_path(args.video or cfg["inputs"]["object_a_video"])
    out_dir = ensure_dir(args.output_dir or cfg["object_a"]["output_dir"])
    meta_path = out_dir.parent / "frames_metadata.json"
    sheet_path = out_dir.parent / "contact_sheet.jpg"
    count = args.count or int(cfg["object_a"]["frame_count"])
    max_width = args.max_width or int(cfg["object_a"]["max_image_width"])

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    indices = sample_indices(total, count)

    metadata = {
        "video": str(video_path),
        "total_frames": total,
        "fps": fps,
        "source_size": [width, height],
        "selected_count": int(len(indices)),
        "frames": [],
    }
    contact_images: list[np.ndarray] = []

    for out_idx, frame_idx in enumerate(tqdm(indices, desc="Extracting object A frames")):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok:
            continue
        frame = resize_max_width(frame, max_width)
        sharpness = laplacian_sharpness(frame)
        filename = f"frame_{out_idx:04d}.png"
        imwrite(out_dir / filename, frame)
        if out_idx % max(1, len(indices) // 24) == 0:
            contact_images.append(frame)
        metadata["frames"].append(
            {
                "file": str(Path(cfg["object_a"]["output_dir"]) / filename),
                "source_frame": int(frame_idx),
                "time_sec": float(frame_idx / fps) if fps > 0 else None,
                "sharpness_laplacian_var": sharpness,
                "size": [int(frame.shape[1]), int(frame.shape[0])],
            }
        )

    cap.release()
    write_json(meta_path, metadata)
    if contact_images:
        imwrite(sheet_path, make_contact_sheet(contact_images, cols=6, cell_width=220))

    print(f"Wrote {len(metadata['frames'])} frames to {out_dir}")
    print(f"Metadata: {meta_path}")
    print(f"Contact sheet: {sheet_path}")


if __name__ == "__main__":
    main()
