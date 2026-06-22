from __future__ import annotations

import argparse
import json

import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils import ensure_dir, imread, imwrite, load_config, resolve_path


COLMAP_OBJECT_A_STATS = {
    "registered_images": 49,
    "sparse_points": 2086,
    "observations": 11424,
    "mean_track_length": 5.476510,
    "mean_observations_per_image": 233.142857,
    "mean_reprojection_error_px": 0.686523,
}


def read_json(path):
    with resolve_path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def fit_to_canvas(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    target_w, target_h = size
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    resized = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_h, target_w, 3), 245, dtype=np.uint8)
    y = (target_h - resized.shape[0]) // 2
    x = (target_w - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def make_overview() -> None:
    out_dir = ensure_dir("outputs/figures")
    contact = imread("data/object_a/contact_sheet.jpg")
    object_c = imread("data/object_c/object_c_foreground_preview.jpg")
    fusion = imread("outputs/preview/fusion_preview_first_frame.jpg")
    if contact is None or object_c is None or fusion is None:
        raise FileNotFoundError("Run prepare_object_a.py, prepare_object_c.py, and generate_fusion_preview.py first.")

    panels = [
        fit_to_canvas(contact, (640, 360)),
        fit_to_canvas(object_c, (360, 360)),
        fit_to_canvas(fusion, (640, 360)),
    ]
    canvas = np.full((860, 1320, 3), 250, dtype=np.uint8)
    canvas[55:415, 40:680] = panels[0]
    canvas[55:415, 730:1090] = panels[1]
    canvas[455:815, 340:980] = panels[2]
    labels = [
        ("Object A sampled video frames", (40, 38)),
        ("Object C foreground image", (730, 38)),
        ("Unified point-cloud fusion preview", (340, 438)),
    ]
    for text, pos in labels:
        cv2.putText(canvas, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.78, (30, 30, 30), 2, cv2.LINE_AA)
    imwrite(out_dir / "asset_overview.jpg", canvas)


def make_colmap_quality() -> None:
    out_dir = ensure_dir("outputs/figures")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), dpi=180)
    axes[0].bar(["registered\nimages", "sparse\npoints"], [49, 2086], color=["#2a9d8f", "#457b9d"])
    axes[0].set_title("Object A COLMAP reconstruction")
    axes[0].set_ylabel("count")
    for idx, val in enumerate([49, 2086]):
        axes[0].text(idx, val, str(val), ha="center", va="bottom", fontsize=9)

    metrics = [
        COLMAP_OBJECT_A_STATS["mean_track_length"],
        COLMAP_OBJECT_A_STATS["mean_observations_per_image"] / 50.0,
        COLMAP_OBJECT_A_STATS["mean_reprojection_error_px"],
    ]
    labels = ["track\nlength", "obs/img\n/50", "reproj.\npx"]
    axes[1].bar(labels, metrics, color=["#8ab17d", "#e9c46a", "#e76f51"])
    axes[1].set_title("Sparse model quality indicators")
    for idx, val in enumerate(metrics):
        axes[1].text(idx, val, f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(resolve_path(out_dir / "object_a_colmap_quality.png"), bbox_inches="tight")
    plt.close(fig)


def make_method_comparison() -> None:
    out_dir = ensure_dir("outputs/figures")
    methods = ["Multi-view\n3DGS", "Text-to-3D\nSDS", "Single-image\nZero123"]
    geometry = [4.2, 2.7, 3.2]
    texture = [3.8, 3.4, 3.6]
    compute = [2.5, 4.8, 4.1]
    x = np.arange(len(methods))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8, 4), dpi=180)
    ax.bar(x - width, geometry, width, label="geometry fidelity", color="#457b9d")
    ax.bar(x, texture, width, label="texture detail", color="#2a9d8f")
    ax.bar(x + width, compute, width, label="compute cost", color="#e76f51")
    ax.set_ylim(0, 5.4)
    ax.set_xticks(x, methods)
    ax.set_ylabel("relative score / cost (1-5)")
    ax.set_title("Qualitative comparison of asset generation routes")
    ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(resolve_path(out_dir / "method_comparison.png"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate report figures for topic 1.")
    parser.add_argument("--config", default="configs/project.yaml")
    parser.parse_args()
    load_config()
    make_overview()
    make_colmap_quality()
    make_method_comparison()
    print("Wrote figures to outputs/figures")


if __name__ == "__main__":
    main()
