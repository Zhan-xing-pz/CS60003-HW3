from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils import ensure_dir, imread, imwrite, resolve_path


def latest_method(model_dir: Path) -> str:
    with (model_dir / "results.json").open("r", encoding="utf-8") as f:
        results = json.load(f)
    return sorted(results, key=lambda name: int(name.split("_")[-1]))[-1]


def fit(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    target_w, target_h = size
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    resized = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_h, target_w, 3), 245, dtype=np.uint8)
    y = (target_h - resized.shape[0]) // 2
    x = (target_w - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def label(image: np.ndarray, text: str) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (20, 24, 28), -1)
    cv2.putText(out, text, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def make_eval_grid(model_dir: Path, method: str, out_path: Path, max_rows: int) -> None:
    gt_dir = model_dir / "test" / method / "gt"
    render_dir = model_dir / "test" / method / "renders"
    names = sorted(p.name for p in render_dir.glob("*.png"))
    if not names:
        raise FileNotFoundError(f"No render images found in {render_dir}")
    chosen = [names[i] for i in np.linspace(0, len(names) - 1, min(max_rows, len(names)), dtype=int)]

    rows = []
    for name in chosen:
        gt = imread(gt_dir / name)
        render = imread(render_dir / name)
        if gt is None or render is None:
            continue
        diff = cv2.absdiff(gt, render)
        diff = np.clip(diff * 2.0, 0, 255).astype(np.uint8)
        rows.append(
            np.hstack(
                [
                    label(fit(gt, (360, 210)), f"GT {name}"),
                    label(fit(render, (360, 210)), f"3DGS render {name}"),
                    label(fit(diff, (360, 210)), "absolute diff x2"),
                ]
            )
        )
    if not rows:
        raise RuntimeError("No valid GT/render pairs found.")
    imwrite(out_path, np.vstack(rows))


def parse_eval_log(stdout_path: Path) -> dict[str, list[tuple[int, float, float]]]:
    pattern = re.compile(r"\[ITER (\d+)\] Evaluating (test|train): L1 ([0-9.]+) PSNR ([0-9.]+)")
    series: dict[str, list[tuple[int, float, float]]] = {"train": [], "test": []}
    if not stdout_path.exists():
        return series
    text = stdout_path.read_text(encoding="utf-8", errors="ignore")
    for match in pattern.finditer(text):
        series[match.group(2)].append((int(match.group(1)), float(match.group(3)), float(match.group(4))))
    return series


def parse_loss_log(stderr_path: Path) -> list[tuple[int, float]]:
    if not stderr_path.exists():
        return []
    text = stderr_path.read_text(encoding="utf-8", errors="ignore")
    pattern = re.compile(r"(\d+)/(\d+) .*?Loss=([0-9.]+)")
    by_iter: dict[int, float] = {}
    for match in pattern.finditer(text):
        by_iter[int(match.group(1))] = float(match.group(3))
    return sorted(by_iter.items())


def make_metric_figure(
    model_dir: Path,
    method: str,
    stdout_log: Path,
    stderr_log: Path,
    out_path: Path,
    title: str,
    notes: list[str],
) -> None:
    with (model_dir / "results.json").open("r", encoding="utf-8") as f:
        results = json.load(f)[method]
    with (model_dir / "per_view.json").open("r", encoding="utf-8") as f:
        per_view = json.load(f)[method]

    eval_series = parse_eval_log(stdout_log)
    train_loss = parse_loss_log(stderr_log)

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.2), dpi=180)
    ax = axes[0, 0]
    if train_loss:
        xs, ys = zip(*train_loss)
        ax.plot(xs, ys, color="#457b9d", linewidth=1.4)
    ax.set_title("Training loss")
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    for split, color in [("train", "#2a9d8f"), ("test", "#e76f51")]:
        if eval_series[split]:
            xs = [x[0] for x in eval_series[split]]
            psnr = [x[2] for x in eval_series[split]]
            ax.plot(xs, psnr, marker="o", label=split, color=color)
    ax.set_title("Evaluation PSNR")
    ax.set_xlabel("iteration")
    ax.set_ylabel("PSNR")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    names = sorted(per_view["PSNR"])
    psnr_vals = [per_view["PSNR"][name] for name in names]
    ssim_vals = [per_view["SSIM"][name] for name in names]
    ax.plot(range(len(names)), psnr_vals, marker="o", color="#457b9d", label="PSNR")
    ax2 = ax.twinx()
    ax2.plot(range(len(names)), ssim_vals, marker="s", color="#2a9d8f", label="SSIM")
    ax.set_title("Held-out test views")
    ax.set_xlabel("view index")
    ax.set_ylabel("PSNR")
    ax2.set_ylabel("SSIM")
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    ax.axis("off")
    summary = (
        f"{title} ({method})\n\n"
        f"SSIM  {results['SSIM']:.4f}\n"
        f"PSNR  {results['PSNR']:.4f}\n"
        f"LPIPS {results['LPIPS']:.4f}\n\n"
        + "\n".join(notes)
    )
    ax.text(0.08, 0.88, summary, va="top", ha="left", fontsize=10.5, family="monospace")

    for ax in axes.ravel()[:3]:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(resolve_path(out_path), bbox_inches="tight")
    plt.close(fig)


def make_render_video(model_dir: Path, method: str, out_path: Path, fps: int, max_frames: int) -> None:
    render_dir = model_dir / "train" / method / "renders"
    frames = sorted(render_dir.glob("*.png"))
    if not frames:
        raise FileNotFoundError(f"No train render frames found in {render_dir}")
    if max_frames and len(frames) > max_frames:
        frames = [frames[i] for i in np.linspace(0, len(frames) - 1, max_frames, dtype=int)]

    first = imread(frames[0])
    if first is None:
        raise RuntimeError(f"Could not read {frames[0]}")
    h, w = first.shape[:2]
    out = resolve_path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame_path in frames:
        frame = imread(frame_path)
        if frame is None:
            continue
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        writer.write(frame)
    writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create report artifacts from a 3DGS model directory.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--stdout-log", required=True)
    parser.add_argument("--stderr-log", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--note", action="append", default=[])
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--max-video-frames", type=int, default=90)
    parser.add_argument("--grid-rows", type=int, default=4)
    args = parser.parse_args()

    model_dir = resolve_path(args.model)
    method = latest_method(model_dir)
    fig_dir = ensure_dir("outputs/figures")
    video_dir = ensure_dir("outputs/videos")

    make_eval_grid(model_dir, method, fig_dir / f"{args.prefix}_eval_grid.jpg", args.grid_rows)
    make_metric_figure(
        model_dir,
        method,
        resolve_path(args.stdout_log),
        resolve_path(args.stderr_log),
        fig_dir / f"{args.prefix}_training_metrics.png",
        args.title,
        args.note,
    )
    make_render_video(model_dir, method, video_dir / f"{args.prefix}_render.mp4", args.fps, args.max_video_frames)
    print(f"Wrote 3DGS artifacts for {args.title} / {method}")


if __name__ == "__main__":
    main()
