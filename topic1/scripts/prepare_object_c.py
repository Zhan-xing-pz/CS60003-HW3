from __future__ import annotations

import argparse

import cv2
import numpy as np
from PIL import Image

from utils import ensure_dir, imread, imwrite, load_config, resize_max_width, resolve_path, write_json


def grabcut_foreground(image_bgr: np.ndarray, iterations: int = 6) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    margin_x = max(8, int(0.08 * w))
    margin_y = max(8, int(0.08 * h))
    rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)
    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(image_bgr, mask, rect, bgd, fgd, iterations, cv2.GC_INIT_WITH_RECT)
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)
    fg = cv2.GaussianBlur(fg, (5, 5), 0)
    return fg


def rembg_foreground(image_path: str) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        from rembg import remove
    except Exception:
        return None
    try:
        rgba = np.array(remove(Image.open(image_path).convert("RGBA")))
        rgb = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
        alpha = rgba[:, :, 3]
        return rgb, alpha
    except Exception as exc:
        print(f"rembg failed, falling back to GrabCut: {exc}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare foreground mask and RGBA image for object C.")
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--image", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--method", choices=["grabcut", "rembg"], default="grabcut")
    parser.add_argument("--max-width", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    image_path = resolve_path(args.image or cfg["inputs"]["object_c_image"])
    out_dir = ensure_dir(args.output_dir or cfg["object_c"]["output_dir"])
    max_width = args.max_width or int(cfg["object_c"]["max_image_width"])

    image_bgr = imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    if args.method == "rembg":
        removed = rembg_foreground(str(image_path))
        if removed is not None:
            image_bgr, alpha = removed
        else:
            image_bgr = resize_max_width(image_bgr, max_width)
            alpha = grabcut_foreground(image_bgr)
    else:
        image_bgr = resize_max_width(image_bgr, max_width)
        alpha = grabcut_foreground(image_bgr)

    rgba = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha
    preview = image_bgr.copy()
    preview[alpha < 128] = (240, 240, 240)

    imwrite(out_dir / "object_c_mask.png", alpha)
    imwrite(out_dir / "object_c_rgba.png", rgba)
    imwrite(out_dir / "object_c_foreground_preview.jpg", preview)
    write_json(
        out_dir / "object_c_metadata.json",
        {
            "source": str(image_path),
            "method": args.method,
            "size": [int(image_bgr.shape[1]), int(image_bgr.shape[0])],
            "foreground_area_ratio": float((alpha > 128).mean()),
        },
    )

    print(f"Wrote object C assets to {out_dir}")


if __name__ == "__main__":
    main()
