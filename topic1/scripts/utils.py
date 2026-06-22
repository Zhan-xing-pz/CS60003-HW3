from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = "configs/project.yaml") -> dict[str, Any]:
    cfg_path = ROOT / path
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def ensure_dir(path: str | Path) -> Path:
    p = resolve_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, data: Any) -> None:
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def resize_max_width(image: np.ndarray, max_width: int) -> np.ndarray:
    h, w = image.shape[:2]
    if w <= max_width:
        return image
    scale = max_width / float(w)
    return cv2.resize(image, (max_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def imread(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    p = resolve_path(path)
    data = np.fromfile(str(p), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite(path: str | Path, image: np.ndarray) -> bool:
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    suffix = p.suffix if p.suffix else ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        return False
    encoded.tofile(str(p))
    return True


def laplacian_sharpness(image_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def make_contact_sheet(images: list[np.ndarray], cols: int = 8, cell_width: int = 180) -> np.ndarray:
    if not images:
        raise ValueError("No images for contact sheet.")
    resized: list[np.ndarray] = []
    for img in images:
        h, w = img.shape[:2]
        scale = cell_width / float(w)
        cell = cv2.resize(img, (cell_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)
        resized.append(cell)
    cell_h = max(img.shape[0] for img in resized)
    rows = int(np.ceil(len(resized) / cols))
    sheet = np.full((rows * cell_h, cols * cell_width, 3), 245, dtype=np.uint8)
    for idx, img in enumerate(resized):
        r, c = divmod(idx, cols)
        y = r * cell_h
        x = c * cell_width
        sheet[y : y + img.shape[0], x : x + img.shape[1]] = img
    return sheet


def write_ply_xyzrgb(path: str | Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    from plyfile import PlyData, PlyElement

    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.clip(np.asarray(rgb), 0, 255).astype(np.uint8)
    vertices = np.empty(
        xyz.shape[0],
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertices["x"] = xyz[:, 0]
    vertices["y"] = xyz[:, 1]
    vertices["z"] = xyz[:, 2]
    vertices["red"] = rgb[:, 0]
    vertices["green"] = rgb[:, 1]
    vertices["blue"] = rgb[:, 2]
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(str(p))


def read_ply_xyzrgb(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    from plyfile import PlyData

    vertex = PlyData.read(str(resolve_path(path)))["vertex"]
    xyz = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float32)
    if {"red", "green", "blue"}.issubset(vertex.data.dtype.names or ()):
        rgb = np.column_stack([vertex["red"], vertex["green"], vertex["blue"]]).astype(np.uint8)
    else:
        rgb = np.full((len(xyz), 3), (120, 200, 170), dtype=np.uint8)
    return xyz, rgb
