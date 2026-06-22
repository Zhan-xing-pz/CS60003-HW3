from __future__ import annotations

import argparse
import collections
import math
from pathlib import Path
import typing

import mcubes
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf.base import ContainerMetadata, Metadata
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from omegaconf.nodes import AnyNode, BooleanNode, StringNode


ROOT = Path(__file__).resolve().parents[1]


class ProgressiveBandFrequency(nn.Module):
    def __init__(self, n_frequencies: int = 12) -> None:
        super().__init__()
        self.n_frequencies = n_frequencies
        self.register_buffer("freq_bands", 2 ** torch.linspace(0, n_frequencies - 1, n_frequencies))

    @property
    def n_output_dims(self) -> int:
        return 3 + 3 * 2 * self.n_frequencies

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = [x * 2.0 - 1.0]
        for freq in self.freq_bands:
            parts.append(torch.sin(freq * x))
            parts.append(torch.cos(freq * x))
        return torch.cat(parts, dim=-1)


class VanillaMLP(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, hidden: int = 64, hidden_layers: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(dim_in, hidden, bias=False), nn.ReLU(inplace=True)]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden, hidden, bias=False), nn.ReLU(inplace=True)]
        layers += [nn.Linear(hidden, dim_out, bias=False)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x.float())


class ObjectBField(nn.Module):
    def __init__(self, radius: float = 2.0) -> None:
        super().__init__()
        self.radius = radius
        self.encoding = ProgressiveBandFrequency(n_frequencies=12)
        self.density_network = VanillaMLP(self.encoding.n_output_dims, 1)
        self.feature_network = VanillaMLP(self.encoding.n_output_dims, 3)

    def encode_points(self, points: torch.Tensor) -> torch.Tensor:
        # Match threestudio's bounded contract_to_unisphere for bbox [-radius, radius].
        normalized = (points + self.radius) / (2.0 * self.radius)
        return self.encoding(normalized.clamp(0.0, 1.0))

    def density(self, points: torch.Tensor) -> torch.Tensor:
        enc = self.encode_points(points)
        raw = self.density_network(enc)
        r = torch.linalg.norm(points, dim=-1, keepdim=True)
        bias = 10.0 * (1.0 - r / 0.5)
        return F.softplus(raw + bias)

    def color(self, points: torch.Tensor) -> torch.Tensor:
        enc = self.encode_points(points)
        return torch.sigmoid(self.feature_network(enc))


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_state_dict(checkpoint: Path) -> tuple[dict[str, torch.Tensor], int]:
    # Avoid importing the full threestudio package during checkpoint loading.
    config_cls = type("Config", (), {})
    dreamfusion_cls = type("DreamFusion", (), {"Config": config_cls})
    torch.serialization.add_safe_globals(
        [
            str,
            list,
            bool,
            int,
            dict,
            getattr,
            collections.defaultdict,
            typing.Any,
            typing.List,
            typing.Dict,
            DictConfig,
            ListConfig,
            ContainerMetadata,
            Metadata,
            AnyNode,
            StringNode,
            BooleanNode,
            (dreamfusion_cls, "threestudio.systems.dreamfusion.DreamFusion"),
        ]
    )
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    return ckpt["state_dict"], int(ckpt.get("global_step", -1))


def load_field(checkpoint: Path, device: torch.device) -> tuple[ObjectBField, int]:
    state_dict, step = load_state_dict(checkpoint)
    field = ObjectBField(radius=2.0)
    geometry_state = {}
    for key, value in state_dict.items():
        if key.startswith("geometry.density_network."):
            geometry_state[key.removeprefix("geometry.")] = value
        elif key.startswith("geometry.feature_network."):
            geometry_state[key.removeprefix("geometry.")] = value
    missing, unexpected = field.load_state_dict(geometry_state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected geometry keys: {unexpected}")
    missing = [k for k in missing if not k.startswith("encoding.")]
    if missing:
        raise RuntimeError(f"Missing geometry keys: {missing}")
    field.to(device).eval()
    return field, step


@torch.no_grad()
def sample_density_grid(field: ObjectBField, resolution: int, chunk: int, device: torch.device) -> np.ndarray:
    radius = field.radius
    coords = torch.linspace(-radius, radius, resolution, dtype=torch.float32)
    grid_x, grid_y, grid_z = torch.meshgrid(coords, coords, coords, indexing="ij")
    points = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1), grid_z.reshape(-1)], dim=-1)
    values: list[torch.Tensor] = []
    for start in range(0, points.shape[0], chunk):
        batch = points[start : start + chunk].to(device)
        values.append(field.density(batch).squeeze(-1).cpu())
    return torch.cat(values, dim=0).reshape(resolution, resolution, resolution).numpy()


def choose_threshold(density: np.ndarray, requested: float | None) -> float:
    dmin = float(np.min(density))
    dmax = float(np.max(density))
    if requested is not None and dmin < requested < dmax:
        return float(requested)
    positive = density[density > max(dmin + 1.0e-6, 1.0e-6)]
    if positive.size == 0:
        return float((dmin + dmax) * 0.5)
    # A high percentile keeps the mesh compact when DreamFusion training is short.
    return float(np.percentile(positive, 88.0))


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Object B DreamFusion density-field mesh exported from checkpoint\n")
        for v, c in zip(vertices, colors):
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n")
        for face in faces:
            a, b, c = face + 1
            f.write(f"f {a} {b} {c}\n")


def sample_mesh_points(
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray,
    count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    tri = vertices[faces]
    areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    probs = areas / max(float(areas.sum()), 1.0e-8)
    choices = rng.choice(len(faces), size=count, replace=True, p=probs)
    chosen = tri[choices]
    u = rng.random((count, 1), dtype=np.float32)
    v = rng.random((count, 1), dtype=np.float32)
    flip = (u + v) > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]
    w = 1.0 - u - v
    points = chosen[:, 0] * w + chosen[:, 1] * u + chosen[:, 2] * v
    face_colors = colors[faces[choices]]
    point_colors = face_colors[:, 0] * w + face_colors[:, 1] * u + face_colors[:, 2] * v
    return points.astype(np.float32), np.clip(point_colors * 255.0, 0, 255).astype(np.uint8)


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(xyz, rgb):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


@torch.no_grad()
def vertex_colors(field: ObjectBField, vertices: np.ndarray, chunk: int, device: torch.device) -> np.ndarray:
    pts = torch.from_numpy(vertices.astype(np.float32))
    colors: list[torch.Tensor] = []
    for start in range(0, pts.shape[0], chunk):
        colors.append(field.color(pts[start : start + chunk].to(device)).cpu())
    return torch.cat(colors, dim=0).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export object B mesh directly from the DreamFusion checkpoint.")
    parser.add_argument(
        "--checkpoint",
        default="outputs/checkpoints/object_b/last.ckpt",
    )
    parser.add_argument("--obj", default="outputs/assets/object_b/object_b_dragon_1200.obj")
    parser.add_argument("--ply", default="outputs/assets/object_b/object_b_dreamfusion_1200_sampled.ply")
    parser.add_argument("--resolution", type=int, default=56)
    parser.add_argument("--chunk", type=int, default=65536)
    parser.add_argument("--threshold", type=float, default=25.0)
    parser.add_argument("--sample-count", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=60003)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    field, step = load_field(resolve(args.checkpoint), device)
    density = sample_density_grid(field, args.resolution, args.chunk, device)
    threshold = choose_threshold(density, args.threshold)
    print(
        "density stats:",
        f"min={density.min():.6f}",
        f"max={density.max():.6f}",
        f"mean={density.mean():.6f}",
        f"threshold={threshold:.6f}",
        f"step={step}",
    )

    volume = density - threshold
    if not (volume.min() <= 0.0 <= volume.max()):
        raise RuntimeError("No zero crossing found in sampled density field.")
    vertices, faces = mcubes.marching_cubes(volume, 0.0)
    vertices = -field.radius + (vertices.astype(np.float32) / float(args.resolution - 1)) * (2.0 * field.radius)
    faces = faces.astype(np.int32)
    colors = vertex_colors(field, vertices, args.chunk, device)

    obj_path = resolve(args.obj)
    ply_path = resolve(args.ply)
    write_obj(obj_path, vertices, faces, colors)
    points, point_colors = sample_mesh_points(vertices, faces, colors, args.sample_count, args.seed)
    write_ply(ply_path, points, point_colors)
    print(f"OBJ: {obj_path} ({len(vertices)} vertices, {len(faces)} faces)")
    print(f"PLY: {ply_path} ({len(points)} points)")


if __name__ == "__main__":
    main()
