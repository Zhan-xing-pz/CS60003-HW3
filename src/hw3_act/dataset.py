from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .io_utils import read_jsonl


STATE_CANDIDATES = (
    "observation.state",
    "observation_state",
    "state",
    "robot_state",
)
ACTION_CANDIDATES = ("action", "actions")
SUCCESS_CANDIDATES = ("success", "is_success", "episode_success")


def _as_array(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.float32)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(np.float32)
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=np.float32)
    return np.asarray([value], dtype=np.float32)


def _first_existing(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if name in columns:
            return name
    return None


class SyntheticACTDataset(Dataset):
    def __init__(
        self,
        size: int,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        seed: int,
        domain_shift: float = 0.0,
    ) -> None:
        self.size = size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(0.0, 0.5, size=(state_dim, action_dim)).astype(np.float32)
        self.phase = rng.normal(0.0, 0.2, size=(chunk_size, action_dim)).astype(np.float32)
        self.states = rng.normal(domain_shift, 1.0, size=(size, state_dim)).astype(np.float32)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        state = self.states[idx]
        base = np.tanh(state @ self.weights)
        action = np.stack([base + self.phase[t] for t in range(self.chunk_size)], axis=0)
        action = action + 0.01 * np.sin(np.arange(self.chunk_size, dtype=np.float32))[:, None]
        return {
            "state": torch.from_numpy(state),
            "action": torch.from_numpy(action.astype(np.float32)),
            "success": torch.tensor(float("nan"), dtype=torch.float32),
        }


class ManifestACTDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        chunk_size: int,
        max_episodes: int | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {self.manifest_path}. Run scripts/prepare_data.py first."
            )
        records = read_jsonl(self.manifest_path)
        if max_episodes is not None:
            records = records[:max_episodes]
        self.chunk_size = chunk_size
        self.samples: list[dict[str, Any]] = []
        self._cache: dict[str, pd.DataFrame] = {}
        self._hf_cache: dict[str, Any] = {}
        for rec in records:
            path = rec.get("path")
            hf_repo = rec.get("hf_repo")
            if not path and not hf_repo:
                continue
            length = int(rec.get("num_rows") or rec.get("length") or 0)
            if length <= 0:
                length = 1
            for start in range(max(1, length - chunk_size + 1)):
                self.samples.append({**rec, "start": start})
        if not self.samples:
            raise ValueError(f"No usable samples found in {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def _frame(self, path: str) -> pd.DataFrame:
        if path not in self._cache:
            self._cache[path] = pd.read_parquet(path)
        return self._cache[path]

    def _hf_row(self, repo_id: str, row_index: int) -> dict[str, Any]:
        if repo_id not in self._hf_cache:
            from datasets import load_dataset

            self._hf_cache[repo_id] = load_dataset(repo_id, split="train")
        return dict(self._hf_cache[repo_id][row_index])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rec = self.samples[idx]
        if rec.get("hf_repo"):
            row = self._hf_row(str(rec["hf_repo"]), int(rec["row_index"]))
            columns = list(row.keys())
            state_col = _first_existing(columns, STATE_CANDIDATES)
            action_col = _first_existing(columns, ACTION_CANDIDATES)
            if state_col is None or action_col is None:
                raise KeyError(
                    f"Missing state/action columns in HF row. "
                    f"State candidates={STATE_CANDIDATES}, action candidates={ACTION_CANDIDATES}."
                )
            state = _as_array(row[state_col])
            action_value = _as_array(row[action_col])
            if action_value.ndim == 1:
                action = np.repeat(action_value[None, :], self.chunk_size, axis=0)
            else:
                action = action_value[: self.chunk_size]
                while len(action) < self.chunk_size:
                    action = np.concatenate([action, action[-1:]], axis=0)
            success = float("nan")
            success_col = _first_existing(columns, SUCCESS_CANDIDATES)
            if success_col is not None:
                success = float(row[success_col])
            return {
                "state": torch.from_numpy(state.astype(np.float32)),
                "action": torch.from_numpy(action.astype(np.float32)),
                "success": torch.tensor(success, dtype=torch.float32),
            }

        df = self._frame(str(rec["path"]))
        cols = list(df.columns)
        state_col = _first_existing(cols, STATE_CANDIDATES)
        action_col = _first_existing(cols, ACTION_CANDIDATES)
        if state_col is None or action_col is None:
            raise KeyError(
                f"Missing state/action columns in {rec['path']}. "
                f"State candidates={STATE_CANDIDATES}, action candidates={ACTION_CANDIDATES}."
            )

        start = int(rec["start"])
        end = min(start + self.chunk_size, len(df))
        state = _as_array(df.iloc[start][state_col])
        actions = [_as_array(v) for v in df.iloc[start:end][action_col].tolist()]
        while len(actions) < self.chunk_size:
            actions.append(actions[-1].copy())
        action = np.stack(actions[: self.chunk_size], axis=0).astype(np.float32)

        success_col = _first_existing(cols, SUCCESS_CANDIDATES)
        success = float("nan")
        if success_col is not None:
            success = float(df.iloc[min(end - 1, len(df) - 1)][success_col])

        return {
            "state": torch.from_numpy(state.astype(np.float32)),
            "action": torch.from_numpy(action),
            "success": torch.tensor(success, dtype=torch.float32),
        }


def stable_seed(name: str, base: int) -> int:
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()
    return base + int(digest[:6], 16) % 100000


def build_datasets(cfg: dict, model_cfg: dict, seed: int) -> tuple[Dataset, Dataset]:
    mode = cfg.get("mode", "manifest")
    chunk_size = int(model_cfg["chunk_size"])
    if mode == "synthetic":
        syn = cfg.get("synthetic", {})
        train = SyntheticACTDataset(
            size=int(syn.get("train_size", 512)),
            state_dim=int(syn.get("state_dim", model_cfg["state_dim"])),
            action_dim=int(syn.get("action_dim", model_cfg["action_dim"])),
            chunk_size=chunk_size,
            seed=stable_seed("train", seed),
            domain_shift=0.0,
        )
        valid = SyntheticACTDataset(
            size=int(syn.get("valid_size", 128)),
            state_dim=int(syn.get("state_dim", model_cfg["state_dim"])),
            action_dim=int(syn.get("action_dim", model_cfg["action_dim"])),
            chunk_size=chunk_size,
            seed=stable_seed("valid", seed),
            domain_shift=0.35,
        )
        return train, valid

    split_dir = Path(cfg.get("split_dir", "data/splits"))
    train = ManifestACTDataset(split_dir / cfg["train_manifest"], chunk_size=chunk_size)
    valid = ManifestACTDataset(split_dir / cfg["valid_manifest"], chunk_size=chunk_size)
    return train, valid


def build_eval_dataset(cfg: dict, model_cfg: dict, seed: int) -> Dataset:
    mode = cfg.get("mode", "manifest")
    chunk_size = int(model_cfg["chunk_size"])
    if mode == "synthetic":
        syn = cfg.get("synthetic", {})
        return SyntheticACTDataset(
            size=int(syn.get("eval_size", 128)),
            state_dim=int(syn.get("state_dim", model_cfg["state_dim"])),
            action_dim=int(syn.get("action_dim", model_cfg["action_dim"])),
            chunk_size=chunk_size,
            seed=stable_seed("eval", seed),
            domain_shift=0.65,
        )
    split_dir = Path(cfg.get("split_dir", "data/splits"))
    return ManifestACTDataset(split_dir / cfg["eval_manifest"], chunk_size=chunk_size)
