from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import resolve_device, set_seed
from .dataset import build_eval_dataset
from .io_utils import ensure_dir
from .lerobot_utils import checkpoint_pretrained_dir, project_env, run_output_dir
from .model import build_model


def evaluate_from_config(cfg: dict[str, Any], checkpoint: str | Path) -> Path:
    if cfg.get("data", {}).get("mode") == "lerobot":
        return evaluate_lerobot_from_config(cfg, checkpoint)

    seed = int(cfg["run"].get("seed", 42))
    set_seed(seed)
    out_dir = ensure_dir(Path(cfg["run"].get("output_root", "outputs")) / cfg["run"]["name"])
    metrics_dir = ensure_dir(out_dir / "metrics")
    device = resolve_device(cfg["train"].get("device", "auto"))

    ckpt = torch.load(checkpoint, map_location=device)
    model = build_model(cfg["model"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ds = build_eval_dataset(cfg["data"], cfg["model"], seed)
    loader = DataLoader(
        ds,
        batch_size=int(cfg["eval"].get("batch_size", cfg["train"]["batch_size"])),
        shuffle=False,
        num_workers=int(cfg["eval"].get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )

    rows: list[dict[str, float]] = []
    max_batches = cfg["eval"].get("max_batches")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"{cfg['run']['name']} eval_D"), start=1):
            if max_batches is not None and batch_idx > max_batches:
                break
            state = batch["state"].to(device)
            action = batch["action"].to(device)
            pred = model(state)
            per_item_l1 = torch.mean(torch.abs(pred - action), dim=(1, 2)).detach().cpu().numpy()
            success = batch.get("success")
            success_np = success.detach().cpu().numpy() if success is not None else np.full(len(per_item_l1), np.nan)
            for i, action_l1 in enumerate(per_item_l1):
                rows.append(
                    {
                        "sample_index": len(rows),
                        "batch_index": batch_idx,
                        "action_l1": float(action_l1),
                        "success": float(success_np[i]) if not math.isnan(float(success_np[i])) else np.nan,
                        "episode_len": float(cfg["model"]["chunk_size"]),
                    }
                )

    episodes = pd.DataFrame(rows)
    episodes.to_csv(metrics_dir / "eval_D_episodes.csv", index=False)
    valid_success = episodes["success"].dropna() if not episodes.empty else pd.Series(dtype=float)
    success_rate = float(valid_success.mean()) if len(valid_success) else np.nan
    summary = pd.DataFrame(
        [
            {
                "run_name": cfg["run"]["name"],
                "checkpoint": str(checkpoint),
                "success_rate": success_rate,
                "mean_action_l1": float(episodes["action_l1"].mean()) if not episodes.empty else np.nan,
                "std_action_l1": float(episodes["action_l1"].std()) if len(episodes) > 1 else 0.0,
                "num_episodes": int(len(episodes)),
                "avg_episode_len": float(episodes["episode_len"].mean()) if not episodes.empty else np.nan,
            }
        ]
    )
    summary.to_csv(metrics_dir / "eval_D_summary.csv", index=False)
    print(summary.to_string(index=False))
    return metrics_dir / "eval_D_summary.csv"


def evaluate_lerobot_from_config(cfg: dict[str, Any], checkpoint: str | Path) -> Path:
    summary, episodes = evaluate_lerobot_checkpoint_metrics(cfg, checkpoint)
    out_dir = ensure_dir(run_output_dir(cfg))
    metrics_dir = ensure_dir(out_dir / "metrics")
    episodes.to_csv(metrics_dir / "eval_D_episodes.csv", index=False)
    pd.DataFrame([summary]).to_csv(metrics_dir / "eval_D_summary.csv", index=False)
    print(pd.DataFrame([summary]).to_string(index=False))
    return metrics_dir / "eval_D_summary.csv"


def evaluate_lerobot_checkpoint_metrics(
    cfg: dict[str, Any],
    checkpoint: str | Path,
    max_batches: int | None = None,
    desc: str | None = None,
    repo_id: str | None = None,
    root: str | Path | None = None,
    episodes: list[int] | None = None,
    split_name: str = "eval_D",
) -> tuple[dict[str, Any], pd.DataFrame]:
    import os

    os.environ.update(project_env())
    import torch
    from torch.utils.data import DataLoader

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    seed = int(cfg["run"].get("seed", 42))
    set_seed(seed)
    device = resolve_device(cfg["train"].get("device", "cuda"))
    pretrained_dir = checkpoint_pretrained_dir(checkpoint)

    policy_cfg = PreTrainedConfig.from_pretrained(pretrained_dir)
    policy_cfg.device = str(device)
    policy_cls = get_policy_class(policy_cfg.type)
    policy = policy_cls.from_pretrained(pretrained_dir, config=policy_cfg, local_files_only=True)
    policy.eval()
    preprocessor, _ = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=str(pretrained_dir))

    data_cfg = cfg["data"]
    dataset_repo_id = repo_id or data_cfg["eval_repo_id"]
    dataset_root = root or data_cfg["eval_root"]
    ds_meta = LeRobotDatasetMetadata(dataset_repo_id, root=dataset_root)
    delta_timestamps = resolve_delta_timestamps(policy_cfg, ds_meta)
    dataset = LeRobotDataset(
        dataset_repo_id,
        root=dataset_root,
        episodes=episodes,
        delta_timestamps=delta_timestamps,
        video_backend="pyav",
        tolerance_s=float(cfg["train"].get("tolerance_s", 0.02)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["eval"].get("batch_size", cfg["train"]["batch_size"])),
        shuffle=False,
        num_workers=int(cfg["eval"].get("num_workers", 0)),
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    rows = []
    if max_batches is None:
        max_batches = cfg["eval"].get("max_batches")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc=desc or f"{cfg['run']['name']} eval_D"), start=1):
            if max_batches is not None and batch_idx > max_batches:
                break
            batch = preprocessor(batch)
            actions_hat = policy.predict_action_chunk(batch)
            action_target = batch["action"]
            action_l1_tensor = torch.abs(action_target - actions_hat)
            if "action_is_pad" in batch:
                valid = ~batch["action_is_pad"].unsqueeze(-1)
                action_l1_tensor = action_l1_tensor * valid
                denom = valid.expand_as(action_l1_tensor).sum().clamp_min(1)
                action_l1 = float(action_l1_tensor.sum().detach().cpu() / denom.detach().cpu())
            else:
                action_l1 = float(action_l1_tensor.mean().detach().cpu())
            rows.append(
                {
                    "batch_index": batch_idx,
                    "action_l1": action_l1,
                    "loss": action_l1,
                    "success": np.nan,
                    "batch_size": int(batch["action"].shape[0]) if "action" in batch else np.nan,
                }
            )

    episodes = pd.DataFrame(rows)
    summary = {
        "run_name": cfg["run"]["name"],
        "checkpoint": str(pretrained_dir),
        "success_rate": np.nan,
        "mean_action_l1": float(episodes["action_l1"].mean()) if not episodes.empty else np.nan,
        "std_action_l1": float(episodes["action_l1"].std()) if len(episodes) > 1 else 0.0,
        "num_samples": int(len(dataset)),
        "avg_episode_len": np.nan,
        "eval_batches": int(len(episodes)),
        "split": split_name,
    }
    return summary, episodes
