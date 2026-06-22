from __future__ import annotations

import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path("/data/anaconda3/envs/pz/bin/python")
LEROBOT_TRAIN = Path("/data/anaconda3/envs/pz/bin/lerobot-train")

HF_REPO_ID = "huiwon/calvin_task_ABC_D"
HF_ENDPOINT = "https://hf-mirror.com"

ENV_TO_PART = {"A": 0, "B": 1, "C": 2, "D": 3}
ENV_TO_REPO_ID = {
    "A": "local/calvin_A",
    "B": "local/calvin_B",
    "C": "local/calvin_C",
    "D": "local/calvin_D",
    "ABC": "local/calvin_ABC",
}


def project_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("HF_ENDPOINT", HF_ENDPOINT)
    env["HF_HOME"] = str(PROJECT_ROOT / ".hf_home")
    env["HF_HUB_CACHE"] = str(PROJECT_ROOT / ".hf_home" / "hub")
    env["PIP_CACHE_DIR"] = str(PROJECT_ROOT / ".pip_cache")
    env.setdefault("WANDB_MODE", "offline")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


def env_root(env_name: str, data_root: str | Path = "data/lerobot") -> Path:
    repo_id = ENV_TO_REPO_ID[env_name]
    return PROJECT_ROOT / data_root / repo_id


def run_output_dir(cfg: dict[str, Any]) -> Path:
    return PROJECT_ROOT / cfg["run"].get("output_root", "outputs") / cfg["run"]["name"]


def checkpoint_pretrained_dir(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    if (run_dir / "config.json").exists() and (run_dir / "model.safetensors").exists():
        return run_dir
    if (run_dir / "pretrained_model").exists():
        return run_dir / "pretrained_model"
    last = run_dir / "checkpoints" / "last" / "pretrained_model"
    if last.exists():
        return last
    checkpoints = sorted((run_dir / "checkpoints").glob("*/pretrained_model"))
    if not checkpoints:
        raise FileNotFoundError(f"No LeRobot pretrained_model checkpoint found under {run_dir}")
    return checkpoints[-1]
