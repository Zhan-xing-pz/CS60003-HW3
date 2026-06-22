from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

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


def python_executable() -> Path:
    override = os.environ.get("HW3_PYTHON")
    return Path(override or sys.executable).expanduser().resolve()


def lerobot_train_executable() -> Path:
    override = os.environ.get("HW3_LEROBOT_TRAIN")
    if override:
        executable = Path(override).expanduser().resolve()
        if executable.is_file():
            return executable
        raise FileNotFoundError(f"HW3_LEROBOT_TRAIN does not point to a file: {executable}")

    discovered = shutil.which("lerobot-train")
    if discovered:
        return Path(discovered).resolve()

    python_dir = python_executable().parent
    for name in ("lerobot-train", "lerobot-train.exe"):
        candidate = python_dir / name
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not find 'lerobot-train'. Install the pinned requirements in the active "
        "environment or set HW3_LEROBOT_TRAIN to the executable path."
    )


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
