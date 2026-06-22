from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hw3_act.lerobot_utils import ENV_TO_REPO_ID, project_env, python_executable


def convert_env(env_name: str, root: Path, force: bool) -> Path:
    repo_id = ENV_TO_REPO_ID[env_name]
    final_root = root / repo_id
    if (final_root / "meta" / "info.json").exists():
        info = json.loads((final_root / "meta" / "info.json").read_text(encoding="utf-8"))
        if info.get("codebase_version") == "v3.0" and not force:
            print(f"{env_name}: already v3.0 at {final_root}")
            return final_root

    cmd = [
        str(python_executable()),
        "-m",
        "lerobot.datasets.v30.convert_dataset_v21_to_v30",
        f"--repo-id={repo_id}",
        f"--root={root}",
        "--push-to-hub=false",
        "--data-file-size-in-mb=100",
        "--video-file-size-in-mb=500",
    ]
    if force:
        cmd.append("--force-conversion")
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=project_env(), check=True)
    return final_root


def merge_abc(root: Path, output_repo_id: str = "local/calvin_ABC") -> Path:
    from lerobot.datasets.dataset_tools import merge_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    output_root = root / output_repo_id
    if (output_root / "meta" / "info.json").exists():
        print(f"ABC merged dataset already exists at {output_root}")
        return output_root

    datasets = [
        LeRobotDataset(ENV_TO_REPO_ID[name], root=root / ENV_TO_REPO_ID[name])
        for name in ["A", "B", "C"]
    ]
    print(f"Merging A/B/C into {output_root}", flush=True)
    merge_datasets(datasets, output_repo_id=output_repo_id, output_dir=output_root)
    return output_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert downloaded CALVIN v2.1 datasets to LeRobot v3.0.")
    parser.add_argument("--envs", nargs="+", default=["A", "B", "C", "D"], choices=["A", "B", "C", "D"])
    parser.add_argument("--root", default="data/lerobot_v21")
    parser.add_argument("--merge-abc", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    os.environ.update(project_env())
    root = ROOT / args.root
    for env_name in args.envs:
        convert_env(env_name, root, args.force)
    if args.merge_abc:
        merge_abc(root)


if __name__ == "__main__":
    main()
