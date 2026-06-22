from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .io_utils import ensure_dir


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


_TRAIN_LOG_RE = re.compile(
    r"step:(?P<step>\d+(?:\.\d+)?[KMB]?)\s+"
    r"smpl:(?P<samples>\d+(?:\.\d+)?[KMB]?)\s+"
    r"ep:(?P<episodes>\d+(?:\.\d+)?[KMB]?)\s+"
    r"epch:(?P<epoch>\d+(?:\.\d+)?)\s+"
    r"loss:(?P<loss>[-+]?\d+(?:\.\d+)?)\s+"
    r"grdn:(?P<grad_norm>[-+]?\d+(?:\.\d+)?)\s+"
    r"lr:(?P<lr>[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)\s+"
    r"updt_s:(?P<update_s>[-+]?\d+(?:\.\d+)?)\s+"
    r"data_s:(?P<data_s>[-+]?\d+(?:\.\d+)?)"
)


def _parse_compact_number(value: str) -> float:
    value = value.strip()
    scale = 1.0
    if value.endswith("K"):
        scale = 1_000.0
        value = value[:-1]
    elif value.endswith("M"):
        scale = 1_000_000.0
        value = value[:-1]
    elif value.endswith("B"):
        scale = 1_000_000_000.0
        value = value[:-1]
    return float(value) * scale


def parse_lerobot_train_log(run_dir: str | Path) -> pd.DataFrame:
    run = Path(run_dir)
    log_path = run / "train.log"
    rows = []
    if not log_path.exists():
        return pd.DataFrame()
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _TRAIN_LOG_RE.search(line)
        if not match:
            continue
        row = match.groupdict()
        rows.append(
            {
                "step": int(_parse_compact_number(row["step"])),
                "samples": int(_parse_compact_number(row["samples"])),
                "episodes": int(_parse_compact_number(row["episodes"])),
                "epoch": float(row["epoch"]),
                "loss": float(row["loss"]),
                "grad_norm": float(row["grad_norm"]),
                "lr": float(row["lr"]),
                "update_s": float(row["update_s"]),
                "data_s": float(row["data_s"]),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        metrics_dir = ensure_dir(run / "metrics")
        df.to_csv(metrics_dir / "train_metrics.csv", index=False)
    return df


def _training_metrics(run: Path) -> pd.DataFrame | None:
    df = _read_csv(run / "metrics" / "train_metrics.csv")
    if df is not None and not df.empty:
        return df
    parsed = parse_lerobot_train_log(run)
    return parsed if not parsed.empty else None


def plot_runs(run_dirs: list[str | Path], output_dir: str | Path) -> None:
    output = ensure_dir(output_dir)
    runs = [Path(p) for p in run_dirs]

    plt.figure(figsize=(8, 5))
    any_train = False
    for run in runs:
        df = _training_metrics(run)
        if df is not None and not df.empty:
            any_train = True
            x_col = "step" if "step" in df.columns else "epoch"
            plt.plot(df[x_col], df["loss"], marker="o", label=run.name)
    if any_train:
        plt.xlabel("Step")
        plt.ylabel("Training loss")
        plt.title("Training Loss Curve")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output / "train_loss_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    any_valid = False
    for run in runs:
        df = _read_csv(run / "metrics" / "valid_metrics.csv")
        if df is not None and not df.empty:
            any_valid = True
            plt.plot(df["epoch"], df["action_l1"], marker="o", label=run.name)
    if any_valid:
        plt.xlabel("Epoch")
        plt.ylabel("Validation Action L1")
        plt.title("Validation Action L1 Curve")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output / "valid_action_l1_curve.png", dpi=200)
    plt.close()

    summaries = []
    for run in runs:
        df = _read_csv(run / "metrics" / "eval_D_summary.csv")
        if df is not None and not df.empty:
            summaries.append(df.iloc[0].to_dict())
    if summaries:
        sdf = pd.DataFrame(summaries)
        metric = "success_rate" if "success_rate" in sdf and sdf["success_rate"].notna().any() else "mean_action_l1"
        plt.figure(figsize=(8, 5))
        plt.bar(sdf["run_name"], sdf[metric])
        plt.ylabel(metric)
        plt.title("Zero-Shot Evaluation on Environment D")
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        plt.savefig(output / "eval_D_bar.png", dpi=200)
        plt.close()

    plt.figure(figsize=(8, 5))
    any_dist = False
    for run in runs:
        df = _read_csv(run / "metrics" / "eval_D_episodes.csv")
        if df is not None and not df.empty:
            any_dist = True
            plt.hist(df["action_l1"], bins=24, alpha=0.55, label=run.name)
    if any_dist:
        plt.xlabel("Action L1")
        plt.ylabel("Count")
        plt.title("Action L1 Distribution on Environment D")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output / "action_l1_distribution.png", dpi=200)
    plt.close()
