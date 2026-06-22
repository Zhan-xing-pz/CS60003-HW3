from __future__ import annotations

import math
import json
import random
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import resolve_device, save_config, set_seed
from .dataset import build_datasets
from .io_utils import append_csv, ensure_dir, write_json
from .eval import evaluate_lerobot_checkpoint_metrics
from .model import build_model
from .lerobot_utils import LEROBOT_TRAIN, project_env, run_output_dir


def _batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def _loss(pred: torch.Tensor, target: torch.Tensor, name: str) -> torch.Tensor:
    if name == "l1":
        return torch.nn.functional.l1_loss(pred, target)
    if name == "mse":
        return torch.nn.functional.mse_loss(pred, target)
    raise ValueError(f"Unsupported loss: {name}")


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_name: str,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int | None,
    desc: str,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_l1 = 0.0
    total_items = 0
    iterator = tqdm(loader, desc=desc, leave=False)
    for batch_idx, batch in enumerate(iterator, start=1):
        if max_batches is not None and batch_idx > max_batches:
            break
        batch = _batch_to_device(batch, device)
        with torch.set_grad_enabled(training):
            pred = model(batch["state"])
            loss = _loss(pred, batch["action"], loss_name)
            l1 = torch.nn.functional.l1_loss(pred, batch["action"])
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        n = batch["state"].shape[0]
        total_loss += float(loss.detach().cpu()) * n
        total_l1 += float(l1.detach().cpu()) * n
        total_items += n
        iterator.set_postfix(loss=total_loss / max(total_items, 1), action_l1=total_l1 / max(total_items, 1))
    return {
        "loss": total_loss / max(total_items, 1),
        "action_l1": total_l1 / max(total_items, 1),
        "num_samples": float(total_items),
    }


def train_from_config(cfg: dict[str, Any]) -> Path:
    if cfg.get("data", {}).get("mode") == "lerobot":
        return train_lerobot_from_config(cfg)

    seed = int(cfg["run"].get("seed", 42))
    set_seed(seed)
    run_name = cfg["run"]["name"]
    out_dir = ensure_dir(Path(cfg["run"].get("output_root", "outputs")) / run_name)
    ckpt_dir = ensure_dir(out_dir / "checkpoints")
    metrics_dir = ensure_dir(out_dir / "metrics")
    for stale_metric in ("train_metrics.csv", "valid_metrics.csv"):
        stale_path = metrics_dir / stale_metric
        if stale_path.exists():
            stale_path.unlink()
    save_config(cfg, out_dir / "config_resolved.yaml")

    device = resolve_device(cfg["train"].get("device", "auto"))
    train_ds, valid_ds = build_datasets(cfg["data"], cfg["model"], seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"].get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["train"].get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )

    model = build_model(cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["learning_rate"]),
        weight_decay=float(cfg["train"].get("weight_decay", 0.0)),
    )

    best_l1 = math.inf
    best_epoch = -1
    started = time.time()
    epochs = int(cfg["train"]["epochs"])
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            cfg["train"].get("loss", "l1"),
            optimizer,
            cfg["train"].get("max_train_batches"),
            desc=f"{run_name} train {epoch}/{epochs}",
        )
        valid_metrics = run_epoch(
            model,
            valid_loader,
            device,
            cfg["train"].get("loss", "l1"),
            None,
            cfg["train"].get("max_valid_batches"),
            desc=f"{run_name} valid {epoch}/{epochs}",
        )
        train_row = {"epoch": epoch, **train_metrics}
        valid_row = {"epoch": epoch, **valid_metrics}
        append_csv(metrics_dir / "train_metrics.csv", train_row)
        append_csv(metrics_dir / "valid_metrics.csv", valid_row)

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": cfg,
            "valid_action_l1": valid_metrics["action_l1"],
        }
        torch.save(ckpt, ckpt_dir / "last.pt")
        if valid_metrics["action_l1"] < best_l1:
            best_l1 = valid_metrics["action_l1"]
            best_epoch = epoch
            torch.save(ckpt, ckpt_dir / "best.pt")

        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.6f} "
            f"valid_action_l1={valid_metrics['action_l1']:.6f} best={best_l1:.6f}"
        )

    summary = {
        "run_name": run_name,
        "output_dir": str(out_dir),
        "best_epoch": best_epoch,
        "best_valid_action_l1": best_l1,
        "epochs": epochs,
        "train_samples": len(train_ds),
        "valid_samples": len(valid_ds),
        "device": str(device),
        "duration_sec": round(time.time() - started, 3),
    }
    write_json(out_dir / "run_summary.json", summary)
    return out_dir


def train_lerobot_from_config(cfg: dict[str, Any]) -> Path:
    out_dir = run_output_dir(cfg)
    # LeRobot refuses to start when output_dir already exists unless resuming.
    # Let lerobot-train own the run directory, then write our extra artifacts.
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    data_cfg = cfg["data"]
    split = _build_lerobot_episode_split(cfg)

    cmd = [
        str(LEROBOT_TRAIN),
        "--policy.type=act",
        f"--dataset.repo_id={data_cfg['train_repo_id']}",
        f"--dataset.root={data_cfg['train_root']}",
        f"--dataset.episodes={json.dumps(split['train_episodes'], separators=(',', ':'))}",
        f"--output_dir={out_dir}",
        f"--job_name={cfg['run']['name']}",
        f"--seed={cfg['run'].get('seed', 42)}",
        f"--steps={int(train_cfg['steps'])}",
        f"--batch_size={int(train_cfg['batch_size'])}",
        f"--num_workers={int(train_cfg.get('num_workers', 0))}",
        f"--log_freq={int(train_cfg.get('log_every', 20))}",
        f"--save_freq={int(train_cfg.get('save_freq', 1000))}",
        "--save_checkpoint=true",
        f"--tolerance_s={float(train_cfg.get('tolerance_s', 0.02))}",
        "--wandb.enable=false",
        f"--wandb.mode={train_cfg.get('wandb_mode', 'offline')}",
        "--wandb.project=hw3_lerobot_act",
        f"--policy.device={train_cfg.get('device', 'cuda')}",
        "--policy.push_to_hub=false",
        f"--policy.chunk_size={int(model_cfg['chunk_size'])}",
        f"--policy.n_action_steps={int(model_cfg.get('n_action_steps', model_cfg['chunk_size']))}",
        f"--policy.dim_model={int(model_cfg['hidden_dim'])}",
        f"--policy.n_heads={int(model_cfg['num_heads'])}",
        f"--policy.n_encoder_layers={int(model_cfg['num_layers'])}",
        f"--policy.n_decoder_layers={int(model_cfg['num_layers'])}",
        f"--policy.dropout={float(model_cfg.get('dropout', 0.1))}",
        f"--policy.optimizer_lr={float(train_cfg['learning_rate'])}",
        f"--policy.optimizer_weight_decay={float(train_cfg.get('weight_decay', 0.0))}",
    ]
    if int(train_cfg.get("eval_freq", 0)) > 0:
        cmd.append(f"--eval_freq={int(train_cfg['eval_freq'])}")
    else:
        cmd.append("--eval_freq=0")

    if out_dir.exists():
        has_checkpoint = any(out_dir.glob("**/*.safetensors")) or any(out_dir.glob("**/*.pt"))
        if not has_checkpoint:
            backup = out_dir.with_name(f"{out_dir.name}_failed_{int(time.time())}")
            out_dir.rename(backup)
            print(f"Moved stale LeRobot output directory to {backup}", flush=True)

    print("Running LeRobot training command:", flush=True)
    print(" ".join(cmd), flush=True)
    command_log_dir = out_dir.parent / "_command_logs"
    command_log_dir.mkdir(parents=True, exist_ok=True)
    (command_log_dir / f"{out_dir.name}_train_command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
    split_dir = ensure_dir(Path("data") / "splits")
    split_path = split_dir / f"{out_dir.name}_episode_split.json"
    write_json(split_path, split)
    log_path = command_log_dir / f"{out_dir.name}_train.log"
    with log_path.open("w", encoding="utf-8", errors="replace") as log_f:
        proc = subprocess.Popen(
            cmd,
            cwd=Path(__file__).resolve().parents[2],
            env=project_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        reader_thread = threading.Thread(target=_stream_process_output, args=(proc, log_f), daemon=True)
        reader_thread.start()
        _watch_lerobot_checkpoints(cfg, out_dir, proc, split)
        return_code = proc.wait()
        reader_thread.join(timeout=30)
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)
    _watch_lerobot_checkpoints(cfg, out_dir, proc, split, final_pass=True)
    (out_dir / "train_command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
    (out_dir / "train.log").write_text(log_path.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
    save_config(cfg, out_dir / "config_resolved.yaml")
    write_json(
        out_dir / "run_summary.json",
        {
            "run_name": cfg["run"]["name"],
            "output_dir": str(out_dir),
            "train_repo_id": data_cfg["train_repo_id"],
            "train_root": data_cfg["train_root"],
            "steps": int(train_cfg["steps"]),
            "batch_size": int(train_cfg["batch_size"]),
            "best_checkpoint": str(out_dir / "checkpoints" / "best" / "pretrained_model"),
            "checkpoint_valid_batches": int(train_cfg.get("checkpoint_valid_batches", train_cfg.get("checkpoint_eval_batches", 20))),
            "episode_split": str(split_path),
            "log_path": str(log_path),
        },
    )
    return out_dir


def _read_total_episodes(root: str | Path) -> int:
    with (Path(root) / "meta" / "info.json").open("r", encoding="utf-8") as f:
        return int(json.load(f)["total_episodes"])


def _infer_lerobot_segments(data_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    if "validation_segments" in data_cfg:
        return data_cfg["validation_segments"]

    train_root = Path(data_cfg["train_root"])
    if train_root.name.endswith("ABC"):
        base = train_root.parent
        counts = [
            _read_total_episodes(base / "calvin_A"),
            _read_total_episodes(base / "calvin_B"),
            _read_total_episodes(base / "calvin_C"),
        ]
        starts = [0, counts[0], counts[0] + counts[1]]
        return [
            {"name": "A", "start": starts[0], "count": counts[0]},
            {"name": "B", "start": starts[1], "count": counts[1]},
            {"name": "C", "start": starts[2], "count": counts[2]},
        ]

    return [{"name": train_root.name, "start": 0, "count": _read_total_episodes(train_root)}]


def _build_lerobot_episode_split(cfg: dict[str, Any]) -> dict[str, Any]:
    data_cfg = cfg["data"]
    seed = int(cfg["run"].get("seed", 42))
    fraction = float(data_cfg.get("validation_fraction", 0.1))
    rng = random.Random(seed)
    train_episodes: list[int] = []
    valid_episodes: list[int] = []
    split_segments: list[dict[str, Any]] = []

    for segment in _infer_lerobot_segments(data_cfg):
        name = str(segment["name"])
        start = int(segment["start"])
        count = int(segment["count"])
        episode_ids = list(range(start, start + count))
        shuffled = episode_ids[:]
        rng.shuffle(shuffled)
        valid_count = max(1, round(count * fraction))
        segment_valid = sorted(shuffled[:valid_count])
        segment_train = sorted(shuffled[valid_count:])
        valid_episodes.extend(segment_valid)
        train_episodes.extend(segment_train)
        split_segments.append(
            {
                "name": name,
                "start": start,
                "count": count,
                "train_count": len(segment_train),
                "valid_count": len(segment_valid),
            }
        )

    return {
        "run_name": cfg["run"]["name"],
        "train_repo_id": data_cfg["train_repo_id"],
        "train_root": data_cfg["train_root"],
        "validation_repo_id": data_cfg["train_repo_id"],
        "validation_root": data_cfg["train_root"],
        "validation_fraction": fraction,
        "seed": seed,
        "segments": split_segments,
        "train_episodes": sorted(train_episodes),
        "valid_episodes": sorted(valid_episodes),
    }


def _stream_process_output(proc: subprocess.Popen[str], log_f: Any) -> None:
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        log_f.write(line)
        log_f.flush()


def _checkpoint_step(path: Path) -> int:
    try:
        return int(path.name)
    except ValueError:
        return -1


def _is_complete_lerobot_checkpoint(path: Path, min_age_s: float = 5.0) -> bool:
    pretrained = path / "pretrained_model"
    model_file = pretrained / "model.safetensors"
    if not model_file.is_file() or model_file.stat().st_size == 0:
        return False
    if time.time() - model_file.stat().st_mtime < min_age_s:
        return False
    return any(pretrained.glob("*.json")) or any(pretrained.glob("*.yaml"))


def _load_best_record(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    import json

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _copy_best_checkpoint(source_pretrained: Path, best_pretrained: Path) -> None:
    tmp = best_pretrained.parent / f".{best_pretrained.name}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(source_pretrained, tmp)
    if best_pretrained.exists():
        shutil.rmtree(best_pretrained)
    tmp.rename(best_pretrained)


def _watch_lerobot_checkpoints(
    cfg: dict[str, Any],
    out_dir: Path,
    proc: subprocess.Popen[str],
    split: dict[str, Any],
    final_pass: bool = False,
) -> None:
    train_cfg = cfg["train"]
    if not bool(train_cfg.get("keep_only_best_checkpoint", True)):
        if not final_pass:
            proc.wait()
        return

    ckpt_root = out_dir / "checkpoints"
    valid_csv = out_dir / "metrics" / "checkpoint_valid_metrics.csv"
    best_dir = ckpt_root / "best"
    best_pretrained = best_dir / "pretrained_model"
    best_record_path = best_dir / "best_checkpoint.json"
    processed: set[str] = set()
    best_record = _load_best_record(best_record_path)
    best_l1 = float(best_record["mean_action_l1"]) if best_record else math.inf
    valid_batches = int(train_cfg.get("checkpoint_valid_batches", train_cfg.get("checkpoint_eval_batches", 20)))
    poll_s = float(train_cfg.get("checkpoint_poll_s", 10))

    while True:
        if not ckpt_root.exists():
            if final_pass or proc.poll() is not None:
                break
            time.sleep(poll_s)
            continue

        numeric_unprocessed = [
            p for p in ckpt_root.glob("[0-9][0-9][0-9][0-9][0-9][0-9]") if p.name not in processed
        ]
        candidates = sorted(numeric_unprocessed, key=_checkpoint_step)
        for ckpt_dir in candidates:
            if not _is_complete_lerobot_checkpoint(ckpt_dir):
                continue
            step = _checkpoint_step(ckpt_dir)
            pretrained = ckpt_dir / "pretrained_model"
            print(
                f"\nEvaluating checkpoint {ckpt_dir.name} on validation split ({valid_batches} batches)...",
                flush=True,
            )
            try:
                summary, _ = evaluate_lerobot_checkpoint_metrics(
                    cfg,
                    pretrained,
                    max_batches=valid_batches,
                    desc=f"{cfg['run']['name']} ckpt {ckpt_dir.name} valid",
                    repo_id=split["validation_repo_id"],
                    root=split["validation_root"],
                    episodes=split["valid_episodes"],
                    split_name="validation",
                )
            except Exception as exc:
                print(f"Checkpoint {ckpt_dir.name} evaluation failed: {exc}", flush=True)
                continue

            mean_l1 = float(summary["mean_action_l1"])
            is_best = mean_l1 < best_l1
            row = {
                "step": step,
                "checkpoint": str(pretrained),
                "mean_action_l1": mean_l1,
                "std_action_l1": float(summary["std_action_l1"]),
                "eval_batches": int(summary["eval_batches"]),
                "split": "validation",
                "is_best": bool(is_best),
                "best_mean_action_l1": float(mean_l1 if is_best else best_l1),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            ensure_dir(valid_csv.parent)
            append_csv(valid_csv, row)

            if is_best:
                best_l1 = mean_l1
                ensure_dir(best_dir)
                _copy_best_checkpoint(pretrained, best_pretrained)
                write_json(
                    best_record_path,
                    {
                        "source_step": step,
                        "source_checkpoint": str(pretrained),
                        "best_checkpoint": str(best_pretrained),
                        "mean_action_l1": mean_l1,
                        "std_action_l1": float(summary["std_action_l1"]),
                        "eval_batches": int(summary["eval_batches"]),
                        "selection_split": "validation",
                        "selection_metric": "validation_mean_action_l1",
                        "selection_mode": "lower_is_better",
                        "validation_repo_id": split["validation_repo_id"],
                        "validation_root": split["validation_root"],
                        "num_validation_episodes": len(split["valid_episodes"]),
                        "timestamp": row["timestamp"],
                    },
                )
                print(f"New best checkpoint at step {step}: mean_action_l1={mean_l1:.6f}", flush=True)
            else:
                print(f"Discarding checkpoint {step}: mean_action_l1={mean_l1:.6f}, best={best_l1:.6f}", flush=True)

            processed.add(ckpt_dir.name)
            shutil.rmtree(ckpt_dir)

        if final_pass or proc.poll() is not None:
            numeric_unprocessed = [
                p for p in ckpt_root.glob("[0-9][0-9][0-9][0-9][0-9][0-9]") if p.name not in processed
            ]
            if not numeric_unprocessed:
                break
            if not any(_is_complete_lerobot_checkpoint(p) for p in numeric_unprocessed):
                time.sleep(poll_s)
                continue
        time.sleep(poll_s)
