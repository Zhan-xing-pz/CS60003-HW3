from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hw3_act.config import load_config
from hw3_act.eval import evaluate_lerobot_checkpoint_metrics
from hw3_act.io_utils import append_csv, ensure_dir, write_json
from hw3_act.lerobot_utils import checkpoint_pretrained_dir, run_output_dir
from hw3_act.train_loop import _copy_best_checkpoint, _load_best_record


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one LeRobot checkpoint and keep it only if it is best.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--delete-candidate", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = run_output_dir(cfg)
    metrics_dir = ensure_dir(out_dir / "metrics")
    best_dir = out_dir / "checkpoints" / "best"
    best_pretrained = best_dir / "pretrained_model"
    best_record_path = best_dir / "best_checkpoint.json"
    pretrained = checkpoint_pretrained_dir(args.checkpoint)

    best_record = _load_best_record(best_record_path)
    best_l1 = float(best_record["mean_action_l1"]) if best_record else float("inf")
    summary, _ = evaluate_lerobot_checkpoint_metrics(
        cfg,
        pretrained,
        max_batches=args.eval_batches,
        desc=f"{cfg['run']['name']} ckpt {args.step:06d} eval_D",
    )

    mean_l1 = float(summary["mean_action_l1"])
    is_best = mean_l1 < best_l1
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    append_csv(
        metrics_dir / "checkpoint_eval_D.csv",
        {
            "step": args.step,
            "checkpoint": str(pretrained),
            "mean_action_l1": mean_l1,
            "std_action_l1": float(summary["std_action_l1"]),
            "eval_batches": int(summary["eval_batches"]),
            "is_best": bool(is_best),
            "best_mean_action_l1": float(mean_l1 if is_best else best_l1),
            "timestamp": timestamp,
        },
    )

    if is_best:
        ensure_dir(best_dir)
        _copy_best_checkpoint(pretrained, best_pretrained)
        write_json(
            best_record_path,
            {
                "source_step": args.step,
                "source_checkpoint": str(pretrained),
                "best_checkpoint": str(best_pretrained),
                "mean_action_l1": mean_l1,
                "std_action_l1": float(summary["std_action_l1"]),
                "eval_batches": int(summary["eval_batches"]),
                "selection_metric": "mean_action_l1",
                "selection_mode": "lower_is_better",
                "timestamp": timestamp,
            },
        )

    if args.delete_candidate:
        candidate_dir = pretrained.parent
        if candidate_dir.name != "best" and candidate_dir.parent.name == "checkpoints":
            import shutil

            shutil.rmtree(candidate_dir)

    print(f"step={args.step} mean_action_l1={mean_l1:.6f} is_best={is_best}")


if __name__ == "__main__":
    main()
