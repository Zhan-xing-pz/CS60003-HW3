from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from .io_utils import ensure_dir, write_json, write_jsonl


ENV_RE = re.compile(r"(?:env|environment|scene|calvin)[_\-/ ]*([ABCD])", re.IGNORECASE)


def infer_environment(text: str, row: dict[str, Any] | None = None) -> str | None:
    haystack = text
    if row:
        for key in ("environment", "env", "scene", "split", "task", "task_name"):
            value = row.get(key)
            if value is not None:
                haystack += f" {key}={value}"
    match = ENV_RE.search(haystack)
    if match:
        return match.group(1).upper()
    tokens = re.split(r"[^A-Za-z0-9]+", haystack)
    for token in tokens:
        if token.upper() in {"A", "B", "C", "D"}:
            return token.upper()
    return None


def _parquet_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.parquet")):
        env = infer_environment(str(path))
        num_rows = 0
        columns: list[str] = []
        error = ""
        try:
            df = pd.read_parquet(path)
            num_rows = int(len(df))
            columns = list(df.columns)
            if env is None and len(df) > 0:
                env = infer_environment(str(path), df.iloc[0].to_dict())
        except Exception as exc:  # pragma: no cover - audit path
            error = str(exc)
        records.append(
            {
                "path": str(path.resolve()),
                "environment": env or "UNKNOWN",
                "num_rows": num_rows,
                "columns": "|".join(columns[:60]),
                "error": error,
            }
        )
    return records


def _hf_records(repo_id: str) -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(repo_id, split="train")
    records: list[dict[str, Any]] = []
    columns = list(ds.column_names)
    for idx, row in enumerate(ds):
        env = infer_environment(json.dumps({k: str(row.get(k)) for k in columns[:20]}, ensure_ascii=True), row)
        records.append(
            {
                "hf_repo": repo_id,
                "row_index": idx,
                "environment": env or "UNKNOWN",
                "num_rows": 1,
                "columns": "|".join(columns[:60]),
                "error": "",
            }
        )
    return records


def prepare_splits(dataset_root: str, output_dir: str | Path) -> Path:
    out = ensure_dir(output_dir)
    root = Path(dataset_root)
    records = _parquet_records(root) if root.exists() else _hf_records(dataset_root)
    if not records:
        raise FileNotFoundError(f"No parquet records found under {dataset_root}")

    audit = pd.DataFrame(records)
    audit.to_csv(out / "split_audit.csv", index=False)

    known = [
        r
        for r in records
        if r["environment"] in {"A", "B", "C", "D"} and (r.get("path") or r.get("hf_repo"))
    ]
    train_a = [r for r in known if r["environment"] == "A"]
    train_abc = [r for r in known if r["environment"] in {"A", "B", "C"}]
    eval_d = [r for r in known if r["environment"] == "D"]
    write_jsonl(out / "train_A.jsonl", train_a)
    write_jsonl(out / "train_ABC.jsonl", train_abc)
    write_jsonl(out / "eval_D.jsonl", eval_d)

    summary = {
        "dataset_root": dataset_root,
        "total_records": len(records),
        "known_records": len(known),
        "train_A": len(train_a),
        "train_ABC": len(train_abc),
        "eval_D": len(eval_d),
        "unknown": int((audit["environment"] == "UNKNOWN").sum()),
    }
    write_json(out / "split_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return out
