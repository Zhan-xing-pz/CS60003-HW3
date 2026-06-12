from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hw3_act.lerobot_utils import ENV_TO_PART, ENV_TO_REPO_ID, HF_REPO_ID, project_env


META_FILES = [
    "episodes.jsonl",
    "episodes_stats.jsonl",
    "info.json",
    "modality.json",
    "stats.json",
    "tasks.jsonl",
]


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _download_file(repo_file: str, cache_root: Path, dst: Path, retries: int = 8) -> str:
    from huggingface_hub import hf_hub_download

    if dst.exists() and dst.stat().st_size > 0:
        return "cached"
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            src = Path(
                hf_hub_download(
                    repo_id=HF_REPO_ID,
                    repo_type="dataset",
                    filename=repo_file,
                    local_dir=cache_root,
                    resume_download=True,
                )
            )
            _link_or_copy(src, dst)
            return "downloaded"
        except Exception as exc:  # noqa: BLE001 - network mirrors raise several transient exception classes.
            last_error = exc
            wait_s = min(300.0, 5.0 * (2**attempt))
            message = str(exc)
            if "429" in message or "Too Many Requests" in message:
                wait_s = max(wait_s, 60.0)
            print(
                f"[retry {attempt + 1}/{retries}] {repo_file}: {type(exc).__name__}: {message[:200]} "
                f"(sleep {wait_s:.0f}s)",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(wait_s)
    raise RuntimeError(f"failed to download {repo_file} after {retries} retries") from last_error


def _download_one_with_retry(args: tuple[str, Path, Path]) -> str:
    repo_file, cache_root, dst = args
    return _download_file(repo_file, cache_root, dst)


def _download_many(files: list[tuple[str, Path]], cache_root: Path, desc: str, workers: int) -> dict[str, int]:
    counts = {"cached": 0, "downloaded": 0}
    pending = list(files)
    round_index = 0
    max_rounds = 6
    active_workers = max(1, workers)
    while pending and round_index < max_rounds:
        round_index += 1
        failed: list[tuple[str, Path]] = []
        round_desc = desc if round_index == 1 else f"{desc} retry-{round_index}"
        round_counts = _download_many_once(pending, cache_root, round_desc, active_workers, failed)
        counts["cached"] += round_counts["cached"]
        counts["downloaded"] += round_counts["downloaded"]
        pending = failed
        if pending:
            active_workers = max(1, active_workers // 2)
            wait_s = min(600, 60 * round_index)
            print(
                f"{desc}: {len(pending)} files still failed after round {round_index}; "
                f"sleep {wait_s}s and retry with workers={active_workers}",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(wait_s)
    if pending:
        preview = ", ".join(repo_file for repo_file, _ in pending[:5])
        raise RuntimeError(f"{desc}: failed to download {len(pending)} files after {max_rounds} rounds: {preview}")
    return counts


def _download_many_once(
    files: list[tuple[str, Path]],
    cache_root: Path,
    desc: str,
    workers: int,
    failed: list[tuple[str, Path]],
) -> dict[str, int]:
    counts = {"cached": 0, "downloaded": 0}
    if workers <= 1:
        for repo_file, dst in tqdm(files, desc=desc):
            try:
                counts[_download_file(repo_file, cache_root, dst)] += 1
            except Exception as exc:  # noqa: BLE001 - keep long dataset jobs resumable.
                print(f"{desc}: failed {repo_file}: {exc}", file=sys.stderr, flush=True)
                failed.append((repo_file, dst))
        return counts

    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_file = {
            pool.submit(_download_one_with_retry, (repo_file, cache_root, dst)): repo_file
            for repo_file, dst in files
        }
        for future in tqdm(as_completed(future_to_file), total=len(future_to_file), desc=desc):
            repo_file = future_to_file[future]
            try:
                counts[future.result()] += 1
            except Exception as exc:  # noqa: BLE001 - retry failed files as a later batch.
                print(f"{desc}: failed {repo_file}: {exc}", file=sys.stderr, flush=True)
                dst = next(dst for file_name, dst in files if file_name == repo_file)
                failed.append((repo_file, dst))
    return counts


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def download_env(
    env_name: str,
    output_root: Path,
    cache_root: Path,
    max_episodes: int | None,
    workers: int,
) -> Path:
    part = ENV_TO_PART[env_name]
    source_prefix = f"calvin_task_ABC_D_lerobot_{part}_4"
    repo_id = ENV_TO_REPO_ID[env_name]
    dst_root = output_root / repo_id
    dst_root.mkdir(parents=True, exist_ok=True)

    meta_files = [
        (f"{source_prefix}/meta/{name}", dst_root / "meta" / name)
        for name in META_FILES
    ]
    meta_counts = _download_many(meta_files, cache_root, f"{env_name} meta", min(workers, len(meta_files)))

    info = json.loads((dst_root / "meta" / "info.json").read_text(encoding="utf-8"))
    episodes = _read_jsonl(dst_root / "meta" / "episodes.jsonl")
    if max_episodes is not None:
        episodes = episodes[:max_episodes]

    chunk_size = int(info["chunks_size"])
    video_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    files: list[tuple[str, Path]] = []
    for ep in episodes:
        episode_index = int(ep["episode_index"])
        episode_chunk = episode_index // chunk_size
        data_rel = info["data_path"].format(
            episode_chunk=episode_chunk,
            episode_index=episode_index,
        )
        files.append((f"{source_prefix}/{data_rel}", dst_root / data_rel))
        for video_key in video_keys:
            video_rel = info["video_path"].format(
                video_key=video_key,
                episode_chunk=episode_chunk,
                episode_index=episode_index,
            )
            files.append((f"{source_prefix}/{video_rel}", dst_root / video_rel))

    file_counts = _download_many(files, cache_root, f"{env_name} data/videos", workers)

    audit = {
        "environment": env_name,
        "source_prefix": source_prefix,
        "repo_id": repo_id,
        "root": str(dst_root),
        "episodes_downloaded": len(episodes),
        "files_downloaded_or_present": len(files) + len(META_FILES),
        "meta_counts": meta_counts,
        "file_counts": file_counts,
        "video_keys": video_keys,
        "max_episodes": max_episodes,
        "workers": workers,
    }
    (dst_root / "download_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)
    return dst_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Download CALVIN A/B/C/D LeRobot v2.1 data into project.")
    parser.add_argument("--envs", nargs="+", default=["A", "B", "C", "D"], choices=["A", "B", "C", "D"])
    parser.add_argument("--output-root", default="data/lerobot_v21")
    parser.add_argument("--cache-root", default="data/raw/calvin_task_ABC_D")
    parser.add_argument("--max-episodes", type=int, default=None, help="Small subset for smoke tests.")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel downloads.")
    args = parser.parse_args()

    os.environ.update(project_env())
    output_root = ROOT / args.output_root
    cache_root = ROOT / args.cache_root
    cache_root.mkdir(parents=True, exist_ok=True)
    for env_name in args.envs:
        download_env(env_name, output_root, cache_root, args.max_episodes, args.workers)


if __name__ == "__main__":
    main()
