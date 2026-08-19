"""Upload recorded YAM episodes to a Hugging Face dataset repo.

Reads the layout written by scripts/bimanual_teleop_record.py

    <data_root>/<task>/episode_0000/{top.mp4, left_wrist.mp4, right_wrist.mp4, low_dim.npz, meta.json}

and pushes selected episodes to <repo_id> (repo_type=dataset), one commit per episode, at

    <repo_prefix>/<task>/episode_0000/...

Episodes are selected explicitly -- nothing is uploaded unless you say which ones. --episodes
accepts indices, ranges and full directory names; --all takes every complete episode of the task.
An episode counts as complete only if meta.json is present (the recorder writes it last), so a
half-written episode is never pushed. discarded/ is ignored.

Auth: --token, else $HF_TOKEN / $HUGGINGFACE_HUB_TOKEN, else the cached CLI login
(`hf auth login`). Never hard-code a token in this file.

Usage:
    python scripts/upload_yam_data_hf.py --list                          # what is on disk
    python scripts/upload_yam_data_hf.py --episodes 0,2,4-6              # upload a selection
    python scripts/upload_yam_data_hf.py --episodes 3 --dry-run          # show the plan only
    python scripts/upload_yam_data_hf.py --all --skip-existing           # resume a partial upload
    python scripts/upload_yam_data_hf.py --all --ignore-patterns '*.mp4' # low_dim + meta only

Requires: pip install huggingface_hub   (or: uv run --with huggingface_hub python scripts/...)
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set

import tyro

META_FILE = "meta.json"


@dataclass
class Args:
    repo_id: str = "ChongkaiGao/planning"
    """Target Hugging Face dataset repo, "<user>/<name>"."""
    data_root: str = "~/yam_data"
    """Local recording root (matches --save_root of bimanual_teleop_record.py)."""
    task: str = "default_task"
    """Task directory under data_root."""
    episodes: str = ""
    """Which episodes to upload: "0,2,4-6", "episode_0003", or a mix. Empty means none."""
    all: bool = False
    """Upload every complete episode of the task (ignores --episodes)."""
    repo_prefix: str = "yam_data"
    """Path prefix inside the repo; files land at <repo_prefix>/<task>/episode_NNNN/. "" = repo root."""
    token: Optional[str] = None
    """HF write token. Falls back to $HF_TOKEN, $HUGGINGFACE_HUB_TOKEN, then the cached login."""
    private: bool = False
    """If the repo has to be created, create it private."""
    skip_existing: bool = False
    """Skip episodes whose files are already all present in the repo (cheap resume)."""
    ignore_patterns: List[str] = field(default_factory=list)
    """Glob patterns to exclude from each episode, e.g. '*.mp4'."""
    revision: str = "main"
    """Branch to commit to."""
    list: bool = False
    """Print the local episodes and exit."""
    dry_run: bool = False
    """Print what would be uploaded, then exit without touching the network."""


def _task_dir(args: Args) -> Path:
    return Path(args.data_root).expanduser() / args.task


def _find_episodes(task_dir: Path) -> List[Path]:
    """Complete episode dirs (meta.json present), sorted by name; discarded/ excluded."""
    if not task_dir.is_dir():
        sys.exit(f"[error] task directory not found: {task_dir}")
    return sorted(p for p in task_dir.glob("episode_*") if p.is_dir() and (p / META_FILE).is_file())


def _parse_selection(spec: str) -> Set[str]:
    """Turn "0,2,4-6,episode_0009" into {"episode_0000", "episode_0002", ...}."""
    names: Set[str] = set()
    for raw in spec.replace(" ", "").split(","):
        if not raw:
            continue
        if raw.startswith("episode_"):
            names.add(raw)
        elif "-" in raw.strip("-"):
            lo_s, hi_s = raw.split("-", 1)
            try:
                lo, hi = int(lo_s), int(hi_s)
            except ValueError:
                sys.exit(f"[error] bad episode range: {raw!r}")
            if hi < lo:
                sys.exit(f"[error] empty episode range: {raw!r}")
            names.update(f"episode_{i:04d}" for i in range(lo, hi + 1))
        else:
            try:
                names.add(f"episode_{int(raw):04d}")
            except ValueError:
                sys.exit(f"[error] bad episode selector: {raw!r}")
    return names


def _select(args: Args, available: List[Path]) -> List[Path]:
    if args.all:
        return available
    if not args.episodes:
        sys.exit("[error] nothing selected -- pass --episodes 0,2,4-6 (or --all). Use --list to see them.")
    wanted = _parse_selection(args.episodes)
    by_name = {p.name: p for p in available}
    missing = sorted(wanted - by_name.keys())
    if missing:
        sys.exit(f"[error] not found (or incomplete -- no {META_FILE}): {', '.join(missing)}")
    return [by_name[n] for n in sorted(wanted)]


def _episode_files(ep: Path, ignore_patterns: List[str]) -> List[Path]:
    files = sorted(p for p in ep.rglob("*") if p.is_file())
    for pat in ignore_patterns:
        files = [p for p in files if not p.match(pat)]
    return files


def _human(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}GB"


def _resolve_token(args: Args) -> Optional[str]:
    return args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or None


def main(args: Args) -> None:
    task_dir = _task_dir(args)
    available = _find_episodes(task_dir)

    if args.list:
        print(f"[list] {task_dir}  ({len(available)} complete episode(s))")
        for ep in available:
            files = _episode_files(ep, args.ignore_patterns)
            total = sum(f.stat().st_size for f in files)
            print(f"  {ep.name}  {len(files)} file(s)  {_human(total)}")
        return

    selected = _select(args, available)
    prefix = args.repo_prefix.strip("/")
    plan = []
    for ep in selected:
        files = _episode_files(ep, args.ignore_patterns)
        if not files:
            print(f"[skip] {ep.name}: no files left after --ignore-patterns")
            continue
        path_in_repo = "/".join(p for p in (prefix, args.task, ep.name) if p)
        plan.append((ep, path_in_repo, files))

    if not plan:
        sys.exit("[error] nothing to upload")

    total_bytes = sum(f.stat().st_size for _, _, files in plan for f in files)
    print(f"[plan] {len(plan)} episode(s), {_human(total_bytes)} -> {args.repo_id} (dataset, {args.revision})")
    for ep, path_in_repo, files in plan:
        size = _human(sum(f.stat().st_size for f in files))
        print(f"  {ep} -> {path_in_repo}/  ({len(files)} file(s), {size})")

    if args.dry_run:
        print("[dry-run] nothing uploaded")
        return

    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("[error] huggingface_hub not installed -- pip install huggingface_hub")

    api = HfApi(token=_resolve_token(args))
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)

    remote: Set[str] = set()
    if args.skip_existing:
        remote = set(api.list_repo_files(repo_id=args.repo_id, repo_type="dataset", revision=args.revision))

    uploaded = 0
    for ep, path_in_repo, files in plan:
        if args.skip_existing:
            wanted = {f"{path_in_repo}/{f.relative_to(ep).as_posix()}" for f in files}
            if wanted <= remote:
                print(f"[skip] {ep.name}: already in {args.repo_id}")
                continue
        print(f"[upload] {ep.name} -> {path_in_repo}/ ...")
        info = api.upload_folder(
            repo_id=args.repo_id,
            repo_type="dataset",
            revision=args.revision,
            folder_path=str(ep),
            path_in_repo=path_in_repo,
            ignore_patterns=args.ignore_patterns or None,
            commit_message=f"Add {args.task}/{ep.name}",
        )
        uploaded += 1
        print(f"[ok] {ep.name}  {info.commit_url}")

    print(f"[done] {uploaded}/{len(plan)} episode(s) uploaded -> https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main(tyro.cli(Args))
