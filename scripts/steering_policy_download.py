"""Download the pinned inference release, reusing SHA256-verified existing large files.

Run with ~/steering_policy_real/pi05_env/bin/python. Training optimizer states are excluded.
Existing legacy checkpoints/config edits are preserved in their original directories.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import tyro
from huggingface_hub import HfApi, snapshot_download
from steering_policy_models import LOCAL_ROOT, MODELS, REPO_ID, REVISION, TASKS, resolve_assets


@dataclass
class Args:
    root: Path = LOCAL_ROOT
    revision: str = REVISION
    workers: int = 4


def main(args: Args) -> None:
    root = args.root.expanduser().resolve()
    info = HfApi().model_info(REPO_ID, revision=args.revision, files_metadata=True)
    target = root / "releases" / info.sha
    files = [
        s
        for s in info.siblings
        if (
            s.rfilename in ("PI05_STEERING.md", "STEERING.md")
            or s.rfilename.startswith("deployment/")
            or any(s.rfilename.startswith(f"{task}/") for task in TASKS)
        )
        and "/training_state/" not in s.rfilename
    ]
    verified: dict[str, Path] = {}
    releases = root / "releases"
    reuse_roots = [root] + (
        sorted((p for p in releases.iterdir() if p.is_dir() and p != target), reverse=True)
        if releases.is_dir()
        else []
    )
    # Reuse only immutable large binaries. JSON configs may have intentional local edits.
    for entry in files:
        if not entry.lfs or entry.size < 1_000_000:
            continue
        dest = target / entry.rfilename
        if dest.exists():
            continue
        sha = entry.lfs.sha256
        for reuse_root in reuse_roots:
            if sha in verified:
                break
            src = reuse_root / entry.rfilename
            if src.is_file() and src.stat().st_size == entry.size:
                print(f"[verify] {src}", flush=True)
                with src.open("rb") as stream:
                    actual = hashlib.file_digest(stream, "sha256").hexdigest()
                if actual == sha:
                    verified[sha] = src
        if sha in verified:
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.link(verified[sha], dest)
            print(f"[reuse] {entry.rfilename}", flush=True)
    snapshot_download(
        REPO_ID,
        revision=info.sha,
        local_dir=target,
        allow_patterns=[s.rfilename for s in files],
        max_workers=args.workers,
    )
    report: dict[str, object] = {"repo_id": REPO_ID, "revision": info.sha, "root": str(target), "models": {}}
    for task in TASKS:
        for model in MODELS:
            try:
                resolve_assets(task, model, target)
                status = "ready"
            except FileNotFoundError as exc:
                status = str(exc)
            report["models"][f"{task}/{model}"] = status
            print(f"[assets] {task}/{model}: {status}")
    (target / "download_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"[done] {target}")


if __name__ == "__main__":
    main(tyro.cli(Args))
