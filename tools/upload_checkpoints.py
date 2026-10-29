"""Uploads a prepared folder of checkpoints to the private Hugging Face dataset repo.

Run on a RunPod pod created with HF_TOKEN injected from the RunPod secret. As
in training/runpod_prepare.sh, the token is read from the environment or, for
SSH sessions that do not inherit it, from the container's init environment.
It is never printed. Uploads are refused unless the repository is private.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def token() -> str:
    value = os.environ.get("HF_TOKEN", "")
    if not value and Path("/proc/1/environ").is_file():
        for item in Path("/proc/1/environ").read_bytes().split(b"\0"):
            if item.startswith(b"HF_TOKEN="):
                value = item.split(b"=", 1)[1].decode()
    if not value or "RUNPOD_SECRET" in value:
        raise SystemExit("HF_TOKEN is missing or the RunPod secret did not resolve")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--folder", required=True)
    parser.add_argument("--repo", default="verysweatydog/nvg")
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument("--path-in-repo", default="checkpoints")
    parser.add_argument("--message", default="Add SID Live checkpoints")
    args = parser.parse_args()
    from huggingface_hub import HfApi

    api = HfApi(token=token())
    info = api.repo_info(args.repo, repo_type=args.repo_type)
    if not info.private:
        raise SystemExit(f"Refusing to upload: {args.repo} ({args.repo_type}) is public")
    commit = api.upload_folder(repo_id=args.repo, repo_type=args.repo_type, folder_path=args.folder,
                               path_in_repo=args.path_in_repo, commit_message=args.message)
    files = sorted(entry for entry in api.list_repo_files(args.repo, repo_type=args.repo_type)
                   if entry.startswith(args.path_in_repo + "/"))
    print(f"UPLOAD_COMPLETE {commit.oid if hasattr(commit, 'oid') else commit} files={len(files)}")
    for entry in files:
        print("  " + entry)


if __name__ == "__main__":
    main()
