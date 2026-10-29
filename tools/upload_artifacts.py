from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload validated browser assets to a private Hugging Face repo")
    parser.add_argument("--folder", required=True)
    parser.add_argument("--repo", default="verysweatydog/nvg")
    parser.add_argument("--repo-type", choices=("model", "dataset"), default="model")
    args = parser.parse_args()
    api = HfApi()
    try:
        info = api.repo_info(args.repo, repo_type=args.repo_type)
    except Exception:
        info = api.create_repo(args.repo, repo_type=args.repo_type, private=True, exist_ok=True)
    if not getattr(info, "private", False):
        raise RuntimeError(f"Refusing to upload model artifacts to public repo {args.repo}")
    api.upload_folder(
        repo_id=args.repo,
        repo_type=args.repo_type,
        folder_path=Path(args.folder),
        commit_message="Add validated SID browser model and trained pseudo-RAW adapter",
    )
    prefix = "datasets/" if args.repo_type == "dataset" else ""
    print(f"UPLOAD_COMPLETE https://huggingface.co/{prefix}{args.repo}")


if __name__ == "__main__":
    main()
