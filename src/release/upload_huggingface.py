#!/usr/bin/env python3
"""Create Hugging Face model repositories and upload Embed checkpoints."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


CHECKPOINTS = {
    "v1": Path("artifacts/checkpoints/embed/gte-modernbert-base/final"),
    "v2": Path("artifacts/checkpoints/embed/gte-modernbert-base-v2-bm25/final"),
    "v3": Path("artifacts/checkpoints/embed/gte-modernbert-base-v3-dense/final"),
    "v4": Path("artifacts/checkpoints/embed/gte-modernbert-base-v4-distillation/final"),
}

REPOSITORY_NAMES = {
    "v1": "legal-embed-gte-inbatch",
    "v2": "legal-embed-gte-bm25",
    "v3": "legal-embed-gte-dense",
    "v4": "legal-embed-gte-distilled",
}


def validate_checkpoint(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {path}")
    required = ("modules.json", "config_sentence_transformers.json")
    missing = [name for name in required if not (path / name).is_file()]
    weight_files = [
        *path.rglob("model.safetensors"),
        *path.rglob("pytorch_model.bin"),
    ]
    if missing or not weight_files:
        details = []
        if missing:
            details.append(f"missing metadata: {', '.join(missing)}")
        if not weight_files:
            details.append("no model.safetensors or pytorch_model.bin")
        raise ValueError(f"Invalid checkpoint {path}: {'; '.join(details)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--namespace",
        required=True,
        help="Hugging Face username or organisation.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=sorted(CHECKPOINTS),
        default=sorted(CHECKPOINTS),
        help="Model variants to upload (default: v1 v2 v3 v4).",
    )
    parser.add_argument(
        "--private",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create private repositories (default: true).",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Hub branch or revision to upload to (default: main).",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform uploads; without this flag the script is a dry run.",
    )
    args = parser.parse_args()

    plans = []
    for variant in args.variants:
        checkpoint = CHECKPOINTS[variant]
        validate_checkpoint(checkpoint)
        plans.append(
            (
                variant,
                checkpoint,
                f"{args.namespace}/{REPOSITORY_NAMES[variant]}",
            )
        )

    visibility = "private" if args.private else "public"
    for variant, checkpoint, repo_id in plans:
        print(
            f"[plan] {variant}: {checkpoint} -> {repo_id} "
            f"({visibility}, revision={args.revision})",
            flush=True,
        )

    if not args.execute:
        print(
            "[dry-run] Checkpoints are valid. Add --execute to create the "
            "repositories and upload them.",
            flush=True,
        )
        return

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "Install publishing dependencies with: "
            "python -m pip install -e '.[publish]'"
        ) from exc

    # HfApi also supports a token saved by `hf auth login`. Prefer HF_TOKEN on
    # clusters so credentials never appear in shell history or process args.
    token = os.environ.get("HF_TOKEN") or None
    api = HfApi(token=token)
    identity = api.whoami()
    print(f"[auth] logged in as {identity['name']}", flush=True)

    for variant, checkpoint, repo_id in plans:
        print(f"[upload] creating or reusing {repo_id}", flush=True)
        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=args.private,
            exist_ok=True,
        )
        commit = api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=checkpoint,
            path_in_repo="",
            revision=args.revision,
            commit_message=f"Upload Mini Kanon 3 Embed {variant.upper()} final model",
            ignore_patterns=[
                "**/.DS_Store",
                "**/__pycache__/**",
                "**/*.tmp",
            ],
        )
        print(f"[done] {variant}: {commit.repo_url}", flush=True)


if __name__ == "__main__":
    main()
