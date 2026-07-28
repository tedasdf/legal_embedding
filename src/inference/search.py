from __future__ import annotations

import argparse
import json
from pathlib import Path

from inference.retriever import LegalRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search the legal passage index"
    )

    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--passages", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    retriever = LegalRetriever(
        model_dir=args.model,
        embeddings_path=args.embeddings,
        passages_path=args.passages,
    )

    output = retriever.search(
        query=args.query,
        top_k=args.top_k,
    )

    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()