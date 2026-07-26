#!/usr/bin/env python3
"""Evaluate a zero-shot lexical or dense retrieval model."""

from __future__ import annotations

import argparse
from pathlib import Path

from zero_shot import evaluate_bm25, evaluate_dense, write_result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="bm25 or a Hugging Face model name")
    parser.add_argument("--split", type=Path, default=Path("data/processed/embed/test"))
    parser.add_argument("--output", type=Path, default=Path("reports/experiments/embed/zero_shot"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    result = (evaluate_bm25(args.split) if args.model.casefold() == "bm25" else
              evaluate_dense(args.model, args.split, args.batch_size, args.device,
                             args.trust_remote_code))
    paths = write_result(result, args.output)
    print(f"Wrote {paths[0]} and {paths[1]}")
    print(result["metrics"])


if __name__ == "__main__":
    main()
