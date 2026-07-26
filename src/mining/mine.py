#!/usr/bin/env python3
"""Mine BM25/dense negatives or score teacher candidates."""

import argparse
import json

from utils.config import load_config
from mining.bm25 import mine_bm25_hard_negatives
from mining.dense import mine_dense_hard_negatives
from mining.distill import score_teacher_candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--config",
        default="src/configs/embed/mine_bm25.yaml",
    )

    args = parser.parse_args()
    config = load_config(args.config)

    method = str(config.get("method", "")).strip().casefold()

    if method == "bm25":
        report = mine_bm25_hard_negatives(config)

    elif method == "dense":
        report = mine_dense_hard_negatives(config)

    elif method == "teacher_distill":
        report = score_teacher_candidates(config)

    else:
        parser.error(
            "Unknown mining method. Expected one of: "
            "bm25, dense, teacher_distill"
        )

    # BM25 and dense reports contain `statistics`; teacher scoring currently
    # returns its summary directly.
    output = report.get("statistics", report)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()