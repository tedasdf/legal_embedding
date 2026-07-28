

from __future__ import annotations

import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from data.load import load_retrieval_split
from utils.utils import _read_jsonl, _write_jsonl
from mining.util import _sha256, _load_corpus_rows

def _load_jsonl_by_key(path, key): return {row[key]: row for row in _read_jsonl(path)}


def score_teacher_candidates(config: dict) -> dict:
    """Score one positive plus mined candidate negatives with a cross-encoder."""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RuntimeError("Teacher scoring requires sentence-transformers") from exc
    split_dir, mined_path = Path(config["input_split"]), Path(config["candidate_negatives_path"])
    queries, corpus, qrels = load_retrieval_split(split_dir)
    mined = _load_jsonl_by_key(mined_path, "query_id")
    negatives_per_query = int(config["negatives_per_query"])
    records, pairs, spans = [], [], []
    for query_id in sorted(queries):
        negative_ids = [item["passage_id"] for item in mined[query_id]["negative_passages"]]
        positive_id = sorted(qrels[query_id])[0]
        candidate_ids = [positive_id, *negative_ids[:negatives_per_query]]
        if len(candidate_ids) != negatives_per_query + 1:
            raise ValueError(f"Insufficient teacher candidates for {query_id}")
        start = len(pairs)
        pairs.extend((queries[query_id], corpus[pid]) for pid in candidate_ids)
        spans.append((start, len(pairs)))
        records.append({"query_id": query_id, "positive_passage_id": positive_id,
                        "candidate_passage_ids": candidate_ids})
    teacher = CrossEncoder(config["teacher_model"], device=config.get("device"),
                           trust_remote_code=bool(config.get("trust_remote_code", True)))
    scores = teacher.predict(pairs, batch_size=int(config.get("batch_size", 32)),
                             show_progress_bar=True, convert_to_numpy=True)
    output_rows = []
    for record, (start, end) in zip(records, spans):
        candidates = [{"passage_id": pid, "teacher_score": round(float(score), 8),
                       "is_positive": index == 0}
                      for index, (pid, score) in enumerate(zip(record["candidate_passage_ids"], scores[start:end]))]
        output_rows.append({"query_id": record["query_id"], "candidates": candidates})
    output_path, report_path = Path(config["output_path"]), Path(config["report_path"])
    _write_jsonl(output_path, output_rows)
    report = {"schema_version": 1, "method": "cross_encoder_teacher_scoring",
              "teacher_model": config["teacher_model"], "input_split": str(split_dir),
              "candidate_negatives": str(mined_path), "output": str(output_path),
              "queries": len(output_rows), "candidates_per_query": negatives_per_query + 1,
              "positive_position": 0}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
