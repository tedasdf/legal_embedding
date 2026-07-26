"""Zero-shot BM25 and dense retrieval evaluation orchestration."""

from __future__ import annotations

import json
import platform
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path

from data.load import load_retrieval_split
from model.bm25 import BM25
from .metrics import evaluate_rankings

tracemalloc.start()

def evaluate_bm25(split_dir: Path, k1=1.5, b=0.75):
    queries, corpus, qrels = load_retrieval_split(split_dir)
    started = time.perf_counter()
    model = BM25(corpus, k1=k1, b=b)
    build_seconds = time.perf_counter() - started
    started = time.perf_counter()
    rankings = {query_id: model.rank(text) for query_id, text in queries.items()}
    search_seconds = time.perf_counter() - started
    return _result("bm25", split_dir, queries, corpus, qrels, rankings, build_seconds,
                   search_seconds, None, {"k1": k1, "b": b})


def evaluate_dense(model_name: str, split_dir: Path, batch_size=16, device=None,
                   trust_remote_code=False):
    try:
        import numpy as np
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Install the 'eval' project dependencies for dense evaluation") from exc
    queries, corpus, qrels = load_retrieval_split(split_dir)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = SentenceTransformer(model_name, device=device, trust_remote_code=trust_remote_code)
    load_seconds = time.perf_counter() - started
    query_ids, passage_ids = list(queries), list(corpus)
    started = time.perf_counter()
    if hasattr(model, "encode_query"):
        query_vectors = model.encode_query([queries[key] for key in query_ids], batch_size=batch_size,
                                           normalize_embeddings=True, convert_to_numpy=True)
        passage_vectors = model.encode_document([corpus[key] for key in passage_ids], batch_size=batch_size,
                                                normalize_embeddings=True, convert_to_numpy=True)
    else:
        query_vectors = model.encode([queries[key] for key in query_ids], batch_size=batch_size,
                                     normalize_embeddings=True, convert_to_numpy=True)
        passage_vectors = model.encode([corpus[key] for key in passage_ids], batch_size=batch_size,
                                       normalize_embeddings=True, convert_to_numpy=True)
    encode_seconds = time.perf_counter() - started
    started = time.perf_counter()
    scores = np.matmul(query_vectors, passage_vectors.T)
    order = np.argsort(-scores, axis=1)
    rankings = {query_id: [passage_ids[index] for index in order[row]]
                for row, query_id in enumerate(query_ids)}
    search_seconds = time.perf_counter() - started
    extra = {"batch_size": batch_size, "device": str(model.device),
             "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 2)
             if torch.cuda.is_available() else None}
    return _result(model_name, split_dir, queries, corpus, qrels, rankings, load_seconds,
                   encode_seconds + search_seconds, int(query_vectors.shape[1]), extra)


def _result(model, split_dir, queries, corpus, qrels, rankings, setup_seconds,
            evaluation_seconds, dimension, parameters):
    per_query = {}
    for query_id in sorted(qrels):
        relevant = qrels[query_id]
        ranked = rankings.get(query_id, [])
        positive_ranks = [
            index + 1
            for index, passage_id in enumerate(ranked)
            if passage_id in relevant
        ]
        per_query[query_id] = {
            "positive_passage_ids": sorted(relevant),
            "best_positive_rank": min(positive_ranks) if positive_ranks else None,
            "top_10_passage_ids": ranked[:10],
        }
    return {
        "schema_version": 1, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": model, "split": str(split_dir), "queries": len(queries),
        "passages": len(corpus), "positive_pairs": sum(map(len, qrels.values())),
        "metrics": evaluate_rankings(rankings, qrels),
        "per_query": per_query,
        "efficiency": {"setup_seconds": round(setup_seconds, 6),
                       "evaluation_seconds": round(evaluation_seconds, 6),
                       "mean_query_latency_ms": round(evaluation_seconds * 1000 / len(queries), 6),
                       "representation_dimension": dimension,
                       "python_peak_memory_mb": round(tracemalloc.get_traced_memory()[1] / 1024**2, 2)},
        "parameters": parameters,
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
    }


def write_result(result: dict, output_dir: Path):
    safe_name = result["model"].replace("/", "__").replace(" ", "_").lower()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{safe_name}.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    metrics = result["metrics"]
    report = [f"# Zero-shot Embed evaluation: {result['model']}", "",
              f"- Split: `{result['split']}`", f"- Queries: {result['queries']}",
              f"- Passages: {result['passages']}", "", "## Metrics", "",
              "| Metric | Value |", "|---|---:|"]
    report += [f"| {name} | {value:.6f} |" for name, value in metrics.items()]
    report += ["", "## Efficiency", "", "```json",
               json.dumps(result["efficiency"], indent=2), "```", ""]
    md_path = output_dir / f"{safe_name}.md"
    md_path.write_text("\n".join(report), encoding="utf-8")
    return json_path, md_path
