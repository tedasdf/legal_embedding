"""Retrieval metrics with no external dependencies."""

from __future__ import annotations

import math


def evaluate_rankings(rankings: dict[str, list[str]], positives: dict[str, set[str]]) -> dict[str, float]:
    """Evaluate one ranked passage list per query against binary qrels."""
    if not positives:
        raise ValueError("No qrels were supplied")
    totals = {"ndcg_at_10": 0.0, "recall_at_1": 0.0, "recall_at_5": 0.0,
              "recall_at_10": 0.0, "mrr": 0.0}
    for query_id, relevant in positives.items():
        ranked = rankings.get(query_id, [])
        for k in (1, 5, 10):
            totals[f"recall_at_{k}"] += len(set(ranked[:k]) & relevant) / len(relevant)
        dcg = sum(1.0 / math.log2(rank + 2) for rank, passage_id in enumerate(ranked[:10])
                  if passage_id in relevant)
        ideal = sum(1.0 / math.log2(rank + 2) for rank in range(min(10, len(relevant))))
        totals["ndcg_at_10"] += dcg / ideal if ideal else 0.0
        reciprocal = next((1.0 / (rank + 1) for rank, passage_id in enumerate(ranked)
                           if passage_id in relevant), 0.0)
        totals["mrr"] += reciprocal
    return {name: round(value / len(positives), 8) for name, value in totals.items()}
