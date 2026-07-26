"""Small deterministic BM25 implementation for the lexical baseline."""

from __future__ import annotations

import collections
import math
import re
from pathlib import Path


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.casefold())


class BM25:
    def __init__(self, documents: dict[str, str], k1: float = 1.5, b: float = 0.75):
        self.ids = list(documents)
        self.k1, self.b = k1, b
        self.term_frequencies = [collections.Counter(tokenize(documents[key])) for key in self.ids]
        self.lengths = [sum(counter.values()) for counter in self.term_frequencies]
        self.average_length = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        document_frequency = collections.Counter()
        for counter in self.term_frequencies:
            document_frequency.update(counter.keys())
        count = len(self.ids)
        self.idf = {term: math.log(1 + (count - frequency + 0.5) / (frequency + 0.5))
                    for term, frequency in document_frequency.items()}

    def rank(self, query: str, limit: int | None = None) -> list[str]:
        return [passage_id for passage_id, _ in self.rank_with_scores(query, limit)]

    def rank_with_scores(self, query: str, limit: int | None = None) -> list[tuple[str, float]]:
        query_terms = tokenize(query)
        scores = []
        for passage_id, frequencies, length in zip(self.ids, self.term_frequencies, self.lengths):
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (1 - self.b + self.b * length / self.average_length)
                score += self.idf.get(term, 0.0) * frequency * (self.k1 + 1) / denominator
            scores.append((score, passage_id))
        scores.sort(key=lambda value: (-value[0], value[1]))
        return [(passage_id, score) for score, passage_id in scores[:limit]]


def mine_bm25_hard_negatives(config: dict) -> dict:
    split_dir = Path(config["input_split"])
    output_path = Path(config["output_path"])
    report_path = Path(config["report_path"])
    num_negatives = int(config["num_negatives"])
    candidate_depth = int(config["candidate_depth"])
    if num_negatives < 1 or candidate_depth < num_negatives:
        raise ValueError("candidate_depth must be at least num_negatives, and both must be positive")

    queries, corpus, qrels = load_retrieval_split(split_dir)
    corpus_rows = _load_corpus_rows(split_dir / "corpus.jsonl")
    model = BM25(corpus, k1=float(config.get("k1", 1.5)), b=float(config.get("b", 0.75)))
    exclude_same_document = bool(config.get("exclude_same_document", True))
    minimum_score = float(config.get("minimum_score", 0.0))
    rows, counts = [], []

    for query_id in sorted(queries):
        positive_ids = set(qrels[query_id])
        positive_document_ids = {corpus_rows[pid]["document_id"] for pid in positive_ids}
        ranked = model.rank_with_scores(queries[query_id], candidate_depth)
        negatives = []
        for rank, (passage_id, score) in enumerate(ranked, 1):
            if passage_id in positive_ids or score <= minimum_score:
                continue
            if exclude_same_document and corpus_rows[passage_id]["document_id"] in positive_document_ids:
                continue
            negatives.append({"passage_id": passage_id, "bm25_score": round(score, 8), "rank": rank})
            if len(negatives) == num_negatives:
                break
        counts.append(len(negatives))
        rows.append({"query_id": query_id, "positive_passage_ids": sorted(positive_ids),
                     "negative_passages": negatives})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    report = {
        "schema_version": 1,
        "method": "bm25_hard_negatives",
        "input_split": str(split_dir),
        "input_checksums": {name: _sha256(split_dir / name)
                            for name in ("queries.jsonl", "corpus.jsonl", "qrels.tsv")},
        "output": str(output_path),
        "parameters": {"num_negatives": num_negatives, "candidate_depth": candidate_depth,
                       "exclude_all_known_positives": True,
                       "exclude_same_document": exclude_same_document,
                       "minimum_score_exclusive": minimum_score,
                       "k1": model.k1, "b": model.b},
        "statistics": {"queries": len(rows), "queries_with_full_quota": sum(n == num_negatives for n in counts),
                       "queries_with_no_negatives": sum(n == 0 for n in counts),
                       "minimum_negatives": min(counts, default=0),
                       "maximum_negatives": max(counts, default=0),
                       "mean_negatives": round(sum(counts) / len(counts), 4) if counts else 0},
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
