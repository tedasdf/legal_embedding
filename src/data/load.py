"""Read the repository retrieval interchange format."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def load_retrieval_split(folder: str | Path):
    folder = Path(folder)
    queries = {row["query_id"]: row["text"] for row in _jsonl(folder / "queries.jsonl")}
    corpus = {row["passage_id"]: row["text"] for row in _jsonl(folder / "corpus.jsonl")}
    qrels = defaultdict(set)
    with (folder / "qrels.tsv").open(encoding="utf-8-sig") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        if header != ["query_id", "passage_id", "relevance"]:
            raise ValueError(f"Unexpected qrels header: {header}")
        for line in handle:
            query_id, passage_id, relevance = line.rstrip("\n").split("\t")
            if float(relevance) > 0:
                qrels[query_id].add(passage_id)
    if set(qrels) - set(queries):
        raise ValueError("qrels contains unknown queries")
    if set().union(*qrels.values()) - set(corpus):
        raise ValueError("qrels contains unknown passages")
    return queries, corpus, dict(qrels)


def _jsonl(path: Path):
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)
