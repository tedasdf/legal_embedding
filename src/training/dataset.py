"""Training-pair construction for the Embed capability."""

from __future__ import annotations

import random
import json
from dataclasses import dataclass
from pathlib import Path

from data.load import load_retrieval_split


@dataclass(frozen=True)
class PositivePair:
    query_id: str
    passage_id: str
    query: str
    passage: str
    negative_passages: tuple[str, ...] = ()


def load_positive_groups(split_dir: str | Path):
    """Return query text, corpus text, and sorted positives per query."""
    queries, corpus, qrels = load_retrieval_split(split_dir)
    positives = {query_id: sorted(passage_ids) for query_id, passage_ids in qrels.items()}
    return queries, corpus, positives


def sample_one_positive_per_query(queries, corpus, positives, seed: int, epoch: int):
    """Avoid treating another known positive for the same query as an in-batch negative.

    Multi-positive queries rotate deterministically across epochs. Query order is
    shuffled reproducibly; passages are not duplicated for a single query batch item.
    """
    rng = random.Random(f"{seed}:{epoch}")
    query_ids = sorted(positives)
    rng.shuffle(query_ids)
    pairs = []
    for query_id in query_ids:
        passage_ids = positives[query_id]
        passage_id = passage_ids[(seed + epoch) % len(passage_ids)]
        pairs.append(PositivePair(query_id, passage_id, queries[query_id], corpus[passage_id]))
    return pairs


def make_no_duplicate_batches(pairs, batch_size: int):
    """Build batches with no repeated query, positive, or negative text."""
    remaining = list(pairs)
    batches = []
    while remaining:
        batch, deferred = [], []
        query_texts, passage_texts = set(), set()
        for pair in remaining:
            documents = {pair.passage, *pair.negative_passages}
            if (len(batch) < batch_size and pair.query not in query_texts
                    and passage_texts.isdisjoint(documents)):
                batch.append(pair)
                query_texts.add(pair.query)
                passage_texts.update(documents)
            else:
                deferred.append(pair)
        if not batch:
            raise ValueError("Unable to construct a no-duplicates training batch")
        batches.append(batch)
        remaining = deferred
    return batches


def attach_mined_negatives(pairs, corpus, positives, path: str | Path, per_query: int):
    """Attach validated mined negatives to positive-pair training examples."""
    if per_query < 1:
        raise ValueError("hard_negatives_per_query must be positive")
    mined = {}
    with Path(path).open(encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            query_id = row["query_id"]
            negative_ids = [item["passage_id"] for item in row["negative_passages"]]
            if set(negative_ids) & set(positives.get(query_id, ())):
                raise ValueError(f"Mined negatives overlap known positives for {query_id}")
            unknown = set(negative_ids) - set(corpus)
            if unknown:
                raise ValueError(f"Mined negatives reference unknown passages for {query_id}: {sorted(unknown)}")
            mined[query_id] = negative_ids[:per_query]
    missing = [pair.query_id for pair in pairs if not mined.get(pair.query_id)]
    if missing:
        raise ValueError(f"No BM25 hard negatives available for {len(missing)} training queries")
    return [PositivePair(pair.query_id, pair.passage_id, pair.query, pair.passage,
                         tuple(corpus[pid] for pid in mined[pair.query_id])) for pair in pairs]
