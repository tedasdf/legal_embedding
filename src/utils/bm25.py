"""Small deterministic BM25 implementation for the lexical baseline."""

from __future__ import annotations

import collections
import math
import re


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
