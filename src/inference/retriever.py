from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer


class LegalRetriever:
    def __init__(
        self,
        model_dir: Path,
        embeddings_path: Path,
        passages_path: Path,
    ) -> None:
        self.model = SentenceTransformer(
            str(model_dir),
            backend="onnx",
            local_files_only=True,
            model_kwargs={
                "provider": "CUDAExecutionProvider",
                "file_name": "onnx/model_gpu_fp16.onnx",
            },
        )

        self.passage_embeddings = np.load(embeddings_path)
        self.passages = self._load_jsonl(passages_path)

        if len(self.passages) != self.passage_embeddings.shape[0]:
            raise ValueError(
                "Passage metadata count does not match embedding count"
            )

    @staticmethod
    def _load_jsonl(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []

        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    rows.append(json.loads(line))

        return rows

    def search(self, query: str, top_k: int = 5) -> dict[str, Any]:
        if not query.strip():
            raise ValueError("Query must not be empty")

        top_k = min(top_k, len(self.passages))

        started = time.perf_counter()

        query_vector = self.model.encode_query(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        encode_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()

        scores = query_vector[0] @ self.passage_embeddings.T

        candidate_indices = np.argpartition(
            scores,
            -top_k,
        )[-top_k:]

        ranked_indices = candidate_indices[
            np.argsort(scores[candidate_indices])[::-1]
        ]

        search_ms = (time.perf_counter() - started) * 1000

        results = []

        for rank, index in enumerate(ranked_indices, start=1):
            passage = self.passages[int(index)]

            results.append(
                {
                    "rank": rank,
                    "score": float(scores[index]),
                    "passage_id": passage.get("passage_id"),
                    "citation": passage.get("citation"),
                    "text": passage.get("text"),
                }
            )

        return {
            "query": query,
            "results": results,
            "latency_ms": {
                "query_encoding": round(encode_ms, 3),
                "search": round(search_ms, 3),
                "total": round(encode_ms + search_ms, 3),
            },
        }