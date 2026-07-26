import hashlib
import json
from pathlib import Path

from data.load import load_retrieval_split

def _load_corpus_rows(path: Path):
    rows = {}
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[row["passage_id"]] = row
    return rows

def _sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()



def _mining_report(method, split_dir, output_path, counts, quota, parameters):
    return {"schema_version": 1, "method": method, "input_split": str(split_dir),
            "input_checksums": {name: _sha256(split_dir / name)
                                for name in ("queries.jsonl", "corpus.jsonl", "qrels.tsv")},
            "output": str(output_path), "parameters": parameters,
            "statistics": {"queries": len(counts),
                           "queries_with_full_quota": sum(n == quota for n in counts),
                           "queries_with_no_negatives": sum(n == 0 for n in counts),
                           "minimum_negatives": min(counts, default=0),
                           "maximum_negatives": max(counts, default=0),
                           "mean_negatives": round(sum(counts) / len(counts), 4) if counts else 0}}

def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")



def mine_dense_hard_negatives(config: dict) -> dict:
    """Mine nearest non-positive passages using a trained dense retriever."""
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Dense mining requires sentence-transformers and numpy") from exc
    split_dir = Path(config["input_split"])
    output_path, report_path = Path(config["output_path"]), Path(config["report_path"])
    queries, corpus, qrels = load_retrieval_split(split_dir)
    corpus_rows = _load_corpus_rows(split_dir / "corpus.jsonl")
    model = SentenceTransformer(config["model_name"], device=config.get("device"),
                                trust_remote_code=bool(config.get("trust_remote_code", True)))
    query_ids, passage_ids = sorted(queries), sorted(corpus)
    batch_size = int(config.get("batch_size", 32))
    encode_query = getattr(model, "encode_query", model.encode)
    encode_document = getattr(model, "encode_document", model.encode)
    qvec = encode_query([queries[qid] for qid in query_ids], batch_size=batch_size,
                        normalize_embeddings=True, convert_to_numpy=True)
    pvec = encode_document([corpus[pid] for pid in passage_ids], batch_size=batch_size,
                           normalize_embeddings=True, convert_to_numpy=True)
    scores = np.matmul(qvec, pvec.T)
    order = np.argsort(-scores, axis=1)
    num_negatives = int(config["num_negatives"])
    candidate_depth = min(int(config["candidate_depth"]), len(passage_ids))
    exclude_same_document = bool(config.get("exclude_same_document", True))
    rows, counts = [], []
    for row_index, query_id in enumerate(query_ids):
        positive_ids = set(qrels[query_id])
        positive_documents = {corpus_rows[pid]["document_id"] for pid in positive_ids}
        negatives = []
        for rank, column in enumerate(order[row_index, :candidate_depth], 1):
            passage_id = passage_ids[int(column)]
            if passage_id in positive_ids:
                continue
            if exclude_same_document and corpus_rows[passage_id]["document_id"] in positive_documents:
                continue
            negatives.append({"passage_id": passage_id,
                              "dense_score": round(float(scores[row_index, column]), 8), "rank": rank})
            if len(negatives) == num_negatives:
                break
        counts.append(len(negatives))
        rows.append({"query_id": query_id, "positive_passage_ids": sorted(positive_ids),
                     "negative_passages": negatives})
    _write_jsonl(output_path, rows)
    report = _mining_report("dense_hard_negatives", split_dir, output_path, counts, num_negatives,
                            {"model_name": config["model_name"], "candidate_depth": candidate_depth,
                             "exclude_all_known_positives": True,
                             "exclude_same_document": exclude_same_document,
                             "normalize_embeddings": True, "similarity": "dot_product"})
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


