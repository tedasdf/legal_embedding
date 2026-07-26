import json, hashlib
from pathlib import Path

def _sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_corpus_rows(path: Path):
    rows = {}
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[row["passage_id"]] = row
    return rows

def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

