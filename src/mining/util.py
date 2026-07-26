import json
from pathlib import Path

def _load_corpus_rows(path: Path):
    rows = {}
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[row["passage_id"]] = row
    return rows