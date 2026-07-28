from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL_FILES = {
    "cpu": (
        "onnx/model_cpu_qint8_avx2.onnx",
        "onnx/model_cpu_fp32.onnx",
        "onnx/model.onnx",
    ),
    "gpu": (
        "onnx/model_gpu_fp16.onnx",
        "onnx/model.onnx",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode a passage corpus and build a dense retrieval index."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Exported SentenceTransformer directory containing the ONNX model.",
    )
    parser.add_argument(
        "--model-file",
        help=(
            "ONNX filename relative to --model-dir. If omitted, the builder "
            "selects the preferred file for --target."
        ),
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        nargs="+",
        required=True,
        help="One or more corpus.jsonl files to combine into the index.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for passage_embeddings.npy, passages.jsonl and manifest.json.",
    )
    parser.add_argument(
        "--target",
        choices=("cpu", "gpu"),
        default="cpu",
        help="ONNX Runtime execution provider used while encoding. Default: cpu.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Passages encoded per batch. Default: 64.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing index files in --output-dir.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_model_file(model_dir: Path, target: str, requested: str | None) -> str:
    if requested:
        candidate = model_dir / requested
        if not candidate.is_file():
            raise FileNotFoundError(f"ONNX model does not exist: {candidate}")
        return requested.replace("\\", "/")

    for relative_path in DEFAULT_MODEL_FILES[target]:
        if (model_dir / relative_path).is_file():
            return relative_path

    available = sorted(
        str(path.relative_to(model_dir)).replace("\\", "/")
        for path in model_dir.rglob("*.onnx")
    )
    raise FileNotFoundError(
        f"No suitable {target} ONNX model was found under {model_dir}. "
        f"Available ONNX files: {available}. Pass --model-file explicitly."
    )


def require_provider(target: str) -> str:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "ONNX Runtime is unavailable. Install sentence-transformers[onnx] "
            "for CPU or sentence-transformers[onnx-gpu] for GPU."
        ) from exc

    provider = (
        "CUDAExecutionProvider"
        if target == "gpu"
        else "CPUExecutionProvider"
    )
    available = ort.get_available_providers()
    if provider not in available:
        raise RuntimeError(
            f"{provider} is unavailable. Available providers: {available}"
        )
    return provider


def load_passages(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    passage_positions: dict[str, int] = {}

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Corpus does not exist: {path}")

        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {path}:{line_number}: {exc}"
                    ) from exc

                passage_id = row.get("passage_id")
                text = row.get("text")
                if not isinstance(passage_id, str) or not passage_id.strip():
                    raise ValueError(
                        f"Missing passage_id in {path}:{line_number}"
                    )
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(
                        f"Missing passage text in {path}:{line_number}"
                    )

                if passage_id in passage_positions:
                    existing = rows[passage_positions[passage_id]]
                    if existing.get("text") != text:
                        raise ValueError(
                            f"Passage ID {passage_id!r} has different text "
                            f"across corpus files."
                        )
                    continue

                passage_positions[passage_id] = len(rows)
                rows.append(row)

    if not rows:
        raise ValueError("The supplied corpus files contain no passages.")
    return rows


def prepare_outputs(output_dir: Path, overwrite: bool) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "embeddings": output_dir / "passage_embeddings.npy",
        "passages": output_dir / "passages.jsonl",
        "manifest": output_dir / "manifest.json",
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Index output already exists: "
            + ", ".join(str(path) for path in existing)
            + ". Pass --overwrite to replace it."
        )
    return outputs


def write_passages(path: Path, passages: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in passages:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

    model_dir = args.model_dir.resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    model_file = select_model_file(model_dir, args.target, args.model_file)
    provider = require_provider(args.target)
    passages = load_passages(args.corpus)
    outputs = prepare_outputs(args.output_dir.resolve(), args.overwrite)

    model = SentenceTransformer(
        str(model_dir),
        backend="onnx",
        local_files_only=True,
        model_kwargs={
            "provider": provider,
            "file_name": model_file,
        },
    )
    texts = [row["text"] for row in passages]
    encode_documents = getattr(model, "encode_document", model.encode)
    embeddings = encode_documents(
        texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

    if embeddings.ndim != 2 or embeddings.shape[0] != len(passages):
        raise ValueError(
            f"Unexpected embedding shape {embeddings.shape}; "
            f"expected {len(passages)} rows."
        )
    if not np.isfinite(embeddings).all():
        raise ValueError("The generated passage embeddings contain NaN or infinity.")

    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError(
            "Passage embeddings are not normalized; dot product would not "
            "equal cosine similarity."
        )

    np.save(outputs["embeddings"], embeddings, allow_pickle=False)
    write_passages(outputs["passages"], passages)

    onnx_path = model_dir / model_file
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "directory": str(model_dir),
            "onnx_file": model_file,
            "onnx_sha256": sha256_file(onnx_path),
            "backend": "onnx",
            "provider": provider,
        },
        "index": {
            "passage_count": len(passages),
            "embedding_dimension": int(embeddings.shape[1]),
            "dtype": str(embeddings.dtype),
            "normalized": True,
            "similarity": "dot_product",
            "embeddings_file": outputs["embeddings"].name,
            "embeddings_sha256": sha256_file(outputs["embeddings"]),
            "passages_file": outputs["passages"].name,
            "passages_sha256": sha256_file(outputs["passages"]),
        },
        "sources": [
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for path in args.corpus
        ],
    }
    outputs["manifest"].write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Passages: {len(passages)}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Model file: {model_file}")
    print(f"Provider: {provider}")
    print(f"Index written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
