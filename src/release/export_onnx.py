import argparse
import shutil
from pathlib import Path

from sentence_transformers import (
    SentenceTransformer,
    export_dynamic_quantized_onnx_model,
    export_optimized_onnx_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a local SentenceTransformer checkpoint or Hugging Face "
            "model to optimized ONNX."
        )
    )
    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Local checkpoint directory or Hugging Face model ID, for example "
            "'artifacts/checkpoints/embed/gte-modernbert-base/final' or "
            "'Sing0402/legal-embed-gte-inbatch'."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory in which to save the deployable SentenceTransformer model.",
    )
    parser.add_argument(
        "--target",
        choices=("cpu", "gpu"),
        default="cpu",
        help="CPU produces O3 FP32; GPU produces O4 FP16. Default: cpu.",
    )
    parser.add_argument(
        "--quantize-int8",
        action="store_true",
        help="Also create an AVX2 INT8 model. Supported only for --target cpu.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace a non-empty output directory.",
    )
    parser.add_argument(
        "--revision",
        help="Optional Hugging Face revision (branch, tag, or commit).",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not access Hugging Face; useful for local checkpoints or cached models.",
    )
    return parser.parse_args()


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. "
                "Choose another directory or pass --overwrite."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def require_provider(provider: str) -> None:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "ONNX Runtime is unavailable. Install sentence-transformers[onnx] "
            "for CPU or sentence-transformers[onnx-gpu] for GPU."
        ) from exc

    available = ort.get_available_providers()
    if provider not in available:
        raise RuntimeError(
            f"{provider} is unavailable. Available ONNX Runtime providers: "
            f"{available}. Install the matching ONNX Runtime package."
        )


def main() -> None:
    args = parse_args()
    if args.target == "gpu" and args.quantize_int8:
        raise ValueError("--quantize-int8 is CPU-only; omit it for --target gpu.")

    provider = (
        "CUDAExecutionProvider"
        if args.target == "gpu"
        else "CPUExecutionProvider"
    )
    require_provider(provider)
    prepare_output_dir(args.output_dir, args.overwrite)

    model = SentenceTransformer(
        args.model,
        backend="onnx",
        trust_remote_code=True,
        revision=args.revision,
        local_files_only=args.local_files_only,
        model_kwargs={"provider": provider},
    )

    # Save tokenizer, pooling, normalization, metadata, and the initial ONNX model.
    model.save_pretrained(str(args.output_dir))

    optimization_level = "O4" if args.target == "gpu" else "O3"
    file_suffix = "gpu_fp16" if args.target == "gpu" else "cpu_fp32"
    export_optimized_onnx_model(
        model=model,
        optimization_config=optimization_level,
        model_name_or_path=str(args.output_dir),
        file_suffix=file_suffix,
    )

    if args.quantize_int8:
        export_dynamic_quantized_onnx_model(
            model=model,
            quantization_config="avx2",
            model_name_or_path=str(args.output_dir),
            file_suffix="cpu_qint8_avx2",
        )

    source_kind = "local directory" if Path(args.model).is_dir() else "Hugging Face model ID"
    print(f"Source ({source_kind}): {args.model}")
    print(f"Target: {args.target} ({provider}, {optimization_level})")
    print(f"ONNX model exported to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
