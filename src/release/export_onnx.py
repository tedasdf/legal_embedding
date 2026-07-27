from pathlib import Path

from sentence_transformers import (
    SentenceTransformer,
    export_dynamic_quantized_onnx_model,
    export_optimized_onnx_model,
)


MODEL_ID = "Sing0402/auslegal-embed-gte-inbatch"
OUTPUT_DIR = Path("artifacts/auslegal-embed-gte-inbatch")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


    model = SentenceTransformer("Sing0402/auslegal-embed-gte-inbatch")

    # Saves tokenizer, pooling configuration, normalization configuration,
    # model metadata and the exported ONNX model.
    model.save_pretrained(str(OUTPUT_DIR))

    # Produce an ONNX Runtime-optimized FP32 model.
    export_optimized_onnx_model(
        model=model,
        optimization_config="O3",
        model_name_or_path=str(OUTPUT_DIR),
        file_suffix="optimized",
    )

    # Produce a dynamically quantized INT8 model.
    # AVX2 is the safest initial choice for ordinary x86-64 CPUs.
    export_dynamic_quantized_onnx_model(
        model=model,
        quantization_config="avx2",
        model_name_or_path=str(OUTPUT_DIR),
        file_suffix="qint8_avx2",
    )

    print(f"ONNX model exported to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()