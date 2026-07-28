from pathlib import Path
from time import perf_counter_ns

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_DIR = Path(
    "artifacts/releases/legal-embed-gte-inbatch/cpu"
)


def elapsed_ms(start_ns: int, end_ns: int) -> float:
    return (end_ns - start_ns) / 1_000_000


def benchmark(
    model: SentenceTransformer,
    texts: list[str],
    *,
    batch_size: int,
    warmup_runs: int = 10,
    measured_runs: int = 100,
    mode: str = "encode",
) -> dict[str, float]:
    encode_function = {
        "encode": model.encode,
        "query": model.encode_query,
        "document": model.encode_document,
    }[mode]

    # GPU/ONNX warm-up:
    # initialises kernels, memory allocations and execution plans.
    for _ in range(warmup_runs):
        encode_function(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    latencies_ms: list[float] = []

    for _ in range(measured_runs):
        started = perf_counter_ns()

        embeddings = encode_function(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        finished = perf_counter_ns()
        latencies_ms.append(elapsed_ms(started, finished))

    values = np.asarray(latencies_ms)
    mean_latency_ms = float(values.mean())

    return {
        "runs": measured_runs,
        "texts_per_request": len(texts),
        "embedding_dimension": embeddings.shape[-1],
        "mean_ms": mean_latency_ms,
        "median_p50_ms": float(np.percentile(values, 50)),
        "p90_ms": float(np.percentile(values, 90)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "minimum_ms": float(values.min()),
        "maximum_ms": float(values.max()),
        "average_ms_per_text": mean_latency_ms / len(texts),
        "throughput_texts_per_second": (
            len(texts) / (mean_latency_ms / 1_000)
        ),
    }


def print_results(name: str, results: dict[str, float]) -> None:
    print(f"\n{name}")
    print("-" * len(name))
    print(f"Runs:                    {results['runs']}")
    print(f"Texts per request:       {results['texts_per_request']}")
    print(f"Embedding dimension:     {results['embedding_dimension']}")
    print(f"Mean latency:            {results['mean_ms']:.2f} ms")
    print(f"Median latency (p50):    {results['median_p50_ms']:.2f} ms")
    print(f"p90 latency:             {results['p90_ms']:.2f} ms")
    print(f"p95 latency:             {results['p95_ms']:.2f} ms")
    print(f"p99 latency:             {results['p99_ms']:.2f} ms")
    print(f"Minimum latency:         {results['minimum_ms']:.2f} ms")
    print(f"Maximum latency:         {results['maximum_ms']:.2f} ms")
    print(
        f"Average per text:        "
        f"{results['average_ms_per_text']:.2f} ms"
    )
    print(
        f"Throughput:              "
        f"{results['throughput_texts_per_second']:.2f} texts/s"
    )


# Measure model startup separately.
load_started = perf_counter_ns()

model = SentenceTransformer(
    str(MODEL_DIR),
    backend="onnx",
    local_files_only=True,
    model_kwargs={
        "provider": "CUDAExecutionProvider",
        "file_name": "onnx/model_cpu_fp16.onnx",
    },
)

load_finished = perf_counter_ns()

print(f"Backend: {model.get_backend()}")
print(
    f"Model loading time: "
    f"{elapsed_ms(load_started, load_finished):.2f} ms"
)


# Measure the very first request separately.
query = ["What constitutes a breach of contract?"]

cold_started = perf_counter_ns()

cold_embedding = model.encode_query(
    query,
    batch_size=1,
    convert_to_numpy=True,
    show_progress_bar=False,
)

cold_finished = perf_counter_ns()

print(
    f"Cold query latency: "
    f"{elapsed_ms(cold_started, cold_finished):.2f} ms"
)
print(f"Cold embedding shape: {cold_embedding.shape}")


# Single-query online latency.
single_query_results = benchmark(
    model,
    query,
    batch_size=1,
    warmup_runs=10,
    measured_runs=100,
    mode="query",
)

print_results("Single-query benchmark", single_query_results)


# Batched document throughput.
documents = [
    (
        "A party failed to perform its contractual obligations, "
        "resulting in a claim for breach of contract."
    ),
    (
        "The applicant argued that the administrative decision "
        "was affected by jurisdictional error."
    ),
] * 8  # 16 documents

batch_results = benchmark(
    model,
    documents,
    batch_size=16,
    warmup_runs=10,
    measured_runs=100,
    mode="document",
)

print_results("Batch-of-16 document benchmark", batch_results)