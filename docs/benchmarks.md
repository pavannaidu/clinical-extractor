# Benchmarks

This repo includes both notebook-based endpoint benchmarking and app-based throughput benchmarking.

## Benchmark Surfaces

### MedGemma serving benchmark

`src/medgemma/03_benchmark_endpoint.py` benchmarks concurrent endpoint invocation against a Delta input table and logs throughput and latency metrics to MLflow.

It is designed to answer questions like:

- How does throughput change with concurrency?
- How many output tokens per second can the endpoint sustain?
- What are the p50, p90, and p99 latencies?
- How does workload type affect the serving curve?

### App load-testing view

`src/app` includes a short, controlled FMAPI burst test using a fixed clinical note and JSON schema. It reports schema-valid responses, retries, endpoint invocations, token-usage coverage, and observed input/output delivery rates against documentation-derived reference lines.

This view is useful for a quick, reproducible workload comparison. It is not a sustained quota, capacity, or cost certification; workspace traffic, retry consumption, and the short observation window all affect the result.

## How to Use Benchmarking in the Showcase

- Use notebook benchmarking when the customer cares about endpoint performance and operational tuning.
- Use the app benchmark view for a quick, controlled demonstration of structured-output reliability and observed delivery throughput.

## Recommended Benchmark Narrative

1. Start with correctness and structured output.
2. Then show throughput and latency.
3. Frame performance as workload-dependent:
   clinical note complexity, schema size, and serving configuration all matter.
