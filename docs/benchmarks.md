# Benchmarks

This repo includes both notebook-based endpoint benchmarking and app-based throughput benchmarking.

## Recorded workflow smoke results — November 18, 2026

The latest attempts of all ten model workflows completed successfully in a
dedicated test namespace. The batch fixture was **MedGemma Synthetic Batch Input**
with one replication: 20 seed records, seven extraction categories, and a schema
per row. Each path processed the same inputs.

| Batch path | Output table suffix | Output rows | Independently schema-valid | IDs and schemas match input |
|---|---|---:|---:|---|
| Classic GPU driver | `medgemma_batch_output` | 20 | 20 | Yes |
| Spark SQL `ai_query` | `medgemma_batch_output_udf` | 20 | 20 | Yes |
| Serverless GPU / Ray | `medgemma_batch_output_sgc` | 20 | 20 | Yes |

These are **60 output rows from 20 distinct inputs**, not 60 independent clinical
cases. Classic driver inference took **85.31 seconds**; materialized SQL inference
took **42.34 seconds**. Those timings exclude cluster provisioning and initial
model loading. The classic job used 17.3 LTS ML GPU; its bundled driver libraries
differed from the separately packaged serving environment. Treat the timings as
measurements of these smoke paths, not a general speedup claim.

The latest endpoint benchmark passed **16/16 measured requests**: eight input rows,
one row per request, concurrency levels `1,2`, and one warm-up request before each
level. Predictions from all five temporary custom endpoints passed their request schemas. Temporary
endpoints were deleted and the test job clusters terminated after validation.
Models, weights, and Delta tables were retained in the isolated test namespace.
Detailed run exports are operator-local under the ignored `output/` directory;
this checked-in summary contains no workspace URLs or resource IDs.

### What these results establish

The jobs can execute, package/register models, serve predictions, and produce
schema-valid batch records on the tested configuration. The training smoke used
tiny synthetic datasets and short runs. It does not establish accuracy on the
full NCBI Disease Corpus, improvement over a base model, or time saved by reviewers.

Two observations explain why field review and realistic delivery tests remain
necessary:

- One schema-valid vitals response rendered blood pressure as `"{118/74}"`.
  A reference expecting `"118/74"` should reject that value despite its string type.
- A cold endpoint request exceeded a 300-second client timeout before a later
  warm request passed. Startup time is outside the inference-only timings above.

An earlier 16-request benchmark also had one schema-validation failure. Its exact
response was not retained and the cause remains unconfirmed; the latest repeat
passed all 16 requests. These results are not a zero-failure reliability guarantee.

### Reproduce the checks in your own workspace

1. Follow [model-job setup](setup.md#model-jobs) using a dedicated demo namespace,
   volumes, and endpoint names. Keep runtime versions and serving dependency pins
   with the result you record.
2. Set the preparation notebook's `synthetic_replications` widget/job-task
   parameter to `1` to materialize 20 inputs. The bundled job default is `50`
   (1,000 rows), so record this override explicitly.
3. Run the classic and Serverless GPU batch notebooks against that input. Compare
   each output table's row count, IDs, per-row schema, parse errors, and schema
   validation results. A successful job state alone does not establish these checks.
4. Run the endpoint benchmark with the chosen input-row count, concurrency levels,
   warm-up count, and output-token budget recorded. Separate cold-start measurements
   from warm inference and include materialized SQL execution in elapsed time.
5. For extraction quality, use [reviewed ground-truth JSON](setup.md#run-a-scored-evaluation)
   and report field agreement and fully-correct records separately from schema
   validity. See [data sources](data.md) for fixture and corpus provenance.

## Benchmark Surfaces

### MedGemma serving benchmark

`src/models/medgemma/xgrammar/03_benchmark_endpoint.py` benchmarks concurrent endpoint invocation against a Delta input table and logs throughput and latency metrics to MLflow.

It is designed to answer questions like:

- How does throughput change with concurrency?
- How many output tokens per second can the endpoint sustain?
- What are the p50, p90, and p99 latencies?
- How does workload type affect the serving curve?

### App load-testing view

`src/app` runs short load tests against selected FMAPI or custom endpoints using the
chosen clinical note and JSON schema; a bundled sample is available by default.
It reports schema-valid responses, retries, endpoint invocations, token-usage
coverage, and observed input/output delivery rates. Documented rate limits, where
available, supply reference lines for comparison.

This view is useful for a quick, reproducible workload comparison. It is not a sustained quota, capacity, or cost certification; workspace traffic, retry consumption, and the short observation window all affect the result.

## How to Use Benchmarking in the Showcase

- Use notebook benchmarking when the customer cares about endpoint performance and operational tuning.
- Use the app benchmark view for a quick, controlled demonstration of structured-output reliability and observed delivery throughput.

## Recommended Benchmark Narrative

1. Start with correctness and structured output.
2. Then show throughput and latency.
3. Frame performance as workload-dependent:
   clinical note complexity, schema size, and serving configuration all matter.
