# Databricks notebook source
# DBTITLE 1,Overview
# MAGIC %md
# MAGIC # MedGemma XGrammar — Serving Endpoint Throughput Benchmark
# MAGIC
# MAGIC Reads medical text records from a Delta table and sends them to the Model
# MAGIC Serving endpoint concurrently, sweeping across concurrency levels to find
# MAGIC peak throughput.  Logs per-level latency percentiles and rows/s to MLflow.
# MAGIC
# MAGIC **Input**: A running endpoint (`02_deploy_endpoint.py`) and the Delta input
# MAGIC table created by `01_prepare_batch_input.py` (`medgemma_batch_input`).

# COMMAND ----------

# DBTITLE 1,Widgets
dbutils.widgets.removeAll()
dbutils.widgets.text("catalog",              "pavan_naidu_catalog",          "Catalog")
dbutils.widgets.text("schema",               "mlops",                        "Schema")
dbutils.widgets.text("endpoint_name",        "medgemma-xgrammar-endpoint",   "Endpoint Name")
dbutils.widgets.dropdown("workload_type", "GPU_MEDIUM", ["GPU_SMALL", "GPU_MEDIUM", "GPU_LARGE", "CPU"], "Workload Type")
dbutils.widgets.text("input_table",          "medgemma_batch_input",         "Input Delta Table")
dbutils.widgets.text("num_rows",             "200",                          "Rows to benchmark (0 = all)")
dbutils.widgets.text("request_batch_size",   "1",                            "Rows per HTTP request")
dbutils.widgets.text("concurrency_levels",   "1,4",                          "Concurrency levels to sweep (comma-separated)")
dbutils.widgets.text("warmup_requests",      "2",                            "Warm-up requests before each level")

# COMMAND ----------

# DBTITLE 1,Configuration
import json, time, statistics, requests as _requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from collections import defaultdict
import mlflow

CATALOG             = dbutils.widgets.get("catalog")
SCHEMA_NAME         = dbutils.widgets.get("schema")
WORKLOAD_TYPE       = dbutils.widgets.get("workload_type")
_GPU_SUFFIX = {"GPU_SMALL": "t4", "GPU_MEDIUM": "a10", "GPU_LARGE": "a100", "CPU": "cpu"}
ENDPOINT_NAME       = f"{dbutils.widgets.get('endpoint_name')}-{_GPU_SUFFIX.get(WORKLOAD_TYPE, WORKLOAD_TYPE.lower())}"
INPUT_TABLE         = dbutils.widgets.get("input_table")
NUM_ROWS            = int(dbutils.widgets.get("num_rows"))
REQUEST_BATCH_SIZE  = int(dbutils.widgets.get("request_batch_size"))
CONCURRENCY_LEVELS  = [int(x.strip()) for x in dbutils.widgets.get("concurrency_levels").split(",")]
WARMUP_REQUESTS     = int(dbutils.widgets.get("warmup_requests"))

INPUT_TABLE_FQN  = f"{CATALOG}.{SCHEMA_NAME}.{INPUT_TABLE}"
MLFLOW_EXPERIMENT_ID = "2468185612186458"

_token   = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
_host    = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
_headers = {"Authorization": f"Bearer {_token}", "Content-Type": "application/json"}
INVOKE_URL     = f"{_host}/serving-endpoints/{ENDPOINT_NAME}/invocations"
INVOKE_HEADERS = _headers

# Fetch endpoint metadata for MLflow traceability
_ep_resp = _requests.get(
    f"{_host}/api/2.0/serving-endpoints/{ENDPOINT_NAME}",
    headers=_headers,
    timeout=30,
)
_served = _ep_resp.json().get("config", {}).get("served_entities", [{}])[0]
ENDPOINT_MODEL_NAME    = _served.get("entity_name", "unknown")
ENDPOINT_MODEL_VERSION = str(_served.get("entity_version", "unknown"))
ENDPOINT_WORKLOAD_TYPE = _served.get("workload_type", "unknown")
ENDPOINT_WORKLOAD_SIZE = _served.get("workload_size", "unknown")
print(f"Endpoint serving: {ENDPOINT_MODEL_NAME} v{ENDPOINT_MODEL_VERSION} "
      f"({ENDPOINT_WORKLOAD_TYPE}/{ENDPOINT_WORKLOAD_SIZE})")

mlflow.set_registry_uri("databricks-uc")

print(f"Endpoint      : {ENDPOINT_NAME}")
print(f"Input table   : {INPUT_TABLE_FQN}")
print(f"Num rows      : {NUM_ROWS if NUM_ROWS > 0 else 'all'}")
print(f"Batch size    : {REQUEST_BATCH_SIZE} rows/request")
print(f"Concurrency   : {CONCURRENCY_LEVELS}")
print(f"Warm-up reqs  : {WARMUP_REQUESTS}")

# COMMAND ----------

# DBTITLE 1,Load Input Data
import pandas as pd

raw_df = spark.table(INPUT_TABLE_FQN).toPandas()
if NUM_ROWS > 0:
    raw_df = raw_df.head(NUM_ROWS)

# De-duplicate schema strings (they may be JSON already or need re-serialising)
def _normalise_schema(s):
    try:
        return json.dumps(json.loads(s))
    except Exception:
        return s

raw_df["schema"] = raw_df["schema"].apply(_normalise_schema)

print(f"Loaded {len(raw_df)} rows from {INPUT_TABLE_FQN}")
print(f"Categories    : {dict(raw_df['category'].value_counts())}")

# Each request is a single chat conversation (ChatModel = one request per row)
batches = []
for _, row in raw_df.iterrows():
    batches.append({
        "messages": [{"role": "user", "content": row["prompt"]}],
        "max_tokens": 512,
        "custom_inputs": {"schema": row["schema"]},
    })

print(f"Request batches: {len(batches)}  (1 row each, chat format)")

# COMMAND ----------

# DBTITLE 1,Request Helper
def send_request(chat_payload):
    """POST one chat request to the endpoint.
    Returns (latency_s, num_rows, input_tokens, output_tokens, error_or_None)."""
    t0 = time.perf_counter()
    try:
        resp = _requests.post(INVOKE_URL, headers=INVOKE_HEADERS, json=chat_payload, timeout=120)
        latency = time.perf_counter() - t0
        if resp.status_code != 200:
            return latency, 1, 0, 0, f"HTTP {resp.status_code}: {resp.text[:200]}"
        body = resp.json()
        content = body["choices"][0]["message"]["content"]
        usage = body.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        error = None
        try:
            json.loads(content)
        except Exception as e:
            error = str(e)
        return latency, 1, input_tokens, output_tokens, error
    except Exception as e:
        latency = time.perf_counter() - t0
        return latency, 1, 0, 0, str(e)

# COMMAND ----------

# DBTITLE 1,Benchmark Runner
def run_level(concurrency, batches, label=""):
    """Warm up then run all batches at the given concurrency.  Returns metrics dict."""

    # ── Warm-up ───────────────────────────────────────────────────────────────
    if WARMUP_REQUESTS > 0:
        print(f"  warm-up ({WARMUP_REQUESTS} req)…", end=" ")
        with ThreadPoolExecutor(max_workers=min(concurrency, WARMUP_REQUESTS)) as pool:
            futs = [pool.submit(send_request, batches[i % len(batches)])
                    for i in range(WARMUP_REQUESTS)]
            for f in as_completed(futs):
                f.result()
        print("done")

    # ── Timed run ─────────────────────────────────────────────────────────────
    latencies     = []
    errors        = []
    total_rows    = 0
    total_in_tok  = 0
    total_out_tok = 0

    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = {pool.submit(send_request, b): b for b in batches}
        for fut in as_completed(futs):
            lat, nrows, in_tok, out_tok, err = fut.result()
            latencies.append(lat)
            total_rows    += nrows
            if err:
                errors.append(err)
            else:
                total_in_tok  += in_tok
                total_out_tok += out_tok
    wall_elapsed = time.perf_counter() - wall_start

    rows_per_s        = total_rows / wall_elapsed if wall_elapsed > 0 else 0.0
    reqs_per_s        = len(batches) / wall_elapsed if wall_elapsed > 0 else 0.0
    out_tokens_per_s  = total_out_tok / wall_elapsed if wall_elapsed > 0 else 0.0
    total_tokens_per_s = (total_in_tok + total_out_tok) / wall_elapsed if wall_elapsed > 0 else 0.0
    mean_out_tokens   = total_out_tok / total_rows if total_rows > 0 else 0.0
    error_rate        = len(errors) / len(batches) if batches else 0.0
    sorted_lats       = sorted(latencies)

    def _pct(p):
        idx = max(0, int(len(sorted_lats) * p / 100) - 1)
        return sorted_lats[idx]

    metrics = {
        "concurrency":          concurrency,
        "total_rows":           total_rows,
        "total_requests":       len(batches),
        "wall_elapsed_s":       round(wall_elapsed, 2),
        "rows_per_s":           round(rows_per_s, 2),
        "reqs_per_s":           round(reqs_per_s, 2),
        "total_input_tokens":   total_in_tok,
        "total_output_tokens":  total_out_tok,
        "output_tokens_per_s":  round(out_tokens_per_s, 2),
        "total_tokens_per_s":   round(total_tokens_per_s, 2),
        "mean_output_tokens":   round(mean_out_tokens, 1),
        "error_count":          len(errors),
        "error_rate":           round(error_rate, 4),
        "latency_mean_s":       round(statistics.mean(latencies), 3),
        "latency_median_s":     round(statistics.median(latencies), 3),
        "latency_p90_s":        round(_pct(90), 3),
        "latency_p99_s":        round(_pct(99), 3),
        "latency_min_s":        round(min(latencies), 3),
        "latency_max_s":        round(max(latencies), 3),
    }

    print(
        f"  concurrency={concurrency:>2}  "
        f"wall={wall_elapsed:6.1f}s  "
        f"out_tok/s={out_tokens_per_s:6.1f}  "
        f"rows/s={rows_per_s:5.2f}  "
        f"p50={metrics['latency_median_s']:.2f}s  "
        f"p90={metrics['latency_p90_s']:.2f}s  "
        f"p99={metrics['latency_p99_s']:.2f}s  "
        f"errors={len(errors)}"
    )
    return metrics

# COMMAND ----------

# DBTITLE 1,Run Concurrency Sweep
print(f"Starting concurrency sweep: {CONCURRENCY_LEVELS}")
print(f"{'─'*90}")

all_metrics = []
for c in CONCURRENCY_LEVELS:
    print(f"\nLevel concurrency={c}")
    m = run_level(c, batches)
    all_metrics.append(m)

# COMMAND ----------

# DBTITLE 1,Summary Table
print(f"\n{'═'*105}")
print(f"{'Concurrency':>12} {'OutTok/s':>9} {'TotTok/s':>9} {'AvgOutTok':>10} "
      f"{'Rows/s':>7} {'Wall(s)':>8} {'p50(s)':>7} {'p90(s)':>7} {'p99(s)':>7} {'Errors':>7}")
print(f"{'─'*105}")
for m in all_metrics:
    print(
        f"{m['concurrency']:>12} "
        f"{m['output_tokens_per_s']:>9.1f} "
        f"{m['total_tokens_per_s']:>9.1f} "
        f"{m['mean_output_tokens']:>10.1f} "
        f"{m['rows_per_s']:>7.2f} "
        f"{m['wall_elapsed_s']:>8.1f} "
        f"{m['latency_median_s']:>7.2f} "
        f"{m['latency_p90_s']:>7.2f} "
        f"{m['latency_p99_s']:>7.2f} "
        f"{m['error_count']:>7}"
    )

best = max(all_metrics, key=lambda m: m["output_tokens_per_s"])
print(f"\n✓ Best throughput: concurrency={best['concurrency']}  "
      f"{best['output_tokens_per_s']:.1f} output_tok/s  "
      f"{best['total_tokens_per_s']:.1f} total_tok/s  "
      f"{best['rows_per_s']:.2f} rows/s")

# COMMAND ----------

# DBTITLE 1,Log to MLflow
run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

with mlflow.start_run(
    experiment_id=MLFLOW_EXPERIMENT_ID,
    run_name=f"endpoint_benchmark_{run_ts}"
) as parent_run:
    mlflow.log_params({
        "endpoint_name":       ENDPOINT_NAME,
        "input_table":         INPUT_TABLE_FQN,
        "num_rows":            len(raw_df),
        "request_batch_size":  REQUEST_BATCH_SIZE,
        "concurrency_levels":  str(CONCURRENCY_LEVELS),
        "warmup_requests":     WARMUP_REQUESTS,
        "registered_model_name": ENDPOINT_MODEL_NAME,
        "model_version":         ENDPOINT_MODEL_VERSION,
        "workload_type":         ENDPOINT_WORKLOAD_TYPE,
        "workload_size":         ENDPOINT_WORKLOAD_SIZE,
    })
    mlflow.log_params({"endpoint_model": ENDPOINT_MODEL_NAME})

    # Best-level summary metrics on the parent run
    mlflow.log_metrics({
        "best_concurrency":        float(best["concurrency"]),
        "best_output_tokens_per_s": best["output_tokens_per_s"],
        "best_total_tokens_per_s": best["total_tokens_per_s"],
        "best_rows_per_s":         best["rows_per_s"],
        "best_error_rate":         best["error_rate"],
    })

    # Log per-level metrics with concurrency as suffix (e.g. output_tokens_per_s_c4)
    for m in all_metrics:
        c = m["concurrency"]
        mlflow.log_metrics({
            f"output_tokens_per_s_c{c}":  m["output_tokens_per_s"],
            f"total_tokens_per_s_c{c}":   m["total_tokens_per_s"],
            f"mean_output_tokens_c{c}":   m["mean_output_tokens"],
            f"rows_per_s_c{c}":           m["rows_per_s"],
            f"wall_elapsed_s_c{c}":       m["wall_elapsed_s"],
            f"latency_median_s_c{c}":     m["latency_median_s"],
            f"latency_p90_s_c{c}":        m["latency_p90_s"],
            f"latency_p99_s_c{c}":        m["latency_p99_s"],
            f"error_count_c{c}":          float(m["error_count"]),
            f"error_rate_c{c}":           m["error_rate"],
        })

    print(f"\n✓ MLflow parent run: {parent_run.info.run_id}")
    print(f"  Experiment: https://fevm-pavan-naidu.cloud.databricks.com"
          f"/?o=7474645380671326#/experiments/{MLFLOW_EXPERIMENT_ID}"
          f"/runs/{parent_run.info.run_id}")
