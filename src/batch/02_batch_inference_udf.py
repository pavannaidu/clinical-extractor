# Databricks notebook source
# DBTITLE 1,Overview
# MAGIC %md
# MAGIC # MedGemma XGrammar — Batch Inference
# MAGIC
# MAGIC Loads the registered `MedGemmaPyfunc` model from Unity Catalog, runs
# MAGIC structured JSON extraction over a Delta table of medical text records,
# MAGIC writes results back to a Delta output table, and logs each inference
# MAGIC trace to an MLflow experiment.
# MAGIC
# MAGIC **Input**:  Delta table `{catalog}.{schema}.{input_table}` with columns:
# MAGIC - `id` (string) — unique row identifier
# MAGIC - `prompt` (string)  — medical text / instruction
# MAGIC - `schema` (string)  — JSON-serialised JSON schema for extraction
# MAGIC
# MAGIC **Output**: Delta table `{catalog}.{schema}.{output_table}` with the
# MAGIC original columns plus:
# MAGIC - `extracted_json` (string)  — raw model output
# MAGIC - `parsed` (string)          — re-serialised parsed JSON, or null on failure
# MAGIC - `parse_error` (string)     — error message when parsing fails
# MAGIC - `inference_ts` (string)    — wall-clock time of inference run
# MAGIC - `mlflow_run_id` (string)   — MLflow run that logged this batch
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - `src/medgemma/01_register.py` has been run and the **challenger**
# MAGIC   model alias is set in Unity Catalog.
# MAGIC - `src/medgemma/02_deploy_endpoint.py` has created the serving endpoint
# MAGIC   for the selected workload type.

# COMMAND ----------

# DBTITLE 1,Widgets
dbutils.widgets.removeAll()
dbutils.widgets.text("catalog",               "pavan_naidu_catalog",            "Catalog")
dbutils.widgets.text("schema",                "mlops",                          "Schema")
dbutils.widgets.text("registered_model_name", "medgemma_xgrammar_structured",   "Registered Model Name")
dbutils.widgets.text("model_alias",           "challenger",                     "Model Alias")
dbutils.widgets.text("endpoint_name",         "medgemma-1_5-4b-xgrammar",       "Endpoint Base Name")
dbutils.widgets.dropdown(
    "workload_type",
    "GPU_MEDIUM",
    ["GPU_SMALL", "GPU_MEDIUM", "GPU_LARGE", "CPU"],
    "Endpoint Workload Type",
)
dbutils.widgets.text("input_table",           "medgemma_batch_input",           "Input Delta Table")
dbutils.widgets.text("output_table",          "medgemma_batch_output",          "Output Delta Table")
dbutils.widgets.text("batch_size",            "32",                             "Rows per Spark partition (batch size)")
dbutils.widgets.text("output_table_udf",      "medgemma_batch_output_udf",      "UDF Output Delta Table")
dbutils.widgets.text("pandas_predict_batch_size", "32",                         "Rows per pandas predict chunk")

# COMMAND ----------

# DBTITLE 1,Install Dependencies
%pip install -U \
  "transformers>=4.51.3" \
  "accelerate==1.4.0" \
  "bitsandbytes==0.46.1" \
  xgrammar==0.1.31 \
  huggingface_hub

# COMMAND ----------

# DBTITLE 1,Configuration
import json
import time
from datetime import datetime, timezone

CATALOG               = dbutils.widgets.get("catalog")
SCHEMA_NAME           = dbutils.widgets.get("schema")
REGISTERED_MODEL_NAME = dbutils.widgets.get("registered_model_name")
MODEL_ALIAS           = dbutils.widgets.get("model_alias")
ENDPOINT_BASE_NAME    = dbutils.widgets.get("endpoint_name")
WORKLOAD_TYPE         = dbutils.widgets.get("workload_type")
INPUT_TABLE           = dbutils.widgets.get("input_table")
OUTPUT_TABLE          = dbutils.widgets.get("output_table")
BATCH_SIZE            = int(dbutils.widgets.get("batch_size"))
OUTPUT_TABLE_UDF      = dbutils.widgets.get("output_table_udf")
PANDAS_PREDICT_BATCH_SIZE = int(dbutils.widgets.get("pandas_predict_batch_size"))

FULL_MODEL_NAME      = f"{CATALOG}.{SCHEMA_NAME}.{REGISTERED_MODEL_NAME}"
MODEL_URI            = f"models:/{FULL_MODEL_NAME}@{MODEL_ALIAS}"
_GPU_SUFFIX          = {
    "GPU_SMALL": "t4",
    "GPU_MEDIUM": "a10",
    "GPU_LARGE": "a100",
    "CPU": "cpu",
}
ENDPOINT_NAME        = (
    f"{ENDPOINT_BASE_NAME}-{_GPU_SUFFIX.get(WORKLOAD_TYPE, WORKLOAD_TYPE.lower())}"
)
INPUT_TABLE_FQN      = f"{CATALOG}.{SCHEMA_NAME}.{INPUT_TABLE}"
OUTPUT_TABLE_FQN     = f"{CATALOG}.{SCHEMA_NAME}.{OUTPUT_TABLE}"
OUTPUT_TABLE_UDF_FQN = f"{CATALOG}.{SCHEMA_NAME}.{OUTPUT_TABLE_UDF}"

MLFLOW_EXPERIMENT_ID = "2468185612186458"

import mlflow
mlflow.set_registry_uri("databricks-uc")

# Module-level timing variables (populated by inference cells)
pandas_elapsed: float = 0.0
udf_elapsed: float    = 0.0

print(f"Model URI        : {MODEL_URI}")
print(f"Serving endpoint : {ENDPOINT_NAME}")
print(f"Input table      : {INPUT_TABLE_FQN}")
print(f"Output table     : {OUTPUT_TABLE_FQN}")
print(f"UDF output table : {OUTPUT_TABLE_UDF_FQN}")
print(f"Experiment       : {MLFLOW_EXPERIMENT_ID}")
print(f"Batch size       : {BATCH_SIZE} rows/partition")
print(f"Pandas chunk size: {PANDAS_PREDICT_BATCH_SIZE}")
existing_count = spark.table(INPUT_TABLE_FQN).count()
print(f"Using existing input table: {INPUT_TABLE_FQN} ({existing_count} rows)")

# COMMAND ----------

# DBTITLE 1,Preview Input Data
display(spark.table(INPUT_TABLE_FQN))

# COMMAND ----------

# DBTITLE 1,Load Registered Model
model = mlflow.pyfunc.load_model(MODEL_URI)
print(f"✓ Loaded model: {MODEL_URI}")

# COMMAND ----------

# DBTITLE 1,Run Batch Inference + Log Traces to MLflow
from pyspark.sql import functions as F

input_df = spark.table(INPUT_TABLE_FQN)
input_pd = input_df.toPandas()
inference_ts = datetime.now(timezone.utc).isoformat()

def safe_parse(raw: str):
    """Return (parsed_json_str, error_str) — exactly one is non-None."""
    try:
        return json.dumps(json.loads(raw)), None
    except Exception as e:
        return None, str(e)

def predict_in_chunks(model, input_pd, chunk_size: int):
    """Run ChatModel prediction row-by-row in chunks to avoid GPU OOM on the driver."""
    outputs = []
    total_rows = len(input_pd)
    for start in range(0, total_rows, chunk_size):
        stop = min(start + chunk_size, total_rows)
        chunk = input_pd.iloc[start:stop]
        for _, row in chunk.iterrows():
            result = model.predict(
                {
                    "messages": [{"role": "user", "content": row["prompt"]}],
                    "custom_inputs": {"schema": row["schema"]},
                }
            )
            outputs.append(result["choices"][0]["message"]["content"])
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        print(f"  Predicted rows {start + 1}-{stop} of {total_rows}")
    return outputs

with mlflow.start_run(
    experiment_id=MLFLOW_EXPERIMENT_ID,
    run_name=f"batch_inference_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
) as run:
    run_id = run.info.run_id

    # ── Log run-level parameters ──────────────────────────────────────────────
    mlflow.log_params({
        "model_uri":        MODEL_URI,
        "model_alias":      MODEL_ALIAS,
        "input_table":      INPUT_TABLE_FQN,
        "output_table":     OUTPUT_TABLE_FQN,
        "total_rows":       len(input_pd),
        "inference_ts":     inference_ts,
    })

    # ── Run inference ──────────────────────────────────────────────────────────
    print(f"Running inference on {len(input_pd)} rows …")
    t0 = time.perf_counter()
    input_pd["extracted_json"] = predict_in_chunks(model, input_pd, PANDAS_PREDICT_BATCH_SIZE)
    t1 = time.perf_counter()
    pandas_elapsed = t1 - t0
    print(f"✓ Inference complete")
    print(f"Pandas inference: {len(input_pd)} rows in {pandas_elapsed:.2f}s  ({len(input_pd)/pandas_elapsed:.1f} rows/s)")

    parsed_col, error_col = zip(*input_pd["extracted_json"].apply(safe_parse))
    input_pd["parsed"]       = list(parsed_col)
    input_pd["parse_error"]  = list(error_col)
    input_pd["inference_ts"] = inference_ts
    input_pd["mlflow_run_id"] = run_id

    # ── Log per-row MLflow Traces ──────────────────────────────────────────────
    from mlflow.entities import SpanType
    for _, row in input_pd.iterrows():
        with mlflow.start_span(
            name="medgemma_predict",
            span_type=SpanType.LLM,
        ) as span:
            span.set_inputs({
                "prompt": row["prompt"],
                "schema": row["schema"],
            })
            span.set_outputs({
                "response": row["extracted_json"],
                "parsed":   row["parsed"],
            })
            span.set_attribute("row_id",        row["id"])
            span.set_attribute("category",      row.get("category", ""))
            span.set_attribute("parse_success", row["parse_error"] is None)
            if row["parse_error"] is not None:
                span.set_attribute("parse_error", row["parse_error"])

    # ── Log aggregate metrics ──────────────────────────────────────────────────
    total     = len(input_pd)
    successes = input_pd["parse_error"].isna().sum()
    failures  = total - successes
    parse_rate = successes / total if total > 0 else 0.0

    mlflow.log_metrics({
        "total_rows":             total,
        "parse_success":          int(successes),
        "parse_failure":          int(failures),
        "parse_success_rate":     parse_rate,
        "pandas_inference_time_s": pandas_elapsed,
        "pandas_rows_per_sec":    len(input_pd) / pandas_elapsed if pandas_elapsed > 0 else 0.0,
    })

    # ── Log per-category metrics ───────────────────────────────────────────────
    if "category" in input_pd.columns:
        for cat, grp in input_pd.groupby("category"):
            cat_ok = grp["parse_error"].isna().sum()
            mlflow.log_metric(f"parse_success_{cat}", int(cat_ok))
            mlflow.log_metric(f"total_rows_{cat}",    len(grp))

    print(f"\n✓ MLflow run: {run_id}")
    print(f"  parse_success_rate = {parse_rate:.1%}  ({successes}/{total})")

# COMMAND ----------

# DBTITLE 1,Write Results to Delta
result_sdf = spark.createDataFrame(input_pd)
# Explicit cast prevents NullType inference on all-null columns (e.g. parse_error when
# every row parses OK). Without this, DBR's edge-skip optimisation can remove the
# column from the physical scan plan, causing ATTRIBUTE_NOT_FOUND on subsequent reads.
result_sdf = result_sdf.withColumn("parse_error", F.col("parse_error").cast("string"))
result_sdf = result_sdf.withColumn("parsed",      F.col("parsed").cast("string"))

(
    result_sdf
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(OUTPUT_TABLE_FQN)
)

print(f"✓ Results written → {OUTPUT_TABLE_FQN}")
display(
    result_sdf.select("id", "category", "extracted_json", "parsed", "parse_error", "inference_ts", "mlflow_run_id")
)

# COMMAND ----------

# DBTITLE 1,Validate Output
output_df = spark.table(OUTPUT_TABLE_FQN)
total     = output_df.count()
successes = output_df.filter(F.col("parse_error").isNull()).count()
failures  = total - successes

print(f"Output table : {OUTPUT_TABLE_FQN}")
print(f"Total rows   : {total}")
print(f"Parsed OK    : {successes}  ({100*successes/total:.1f}%)")
print(f"Parse errors : {failures}")
print(f"MLflow run   : {run_id}")

if "category" in [f.name for f in output_df.schema.fields]:
    print("\nPer-category breakdown:")
    display(
        output_df
        .groupBy("category")
        .agg(
            F.count("*").alias("total"),
            F.sum(F.when(F.col("parse_error").isNull(), 1).otherwise(0)).alias("parsed_ok"),
        )
        .orderBy("category")
    )

# COMMAND ----------

# DBTITLE 1,Why Spark UDF Performance Varies With Scale
# MAGIC %md
# MAGIC ## Why the Spark UDF Appears Slower on Small Datasets
# MAGIC
# MAGIC The UDF pays a **fixed startup cost** that is irrelevant at scale but dominates on small batches:
# MAGIC
# MAGIC | Cost | Pandas (driver) | Spark UDF |
# MAGIC |---|---|---|
# MAGIC | Model load | Already in VRAM | Cold-load on each worker (~60–90 s) |
# MAGIC | Partitions | 1 | `max(1, N ÷ batch_size)` — spread across all worker GPUs |
# MAGIC | Serialisation | None | JVM ↔ Python per partition |
# MAGIC
# MAGIC **Break-even**: The UDF becomes faster once N is large enough that per-worker cold-start
# MAGIC time is amortised across many partitions. Below that threshold the pandas driver path is
# MAGIC faster; above it all GPUs stay busy and the UDF delivers near-linear throughput scaling
# MAGIC with the number of workers.
# MAGIC
# MAGIC **Partitioning**: `n_partitions = max(1, N // batch_size)` partitions are distributed
# MAGIC across W workers. Each worker processes its share in sequence, so all GPUs remain
# MAGIC utilised as long as `n_partitions >> W`.

# COMMAND ----------

# DBTITLE 1,Spark SQL ai_query() Inference
import gc
# Free GPU VRAM from the pandas model
del model
gc.collect()
try:
    import torch
    torch.cuda.empty_cache()
except Exception:
    pass

# ChatModel endpoints work with ai_query() which handles the chat format natively.
# ENDPOINT_NAME uses the same base-name + workload-suffix convention as
# src/medgemma/02_deploy_endpoint.py and src/medgemma/03_benchmark_endpoint.py.
udf_inference_ts = datetime.now(timezone.utc).isoformat()

# ai_query sends each row as a chat request to the serving endpoint
spark.sql(f"""
    CREATE OR REPLACE TABLE {OUTPUT_TABLE_UDF_FQN} AS
    SELECT
        *,
        ai_query(
            '{ENDPOINT_NAME}',
            prompt,
            modelParameters => named_struct('custom_inputs', named_struct('schema', schema))
        ) AS extracted_json,
        '{udf_inference_ts}' AS inference_ts
    FROM {INPUT_TABLE_FQN}
""")

t0_udf = time.perf_counter()
# Force materialization
udf_row_count = spark.table(OUTPUT_TABLE_UDF_FQN).count()
t1_udf = time.perf_counter()
udf_elapsed = t1_udf - t0_udf

print(f"ai_query() inference: {udf_row_count} rows")
print(f"✓ Results written → {OUTPUT_TABLE_UDF_FQN}")

# ── Compute parse success rate on ai_query output ─────────────────────────────
udf_output_pd = spark.table(OUTPUT_TABLE_UDF_FQN).select("extracted_json").toPandas()

udf_parse_ok = udf_output_pd["extracted_json"].apply(lambda x: safe_parse(x)[1] is None).sum()
udf_parse_rate = udf_parse_ok / udf_row_count if udf_row_count > 0 else 0.0
print(f"  parse_success_rate={udf_parse_rate:.1%}")

display(spark.table(OUTPUT_TABLE_UDF_FQN).select("id", "extracted_json", "inference_ts"))

# COMMAND ----------

# DBTITLE 1,Timing Comparison
pandas_rows_s = len(input_pd) / pandas_elapsed if pandas_elapsed > 0 else 0.0
udf_rows_s    = udf_row_count / udf_elapsed    if udf_elapsed > 0    else 0.0
speedup       = pandas_elapsed / udf_elapsed   if udf_elapsed > 0    else float("inf")

print("┌─────────────────────┬──────────────────┬──────────────────┐")
print("│ Method              │ Elapsed (s)      │ Rows/s           │")
print("├─────────────────────┼──────────────────┼──────────────────┤")
print(f"│ Pandas (driver)     │ {pandas_elapsed:<16.2f} │ {pandas_rows_s:<16.1f} │")
print(f"│ ai_query()          │ {udf_elapsed:<16.2f} │ {udf_rows_s:<16.1f} │")
print("└─────────────────────┴──────────────────┴──────────────────┘")
print(f"\nSpeedup (pandas / ai_query): {speedup:.2f}x")

# Log ai_query metrics as nested child run, then speedup to parent
udf_run_name = f"batch_inference_ai_query_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
with mlflow.start_run(run_id=run_id):
    with mlflow.start_run(run_name=udf_run_name, nested=True) as udf_run:
        mlflow.log_params({
            "model_uri":        MODEL_URI,
            "input_table":      INPUT_TABLE_FQN,
            "output_table_udf": OUTPUT_TABLE_UDF_FQN,
            "total_rows":       udf_row_count,
        })
        mlflow.log_metrics({
            "ai_query_inference_time_s": udf_elapsed,
            "ai_query_rows_per_sec":     udf_row_count / udf_elapsed if udf_elapsed > 0 else 0.0,
            "parse_success_rate":   udf_parse_rate,
        })
        print(f"✓ MLflow ai_query run (nested): {udf_run.info.run_id}  parse_success_rate={udf_parse_rate:.1%}")
    mlflow.log_metric("pandas_vs_ai_query_speedup", speedup)
