# Databricks notebook source
# DBTITLE 1,Overview
# MAGIC %md
# MAGIC # MedGemma Batch Inference — Serverless GPU Compute (SGC)
# MAGIC
# MAGIC Distributed batch inference using **Ray Data + vLLM** on Serverless GPU Compute.
# MAGIC XGrammar-guided JSON decoding produces structured output from medical text — no
# MAGIC cluster to manage, auto-scaling GPU workers.
# MAGIC
# MAGIC **Input**: Delta table `{catalog}.{schema}.{input_table}` with columns:
# MAGIC - `id` (string) — unique row identifier
# MAGIC - `category` (string) — record category
# MAGIC - `prompt` (string) — medical text / instruction
# MAGIC - `schema` (string) — JSON-serialised JSON schema for extraction
# MAGIC
# MAGIC **Output**: Delta table `{catalog}.{schema}.{output_table}` with the original
# MAGIC columns plus:
# MAGIC - `extracted_json` (string) — raw model output
# MAGIC - `parsed` (string) — re-serialised parsed JSON, or null on failure
# MAGIC - `parse_error` (string) — error message when parsing fails
# MAGIC - `inference_ts` (string) — wall-clock time of inference

# COMMAND ----------

# DBTITLE 1,Widgets
dbutils.widgets.removeAll()
dbutils.widgets.text("catalog",      "pavan_naidu_catalog",          "Catalog")
dbutils.widgets.text("schema",       "mlops",                        "Schema")
dbutils.widgets.text("hf_model_name", "google/medgemma-1.5-4b-it", "HuggingFace Model Name")
dbutils.widgets.text("input_table",  "medgemma_batch_input",         "Input Delta Table")
dbutils.widgets.text("output_table", "medgemma_batch_output_sgc",    "Output Delta Table")
dbutils.widgets.text("num_gpus",        "2",                            "Number of GPU Workers")
dbutils.widgets.text("batch_size",      "96",                           "Rows per Ray batch")
dbutils.widgets.dropdown("optimization_mode", "throughput", ["wall_clock", "throughput"], "vLLM Optimization Mode")
dbutils.widgets.text("experiment_name", "/Users/pavan.naidu@databricks.com/medgemma-xgrammar", "MLflow Experiment Name")

# COMMAND ----------

# DBTITLE 1,Install Flash Attention
# MAGIC %pip install --force-reinstall --no-cache-dir --no-deps "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"

# COMMAND ----------

# DBTITLE 1,Install vLLM + Ray + Dependencies
# MAGIC %pip install "vllm==0.8.5.post1" "ray[data]>=2.47.1" "transformers<4.54.0" "optree>=0.13.0" hf_transfer "numpy==1.26.4"

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# DBTITLE 1,Verify Versions
from packaging.version import Version
import torch
import vllm
import ray
import transformers

print(f"PyTorch:         {torch.__version__}")
print(f"vLLM:            {vllm.__version__}")
print(f"Ray:             {ray.__version__}")
print(f"Transformers:    {transformers.__version__}")

try:
    import flash_attn
    print(f"Flash Attention: {flash_attn.__version__}")
except ImportError:
    print("Flash Attention: not installed (vLLM will use built-in attention)")

assert Version(ray.__version__) >= Version("2.47.1"), "Ray version must be at least 2.47.1"
print("\n✓ All version checks passed!")

# COMMAND ----------

# DBTITLE 1,Configuration
CATALOG      = dbutils.widgets.get("catalog")
SCHEMA_NAME  = dbutils.widgets.get("schema")
HF_MODEL_NAME = dbutils.widgets.get("hf_model_name")
MODEL_PATH   = f"/Volumes/{dbutils.widgets.get('catalog')}/{dbutils.widgets.get('schema')}/model_weights/{HF_MODEL_NAME}/bf16"
INPUT_TABLE  = dbutils.widgets.get("input_table")
OUTPUT_TABLE = dbutils.widgets.get("output_table")
NUM_GPUS        = int(dbutils.widgets.get("num_gpus"))
RAY_BATCH_SIZE  = int(dbutils.widgets.get("batch_size"))
OPTIMIZATION_MODE = dbutils.widgets.get("optimization_mode").strip().lower()
EXPERIMENT_NAME = dbutils.widgets.get("experiment_name")

if OPTIMIZATION_MODE not in {"wall_clock", "throughput"}:
    raise ValueError(f"optimization_mode must be one of ['wall_clock', 'throughput'], got {OPTIMIZATION_MODE!r}")

INPUT_TABLE_FQN  = f"{CATALOG}.{SCHEMA_NAME}.{INPUT_TABLE}"
OUTPUT_TABLE_FQN = f"{CATALOG}.{SCHEMA_NAME}.{OUTPUT_TABLE}"
PARQUET_OUTPUT_PATH = f"/Volumes/{CATALOG}/{SCHEMA_NAME}/ray_data/medgemma_sgc_output/"

import mlflow
mlflow.set_experiment(EXPERIMENT_NAME)

print(f"Model path       : {MODEL_PATH}")
print(f"Input table      : {INPUT_TABLE_FQN}")
print(f"Output table     : {OUTPUT_TABLE_FQN}")
print(f"Parquet staging  : {PARQUET_OUTPUT_PATH}")
print(f"GPU workers      : {NUM_GPUS}")
print(f"Ray batch size   : {RAY_BATCH_SIZE}")
print(f"Optimisation mode: {OPTIMIZATION_MODE}")
print(f"MLflow experiment : {EXPERIMENT_NAME}")

# COMMAND ----------

# DBTITLE 1,Prepare Text-Only Model Weights
# MedGemma ships as Gemma3ForConditionalGeneration (multimodal).  vLLM 0.8.5
# can't load it natively (tokenizer lacks image_token_id) and overriding to
# Gemma3ForCausalLM fails because weight keys have a "language_model." prefix.
#
# Fix: one-time conversion that strips the prefix from safetensors files and
# writes a text-only config.  Saved alongside the original weights so it only
# runs once.
import json, pathlib, shutil

TEXT_MODEL_PATH = MODEL_PATH.rstrip("/") + "_text_only"
_marker = pathlib.Path(TEXT_MODEL_PATH) / ".conversion_done"

if _marker.exists() and (pathlib.Path(TEXT_MODEL_PATH) / "config.json").exists():
    # Verify existing config has rope_theta (re-convert if from a prior buggy run)
    with open(pathlib.Path(TEXT_MODEL_PATH) / "config.json") as _f:
        _existing = json.load(_f)
    if "rope_theta" not in _existing or not (pathlib.Path(TEXT_MODEL_PATH) / "chat_template.jinja").exists():
        print("⚠ Existing text-only model incomplete — re-converting ...")
        import shutil as _shutil
        _shutil.rmtree(TEXT_MODEL_PATH, ignore_errors=True)

if _marker.exists():
    print(f"✓ Text-only weights already exist at {TEXT_MODEL_PATH}")
else:
    print(f"Converting multimodal weights → text-only at {TEXT_MODEL_PATH} ...")
    from safetensors.torch import load_file, save_file

    src = pathlib.Path(MODEL_PATH)
    dst = pathlib.Path(TEXT_MODEL_PATH)
    dst.mkdir(parents=True, exist_ok=True)

    # 1. Write text-only config.json (promote text_config to top level)
    #    MedGemma's text_config uses "rope_parameters" (nested dict) instead of
    #    the standard rope_theta / rope_local_base_freq / rope_scaling that
    #    vLLM's Gemma3ForCausalLM expects.  Translate them here.
    with open(src / "config.json") as f:
        full_cfg = json.load(f)
    text_cfg = full_cfg["text_config"]
    text_cfg["architectures"] = ["Gemma3ForCausalLM"]
    text_cfg["model_type"] = "gemma3_text"

    # Translate rope_parameters → standard fields if needed
    rope_params = text_cfg.pop("rope_parameters", None)
    if rope_params and "rope_theta" not in text_cfg:
        full_attn = rope_params.get("full_attention", {})
        slide_attn = rope_params.get("sliding_attention", {})
        text_cfg["rope_theta"] = full_attn.get("rope_theta", 1000000)
        text_cfg["rope_local_base_freq"] = slide_attn.get("rope_theta", 10000)
        text_cfg["rope_scaling"] = {
            "factor": full_attn.get("factor", 8.0),
            "rope_type": full_attn.get("rope_type", "linear"),
        }

    # Normalise _sliding_window_pattern → sliding_window_pattern
    if "_sliding_window_pattern" in text_cfg and "sliding_window_pattern" not in text_cfg:
        text_cfg["sliding_window_pattern"] = text_cfg.pop("_sliding_window_pattern")

    with open(dst / "config.json", "w") as f:
        json.dump(text_cfg, f, indent=2)

    # 2. Copy tokenizer + chat template files
    for name in src.iterdir():
        if name.name.startswith("tokenizer") or name.name.startswith("chat_template") or name.name in (
            "special_tokens_map.json", "generation_config.json",
        ):
            shutil.copy2(name, dst / name.name)

    # 3. Resave safetensors with stripped "language_model." prefix
    index_path = src / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        new_weight_map = {}
        shard_files = set(index["weight_map"].values())
        for shard_name in sorted(shard_files):
            print(f"  Processing {shard_name} ...")
            tensors = load_file(str(src / shard_name))
            renamed = {}
            for k, v in tensors.items():
                new_key = k.replace("language_model.", "", 1) if k.startswith("language_model.") else k
                # Skip vision encoder weights (not needed for text-only)
                if new_key.startswith("vision_tower.") or new_key.startswith("multi_modal_projector."):
                    continue
                renamed[new_key] = v
                new_weight_map[new_key] = shard_name
            save_file(renamed, str(dst / shard_name))
        new_index = {"metadata": index.get("metadata", {}), "weight_map": new_weight_map}
        with open(dst / "model.safetensors.index.json", "w") as f:
            json.dump(new_index, f, indent=2)
    else:
        # Single safetensors file
        tensors = load_file(str(src / "model.safetensors"))
        renamed = {
            (k.replace("language_model.", "", 1) if k.startswith("language_model.") else k): v
            for k, v in tensors.items()
            if not k.startswith("vision_tower.") and not k.startswith("multi_modal_projector.")
        }
        save_file(renamed, str(dst / "model.safetensors"))

    _marker.touch()
    print(f"✓ Conversion complete → {TEXT_MODEL_PATH}")

# COMMAND ----------

# DBTITLE 1,Read Input Data
import time
from pyspark.sql import functions as F

phase_timings = {}
phase_start = time.perf_counter()
input_sdf = spark.table(INPUT_TABLE_FQN)

required_cols = {"prompt", "schema"}
missing_cols = required_cols - set(input_sdf.columns)
if missing_cols:
    raise ValueError(f"Input table must contain columns: {sorted(required_cols)}. Missing: {sorted(missing_cols)}")

if "id" not in input_sdf.columns:
    input_sdf = input_sdf.withColumn("id", F.monotonically_increasing_id().cast("string"))
if "category" not in input_sdf.columns:
    input_sdf = input_sdf.withColumn("category", F.lit("default"))

input_pd = input_sdf.toPandas()
input_items = input_pd.to_dict("records")
phase_timings["input_load_s"] = time.perf_counter() - phase_start

print(f"✓ Loaded {len(input_items)} rows from {INPUT_TABLE_FQN}")
display(input_sdf.limit(5))

# COMMAND ----------

# DBTITLE 1,Distributed Inference (Ray Data + vLLM)
from serverless_gpu.ray import ray_launch
import os

os.environ['RAY_TEMP_DIR'] = '/tmp/ray'

@ray_launch(gpus=NUM_GPUS, gpu_type='a10', remote=True)
def run_medgemma_inference():
    """Distributed MedGemma inference with XGrammar-guided JSON decoding."""
    from typing import Dict, List
    from datetime import datetime
    import json
    import os
    import time
    import numpy as np
    import ray
    from ray.data import ActorPoolStrategy
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

    wall_clock_mode = OPTIMIZATION_MODE == "wall_clock"
    target_blocks = max(1, NUM_GPUS * 2)

    dataset_start = time.perf_counter()
    ds = ray.data.from_items(input_items)
    ds = ds.repartition(target_blocks, shuffle=False)
    dataset_creation_s = time.perf_counter() - dataset_start

    class MedGemmaPredictor:
        """vLLM-based predictor with per-row XGrammar guided decoding."""

        def __init__(self):
            os.environ["VLLM_USE_V1"] = "0"
            actor_init_start = time.perf_counter()
            self.actor_id = f"{ray.get_runtime_context().get_worker_id()}:{os.getpid()}"

            # Use the pre-converted text-only weights (Gemma3ForCausalLM).
            self.llm = LLM(
                model=TEXT_MODEL_PATH,
                tensor_parallel_size=1,
                dtype="bfloat16",
                trust_remote_code=True,
                gpu_memory_utilization=0.90,
                max_model_len=8192,
                enable_prefix_caching=True,
                enable_chunked_prefill=True,
                max_num_batched_tokens=8192,
                enforce_eager=wall_clock_mode,
            )
            self.tokenizer = self.llm.get_tokenizer()
            self.actor_init_s = time.perf_counter() - actor_init_start
            print(
                f"✓ vLLM engine initialised actor={self.actor_id} "
                f"init_s={self.actor_init_s:.2f} mode={OPTIMIZATION_MODE} model={MODEL_PATH}"
            )

        def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
            batch_start = time.perf_counter()
            prompts = batch["prompt"].tolist()
            schemas = batch["schema"].tolist()
            ids = batch["id"].tolist()
            categories = batch["category"].tolist()

            formatted_prompts: List[str] = []
            param_list: List[SamplingParams] = []
            valid_rows = []
            results = [
                {
                    "id": row_id,
                    "category": category,
                    "prompt": prompt,
                    "schema": schema,
                    "extracted_json": None,
                    "parse_error": None,
                }
                for row_id, category, prompt, schema in zip(ids, categories, prompts, schemas)
            ]

            for idx, (prompt_text, schema_str) in enumerate(zip(prompts, schemas)):
                formatted = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt_text}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                try:
                    if isinstance(schema_str, dict):
                        schema_obj = schema_str
                    elif isinstance(schema_str, str):
                        schema_obj = json.loads(schema_str)
                    else:
                        raise TypeError(f"unsupported schema type: {type(schema_str).__name__}")
                except Exception as exc:
                    results[idx]["parse_error"] = f"invalid_schema: {exc}"
                    continue

                valid_rows.append(idx)
                formatted_prompts.append(formatted)
                guided = GuidedDecodingParams(
                    json=schema_obj,
                    backend="xgrammar",
                )
                param_list.append(SamplingParams(
                    max_tokens=512,
                    temperature=0.7,
                    guided_decoding=guided,
                ))

            generate_start = time.perf_counter()
            outputs = self.llm.generate(formatted_prompts, sampling_params=param_list) if formatted_prompts else []
            generate_s = time.perf_counter() - generate_start
            ts = datetime.now().isoformat()
            batch_elapsed_s = time.perf_counter() - batch_start

            for idx, output in zip(valid_rows, outputs):
                results[idx]["extracted_json"] = output.outputs[0].text

            print(
                f"actor={self.actor_id} rows={len(prompts)} valid_rows={len(valid_rows)} "
                f"generate_s={generate_s:.2f} batch_s={batch_elapsed_s:.2f}"
            )

            return {
                "id": [row["id"] for row in results],
                "category": [row["category"] for row in results],
                "prompt": [row["prompt"] for row in results],
                "schema": [row["schema"] for row in results],
                "extracted_json": [row["extracted_json"] for row in results],
                "inference_error": [row["parse_error"] for row in results],
                "inference_ts": [ts] * len(results),
                "__actor_id": [self.actor_id] * len(results),
                "__actor_init_s": [self.actor_init_s] * len(results),
                "__generate_s": [generate_s] * len(results),
                "__batch_elapsed_s": [batch_elapsed_s] * len(results),
            }

    distributed_start = time.perf_counter()
    ds = ds.map_batches(
        MedGemmaPredictor,
        compute=ActorPoolStrategy(size=NUM_GPUS),
        batch_size=RAY_BATCH_SIZE,
        num_gpus=1,
    )
    write_start = time.perf_counter()
    ds.write_parquet(PARQUET_OUTPUT_PATH, mode="overwrite")
    parquet_write_s = time.perf_counter() - write_start

    return {
        "parquet_path": PARQUET_OUTPUT_PATH,
        "dataset_creation_s": dataset_creation_s,
        "distributed_compute_s": time.perf_counter() - distributed_start,
        "parquet_write_s": parquet_write_s,
        "target_blocks": target_blocks,
    }

# COMMAND ----------

# DBTITLE 1,Execute Distributed Inference
t0 = time.perf_counter()
result = run_medgemma_inference.distributed()
ray_result = result[0] if isinstance(result, (list, tuple)) else result
sgc_elapsed = time.perf_counter() - t0
parquet_path = ray_result["parquet_path"]
phase_timings["ray_dataset_creation_s"] = ray_result["dataset_creation_s"]
phase_timings["distributed_compute_s"] = ray_result["distributed_compute_s"]
phase_timings["ray_parquet_write_s"] = ray_result["parquet_write_s"]
phase_timings["distributed_total_s"] = sgc_elapsed

print(f"✓ Distributed inference complete in {sgc_elapsed:.2f}s")
print(f"  Parquet output: {parquet_path}")
print(f"  Target Ray blocks: {ray_result['target_blocks']}")

# COMMAND ----------

# DBTITLE 1,Load Parquet → Parse → Write Delta
import json

def safe_parse(raw: str):
    """Return (parsed_json_str, error_str) — exactly one is non-None."""
    if raw is None:
        return None, "Model returned no output"
    try:
        return json.dumps(json.loads(raw)), None
    except Exception as e:
        return None, str(e)

phase_start = time.perf_counter()
result_sdf = spark.read.parquet(parquet_path)
result_pd = result_sdf.toPandas()
phase_timings["parquet_read_s"] = time.perf_counter() - phase_start

phase_start = time.perf_counter()
parsed_col, parse_error_col = zip(*result_pd["extracted_json"].apply(safe_parse))
phase_timings["parse_json_s"] = time.perf_counter() - phase_start

result_pd["parsed"] = list(parsed_col)
result_pd["parse_error"] = list(parse_error_col)
if "inference_error" in result_pd.columns:
    result_pd["parse_error"] = result_pd["parse_error"].combine_first(result_pd["inference_error"])

actor_count = 0
max_actor_init_s = 0.0
sum_generate_s = 0.0
sum_batch_elapsed_s = 0.0
timing_cols = ["__actor_id", "__actor_init_s", "__generate_s", "__batch_elapsed_s"]
if set(timing_cols).issubset(result_pd.columns):
    actor_count = result_pd["__actor_id"].nunique()
    max_actor_init_s = float(result_pd["__actor_init_s"].max())
    sum_generate_s = float(result_pd["__generate_s"].sum())
    sum_batch_elapsed_s = float(result_pd["__batch_elapsed_s"].sum())

phase_timings["actor_init_max_s"] = max_actor_init_s
phase_timings["generate_accumulated_s"] = sum_generate_s
phase_timings["batch_elapsed_accumulated_s"] = sum_batch_elapsed_s
result_pd = result_pd.drop(columns=[col for col in ["inference_error"] + timing_cols if col in result_pd.columns], errors="ignore")

phase_start = time.perf_counter()
result_sdf = spark.createDataFrame(result_pd)
result_sdf = result_sdf.withColumn("parse_error", F.col("parse_error").cast("string"))
result_sdf = result_sdf.withColumn("parsed", F.col("parsed").cast("string"))

(
    result_sdf
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(OUTPUT_TABLE_FQN)
)
phase_timings["delta_write_s"] = time.perf_counter() - phase_start

total = len(result_pd)
successes = result_pd["parse_error"].isna().sum()
print(f"✓ Results written → {OUTPUT_TABLE_FQN}")
success_pct = (100 * successes / total) if total > 0 else 0.0
print(f"  {total} rows, {successes} parsed OK ({success_pct:.1f}%)")

# COMMAND ----------

# DBTITLE 1,Log to MLflow
import mlflow
mlflow.set_registry_uri("databricks-uc")

from datetime import datetime, timezone
from mlflow.entities import SpanType

with mlflow.start_run(
    run_name=f"batch_inference_sgc_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
) as run:
    run_id = run.info.run_id

    mlflow.log_params({
        "method":        "sgc_ray_vllm",
        "model_path":    MODEL_PATH,
        "input_table":   INPUT_TABLE_FQN,
        "output_table":  OUTPUT_TABLE_FQN,
        "num_gpus":      NUM_GPUS,
        "ray_batch_size": RAY_BATCH_SIZE,
        "optimization_mode": OPTIMIZATION_MODE,
        "ray_target_blocks": ray_result["target_blocks"],
        "total_rows":    total,
    })

    mlflow.log_metrics({
        "total_rows":             total,
        "parse_success":          int(successes),
        "parse_failure":          int(total - successes),
        "parse_success_rate":     successes / total if total > 0 else 0.0,
        "sgc_inference_time_s":   sgc_elapsed,
        "sgc_rows_per_sec":       total / sgc_elapsed if sgc_elapsed > 0 else 0.0,
        "actor_count":            actor_count,
        "actor_init_max_s":       max_actor_init_s,
        "generate_accumulated_s": sum_generate_s,
        "batch_elapsed_accumulated_s": sum_batch_elapsed_s,
    })
    mlflow.log_metrics(phase_timings)

    # Per-row traces
    for _, row in result_pd.iterrows():
        with mlflow.start_span(
            name="medgemma_predict_sgc",
            span_type=SpanType.LLM,
        ) as span:
            span.set_inputs({"prompt": row["prompt"], "schema": row["schema"]})
            span.set_outputs({"response": row["extracted_json"], "parsed": row["parsed"]})
            span.set_attribute("row_id", row["id"])
            span.set_attribute("category", row.get("category", ""))
            span.set_attribute("parse_success", row["parse_error"] is None)
            if row["parse_error"] is not None:
                span.set_attribute("parse_error", row["parse_error"])

    # Per-category metrics
    if "category" in result_pd.columns:
        for cat, grp in result_pd.groupby("category"):
            cat_ok = grp["parse_error"].isna().sum()
            mlflow.log_metric(f"parse_success_{cat}", int(cat_ok))
            mlflow.log_metric(f"total_rows_{cat}", len(grp))

    print(f"✓ MLflow run: {run_id}")
    success_rate = (successes / total) if total > 0 else 0.0
    print(f"  parse_success_rate = {success_rate:.1%}  ({successes}/{total})")

# COMMAND ----------

# DBTITLE 1,Validate Output
phase_start = time.perf_counter()
output_df = spark.table(OUTPUT_TABLE_FQN)
total_out = output_df.count()
successes_out = output_df.filter(F.col("parse_error").isNull()).count()
failures_out = total_out - successes_out
phase_timings["validation_s"] = time.perf_counter() - phase_start

with mlflow.start_run(run_id=run_id):
    mlflow.log_metric("validation_s", phase_timings["validation_s"])

print(f"Output table : {OUTPUT_TABLE_FQN}")
print(f"Total rows   : {total_out}")
parsed_pct = (100 * successes_out / total_out) if total_out > 0 else 0.0
print(f"Parsed OK    : {successes_out}  ({parsed_pct:.1f}%)")
print(f"Parse errors : {failures_out}")
rows_per_sec = (total_out / sgc_elapsed) if sgc_elapsed > 0 else 0.0
print(f"SGC elapsed  : {sgc_elapsed:.2f}s  ({rows_per_sec:.1f} rows/s)")
print("Phase timings:")
for phase_name, phase_value in phase_timings.items():
    print(f"  {phase_name}: {phase_value:.2f}s")

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

from jsonschema import validate

if total_out != len(input_items) or total_out == 0 or failures_out:
    raise AssertionError(f"Invalid batch output: input={len(input_items)}, output={total_out}, failures={failures_out}")
for row in result_pd.to_dict("records"):
    validate(instance=json.loads(row["extracted_json"]), schema=json.loads(row["schema"]))
dbutils.notebook.exit(json.dumps({
    "output_table": OUTPUT_TABLE_FQN,
    "rows": total_out,
    "schema_valid_predictions": total_out,
    "mlflow_run_id": run_id,
}))
