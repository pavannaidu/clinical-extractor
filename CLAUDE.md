# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Databricks Asset Bundle (DAB) project that deploys **Google MedGemma 1.5 4B IT** with **XGrammar constrained generation** to produce structured JSON from medical text. Also supports vLLM-based serving for MedGemma and Qwen3 models.

Workspace:
- **FEVM** (default): `fevm-pavan-naidu.cloud.databricks.com`, profile `FEVM`

## DABs Commands

```bash
databricks bundle validate --profile FEVM
databricks bundle deploy --profile FEVM

# MedGemma pipeline
databricks bundle run medgemma_register    --profile FEVM
databricks bundle run medgemma_deploy      --profile FEVM
databricks bundle run medgemma_benchmark   --profile FEVM

# Batch
databricks bundle run prepare_batch_input  --profile FEVM
databricks bundle run batch_inference_udf  --profile FEVM
databricks bundle run batch_inference_sgc  --profile FEVM

# vLLM
databricks bundle run medgemma_vllm        --profile FEVM
databricks bundle run inference_vllm_qwen  --profile FEVM

# Fine-tuning
databricks bundle run finetune_medgemma    --profile FEVM

# Databricks App (review, deploy, then start)
databricks bundle plan   --select apps.clinical_extractor --target dev --profile FEVM
databricks bundle deploy --select apps.clinical_extractor --target dev --profile FEVM
databricks bundle run clinical_extractor                  --target dev --profile FEVM
```

All jobs use the `dev` target (mode: development). Bundle name: `medgemma`.

## Pipeline Architecture

Two model-serving paths, plus batch inference and a web app. Files are numbered by execution order within each pipeline.

```
MedGemma:  src/medgemma/01_register.py (SGC: download + register) -> 02_deploy_endpoint.py -> 03_benchmark_endpoint.py
           src/medgemma/model.py (MLflow ChatModel loaded at serving time)
vLLM:      src/vllm/medgemma_vllm.py (SGC: self-contained download, serve, test, register, deploy)
           src/vllm/inference_vllm_qwen.py (same flow for Qwen3-4B)
Batch:     src/batch/01_prepare_batch_input.py -> 02_batch_inference_udf.py (pandas + ai_query)
                                               -> 03_batch_inference_sgc.py (SGC + Ray)
Finetune:  src/finetune/finetune_medgemma.py (SGC: generate data, QLoRA train, merge, register, deploy)
App:       src/app/ (FastAPI + React)
```

All `.py` files under `src/` (except `src/app/`) are Databricks notebooks in Python source format (`# COMMAND ----------` separators, `# MAGIC %md` markdown cells). The one exception is `src/medgemma/model.py` which is a regular Python module logged as an MLflow artifact.

## Databricks App (`src/app/`)

FastAPI backend + React/Vite frontend for interactive structured JSON extraction.

### Dev commands

```bash
# Frontend dev (Vite dev server proxies /api to localhost:8000)
cd src/app/frontend && npm install && npm run dev

# Build frontend (output goes to src/app/static/)
cd src/app/frontend && npm run build

# Run backend locally
cd src/app && uvicorn main:app --port 8000

# Run backend tests
cd src/app && python -m pytest tests/
```

### Architecture

- `main.py` — FastAPI entrypoint, mounts SPA from `static/`, initializes Lakebase DB on startup (falls back to hardcoded presets if unavailable)
- `api/routes.py` — API routes under `/api/v1` for models, workspace endpoint discovery and model settings, presets, inference, and load testing
- `api/schemas.py` — Pydantic models: `InferenceRequest`, `InferenceResponse`, `Preset`, `PresetCreate`, `PresetUpdate`
- `api/presets.py` — Hardcoded preset configurations (fallback when Lakebase is unavailable)
- `api/database.py` — Lakebase (PostgreSQL via psycopg2) CRUD for presets. It reads `LAKEBASE_DB_*` when operators configure a database binding and environment values; the repository does not currently provide that binding
- `models.json` — Seeds the default serving-endpoint selection and defines endpoint API contracts and per-endpoint settings (max_tokens, timeout, temperature). A saved model selection can override the default list at runtime.
- `app.yaml` — Databricks App deployment config

### Endpoint Contract System

The app supports three ways to pass JSON schemas to serving endpoints, configured per-endpoint in `models.json` under `contracts`:

| Contract | Used by | How schema is sent |
|---|---|---|
| `custom_inputs_schema` (default) | MedGemma XGrammar endpoints | `custom_inputs.schema` field |
| `ppt_json_schema` | `databricks-*` PPT endpoints | `response_format.json_schema` |
| `openai_guided_json` | vLLM endpoints | `guided_json` + `guided_decoding_backend` fields |

Default: endpoints starting with `databricks-` get `ppt_json_schema`, all others get `custom_inputs_schema`.

### Frontend components

React 18 + TypeScript + Vite. Uses Monaco Editor for JSON schema editing.

- `App.tsx` — Main layout: schema editor (visual/JSON modes), prompt input, model comparison slots (up to 3)
- `SchemaBuilder.tsx` — Visual JSON schema builder
- `ModelResultColumn.tsx` — Single model result display
- `FormattedView.tsx` — Formatted JSON result renderer
- `PresetManager.tsx` — Preset CRUD modal

## XGrammar Constrained Generation

`MedGemmaChatModel.load_context` in `src/medgemma/model.py` builds a `xgr.GrammarCompiler` from the tokenizer. At inference, `predict` extracts the JSON schema from `params.custom_inputs["schema"]` and compiles it into a `LogitsProcessor` that constrains token sampling to valid JSON. If schema is omitted from `custom_inputs`, generation is unconstrained. Compiled grammars are cached per schema string.

## MLflow Model Logging Pattern

`01_register.py` uses code-based model logging:

```python
mlflow.pyfunc.log_model(
    python_model=model_code_path,  # path to model.py (co-located in same directory)
    artifacts={"model-weights": LOCAL_MODEL_PATH},
    metadata={"task": "llm/v1/chat"},
    ...
)
```

`model.py` is what MLflow loads at serving time, so inference logic changes must be made there. It must stay co-located with `01_register.py` (same directory) because the path is resolved dynamically.

## Key Configuration

| Variable | Default | Source |
|---|---|---|
| `catalog` | `pavan_naidu_catalog` | `databricks.yml` |
| `schema` | `mlops` | `databricks.yml` |
| MedGemma model path | `/Volumes/${catalog}/${schema}/model_weights/${hf_model_name}/bf16` | Derived in `src/medgemma/01_register.py`; the `model_weights` UC volume must already exist |
| MedGemma registered model | `medgemma_xgrammar_structured` | `databricks.yml` `medgemma_registered_model_name` |
| MedGemma endpoint base | `medgemma-1_5-4b-xgrammar` | `databricks.yml` `medgemma_endpoint_name`; deployment appends the workload suffix (`-a10` by default) |
| MedGemma vLLM endpoint | `medgemma-1-5-4b-vllm-endpoint` | `databricks.yml` `medgemma_vllm.endpoint_name` |
| App name | `clinical-extractor` | `databricks.yml` `resources.apps.clinical_extractor.name` |
| Batch UDF GPU node type | `g5.2xlarge` (2 workers) | `databricks.yml` `gpu_node_type`, `batch_inference_udf.job_clusters` |
| `MLFLOW_EXPERIMENT_ID` | `2468185612186458` | `src/batch/02_batch_inference_udf.py`, `src/medgemma/03_benchmark_endpoint.py` |
| Fine-tuned registered model | `medgemma_xgrammar_finetuned` | `databricks.yml` `finetune_registered_model_name` |
| HuggingFace token | Secret `pavan_naidu/hf_token` | `medgemma/01_register.py`, `vllm/medgemma_vllm.py`, `vllm/inference_vllm_qwen.py`, `finetune/finetune_medgemma.py` |

## Important Constraints

- bitsandbytes-quantized models cannot be serialized by MLflow; code-based `python_model` logging avoids serializing the model object.
- `model.py` auto-selects dtype by GPU: bf16 on SM >= 8.0 (A10/A100), NF4 4-bit on SM 7.5 (T4).
- `batch_inference_*` notebooks call `mlflow.set_registry_uri("databricks-uc")`; avoid duplicate registry URI resets across cells.
- Keep notebook widget names aligned with `databricks.yml` base parameters.
- The `model_weights` Unity Catalog volume must exist before running registration or the SGC batch path. With the defaults, create `pavan_naidu_catalog.mlops.model_weights`; override catalog/schema consistently when using another target.
- The Databricks App (`clinical-extractor`) is defined in `databricks.yml`. A targeted `bundle deploy` uploads it; run `databricks bundle run clinical_extractor` afterward to start a deployment, or use the Databricks Apps deployment flow.
- The repository currently has no Lakebase resource binding or `LAKEBASE_DB_*` configuration. The app therefore falls back to hardcoded presets and in-memory model settings unless operators add a Lakebase binding and matching environment configuration.
- The classic batch UDF job uses a two-worker `g5.2xlarge` job cluster. Serverless GPU jobs instead request `GPU_1xA10` compute directly.
