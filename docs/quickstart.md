# Quickstart

This repo is organized as a showcase first and an implementation workspace second. If you only have a few minutes, use the steps below.

## 1. Start with the Story

- Read [../README.md](../README.md)
- Read [../showcase/decision-framework.md](../showcase/decision-framework.md)
- Pick one workload under [../showcase/workloads](../showcase/workloads)

## 2. Prepare Workspace Prerequisites

The model registration and Serverless GPU batch paths write/read model weights under:

```text
/Volumes/<catalog>/<schema>/model_weights/<hugging-face-model>/bf16
```

Create the `model_weights` Unity Catalog volume before running those jobs. For the default target, run the following in a Databricks SQL editor or notebook with permission to create volumes:

```sql
CREATE VOLUME IF NOT EXISTS pavan_naidu_catalog.mlops.model_weights;
```

If you override `catalog` or `schema`, create the volume in that location instead. MedGemma model downloads and fine-tuning require the `hf_token` secret in the `pavan_naidu` secret scope. The public Qwen3 model and NCBI Disease dataset do not require a Hugging Face token.

The interactive app path does not require this volume.

## 3. Validate the Bundle

```bash
databricks bundle validate --profile FEVM
```

The bundle keeps the current runnable code paths intact and uses FEVM as the default workspace.

## 4. Choose a Demo Path

### Path A: Interactive comparison app

Use this when you want to show the structured-output gap and compare FMAPI vs custom endpoints.

```bash
databricks bundle plan --select apps.clinical_extractor --target dev --profile FEVM
databricks bundle deploy --select apps.clinical_extractor --target dev --profile FEVM
databricks bundle run clinical_extractor --target dev --profile FEVM
```

Then use the `clinical-extractor` app and select a workload pack from `showcase/workloads` as the schema and prompt source.

Reviewing the targeted plan first avoids applying unrelated job changes. The subsequent run creates a fresh app deployment and ensures compute is started.

### Path B: Custom structured extraction with MedGemma + XGrammar

Use this when you want to show domain-specific extraction with guaranteed JSON output.

```bash
databricks bundle run medgemma_register  --profile FEVM
databricks bundle run medgemma_deploy    --profile FEVM
databricks bundle run medgemma_benchmark --profile FEVM
```

### Path C: Batch inference

Use this when the customer cares about scalable offline processing.

Run `medgemma_register` at least once first: the UDF path loads the registered model, while the SGC path reads the weights saved in the pre-created `model_weights` volume.

The bundle passes the `challenger` alias created by registration to the UDF
job. It also derives the online endpoint from `medgemma_endpoint_name` plus the
`workload_type` suffix. If you override the deployment workload type, use the
same value when running `batch_inference_udf`.

```bash
databricks bundle run prepare_batch_input --profile FEVM
databricks bundle run batch_inference_udf --profile FEVM
databricks bundle run batch_inference_sgc --profile FEVM
```

### Path D: Fine-tuning

Use this when the customer needs terminology adaptation beyond a base domain model.

```bash
databricks bundle run finetune_medgemma --profile FEVM
databricks bundle run finetune_qwen3_disease --profile FEVM
```

The Qwen3 job uses a pinned revision of the public NCBI Disease Corpus on
Hugging Face, trains a QLoRA adapter for disease mention extraction, merges
the weights, and registers a new version of the existing
`pavan_naidu_catalog.mlops.qwen3_vllm_structured` model under the
`disease-finetuned` alias. The base endpoint stays pinned to version 1, while
the job deploys the new version to `qwen3-4b-finetuned-vllm-endpoint`. The app bundle
binds this endpoint with `CAN_QUERY`; deploy and run `clinical_extractor`
after the model endpoint is ready to make it selectable in the UI.

## 5. Capability-to-Code Map

- `src/medgemma`: base domain model + XGrammar path
- `src/vllm`: alternate custom serving path
- `src/batch`: offline inference at scale
- `src/finetune`: PEFT fine-tuning flow
- `src/app`: app UX for side-by-side comparison

## 6. Validation

```bash
cd src/app && python -m pytest tests
cd src/app/frontend && npm run typecheck
cd src/app/frontend && npm run build
```

## Recommended Demo Order

1. Explain the structured output gap.
2. Show the decision framework.
3. Open one workload pack.
4. Run the app or MedGemma flow.
5. Close with batch or fine-tuning depending on customer interest.
