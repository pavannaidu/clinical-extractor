# Source guide

```text
src/
├── app/                         FastAPI + React app and its tests
└── models/
    ├── qwen/                    Primary custom-model demo
    │   ├── deploy_vllm.py        Download, register, and deploy base Qwen3
    │   └── finetune_disease.py   QLoRA disease extraction and deployment
    └── medgemma/                Optional medical-model comparison
        ├── deploy_vllm.py       vLLM serving
        ├── finetune.py          QLoRA training with the XGrammar wrapper
        ├── xgrammar/            Register → deploy → benchmark; serving model.py
        └── batch/               Prepare input → UDF or Serverless GPU inference
```

The app is independently deployable from `app/`. Edit its frontend in
`app/frontend/`, then run `npm --prefix src/app/frontend run build` from the
repository root to regenerate the tracked `app/static/` deployment assets.

Files under `models/` are Databricks source notebooks, except
`medgemma/xgrammar/model.py`, the regular Python module loaded by MLflow.
Keep that module beside `xgrammar/01_register.py`; MedGemma fine-tuning also
loads it from that location.

| Workflow | Bundle job keys, in order |
|---|---|
| Base Qwen3 serving | `inference_vllm_qwen` |
| Fine-tuned Qwen3 | `finetune_qwen3_disease` |
| MedGemma vLLM | `medgemma_vllm` |
| MedGemma XGrammar | `medgemma_register` → `medgemma_deploy` → `medgemma_benchmark` |
| MedGemma fine-tuning | `finetune_medgemma` |
| MedGemma batch | `prepare_batch_input` → `batch_inference_udf` or `batch_inference_sgc` |

Resource definitions remain in `resources/`; job keys are stable across this
reorganization. For the Qwen before-and-after comparison, use the disease
extraction task its fine-tuning notebook trains on. MedGemma offers an additional
medical pretrained model comparison; choose models using reviewed evaluation.

Notebook path changes require a full `databricks bundle deploy` to update existing
job definitions alongside workspace files. A targeted app deployment skips jobs.

See [setup and operations](../docs/setup.md) for prerequisites and commands.
