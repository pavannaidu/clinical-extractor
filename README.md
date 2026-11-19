# Clinical Extractor — structured AI for life sciences

Clinical Extractor compares AI Extract, Foundation Model APIs, and custom models on the
same clinical note and JSON schema. It helps teams choose a serving approach by reviewing
extracted facts, schema validity, latency, and token usage side by side.

## What the demo shows

- **Interactive extraction:** build a schema visually or edit JSON, choose a clinical
  preset, and compare up to three models.
- **Custom structured output:** serve MedGemma or Qwen through Model Serving with
  XGrammar or vLLM guided decoding.
- **Evaluation:** compare model outputs with ground truth from a Unity Catalog table,
  with results in Lakebase and metrics and traces in MLflow.
- **Load testing:** run short workload comparisons with latency, throughput, retries,
  and schema-validation results.
- **Batch and fine-tuning references:** run Spark or Serverless GPU batch inference,
  or train QLoRA adapters for MedGemma and Qwen.

The [workload packs](showcase/workloads/README.md) include synthetic examples for
clinical notes, adverse events, trial screening, prior authorization, and drug interactions.
The [decision framework](showcase/decision-framework.md) explains when to use each approach.
Schema-valid JSON establishes output shape; clinical accuracy still needs reviewed evaluation.

## Run it

The default target is the FEVM workspace. Follow [setup and operations](docs/setup.md)
to configure Databricks authentication, app resource bindings, and model-job prerequisites.

```bash
npm --prefix src/app/frontend ci
npm --prefix src/app/frontend run build
databricks bundle validate --strict --target dev --profile FEVM
databricks bundle plan --select apps.clinical_extractor,experiments.clinical_extractor_evaluations --target dev --profile FEVM
databricks bundle deploy --select apps.clinical_extractor,experiments.clinical_extractor_evaluations --target dev --profile FEVM
databricks bundle run clinical_extractor --target dev --profile FEVM
```

## Repository guide

| Location | Purpose |
|---|---|
| `databricks.yml` | Bundle identity, variables, and deployment targets |
| `resources/` | App, experiment, and model-job definitions |
| `src/app/` | FastAPI backend, React frontend, and backend tests |
| `src/models/qwen/` | Primary custom-model demo: base serving and disease-extraction fine-tuning |
| `src/models/medgemma/` | Medical-model comparison: vLLM, XGrammar, fine-tuning, and batch notebooks |
| `showcase/` | Demo story, workload packs, and walkthroughs |
| `docs/` | Setup, architecture, demo runbook, and benchmark references |

See the [source guide](src/README.md) and [documentation index](docs/README.md).

Generated reports in `output/`, local history in `docs/archive/`, caches, and
editor settings are ignored. Setup docs, architecture, showcase examples, and
the prebuilt app assets are shared so a fresh clone can run the demo.
