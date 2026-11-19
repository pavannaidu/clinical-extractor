# Repository guidance

Databricks Asset Bundle for structured clinical extraction: MedGemma + XGrammar,
MedGemma/Qwen vLLM serving, batch inference, QLoRA fine-tuning, and a FastAPI/React app.

## Configuration and commands

- Bundle name: `medgemma`; default target: `dev` in development mode.
- Select the workspace through `--profile` or `DATABRICKS_CONFIG_PROFILE` /
  `DATABRICKS_HOST`; shared configuration contains no workspace URL.
- `databricks.yml` owns variables and targets. `resources/*.yml` owns resource
  definitions, with paths relative to those files (`../src/...`).
- Keep bundle name and resource keys stable to preserve deployment state.
- App names use `app_name`; existing checkouts can preserve `clinical-extractor`
  through ignored `.databricks/bundle/dev/variable-overrides.json`.
- Lakebase project/branch and evaluation warehouse IDs must be supplied through
  ignored local bundle overrides or `BUNDLE_VAR_*`. Existing local deployment
  values stay in `.databricks/`. Setup details: [docs/setup.md](docs/setup.md).

```bash
databricks bundle validate --strict --target dev
databricks bundle deploy --target dev
databricks bundle run <job-key> --target dev

# App and its evaluation experiment: review, deploy, then start.
databricks bundle plan --select apps.clinical_extractor,experiments.clinical_extractor_evaluations --target dev
databricks bundle deploy --select apps.clinical_extractor,experiments.clinical_extractor_evaluations --target dev
databricks bundle run clinical_extractor --target dev
```

Job keys by resource file:

- `medgemma.jobs.yml`: `medgemma_register`, `medgemma_deploy`, `medgemma_benchmark`
- `batch.jobs.yml`: `prepare_batch_input`, `batch_inference_udf`, `batch_inference_sgc`
- `vllm.jobs.yml`: `medgemma_vllm`, `inference_vllm_qwen`
- `finetune.jobs.yml`: `finetune_medgemma`, `finetune_qwen3_disease`

## App development

With the Python virtual environment active, run from the repository root:

```bash
python -m pip install -r src/app/requirements-dev.txt
python -m pytest src/app/tests -q
npm --prefix src/app/frontend ci
npm --prefix src/app/frontend run dev
npm --prefix src/app/frontend run typecheck
npm --prefix src/app/frontend run build
# In a separate terminal:
cd src/app && uvicorn main:app --reload --port 8000
```

- Edit `src/app/frontend/src/` and `frontend/public/`; rebuild tracked `static/`
  before deploying. Vite proxies `/api` to port 8000 during local development.
- `main.py` mounts the SPA and initializes Lakebase. The app binds Postgres,
  AI Extract, the evaluation warehouse and experiment, and serving endpoints in
  `resources/clinical_extractor.app.yml`; runtime environment lives in `app.yaml`.
- `api/routes.py` orchestrates discovery, settings, presets, inference, and load
  testing. Pure AI Extract transformations live in `api/ai_extract.py`; shared
  JSON/schema transformations live in `api/json_output.py`.
- `api/evaluations.py` runs ground-truth evaluations with Lakebase persistence
  and MLflow reporting. Dataset queries use the forwarded user token and SQL scope.
- `api/database.py` reads injected `PG*` settings plus `LAKEBASE_ENDPOINT` and
  refreshes OAuth credentials. Legacy `LAKEBASE_DB_*` values remain supported.
  Without Lakebase the app uses built-in presets and in-memory model settings;
  preset writes and evaluation runs require the database.
- `models.json` seeds model selection, explicit request contracts, and inference
  defaults. Selecting a model does not grant access: attach endpoints with
  `CAN_QUERY` in the app resource definition.
- Supported serving contracts: `custom_inputs_schema` (`custom_inputs.schema`),
  `ppt_json_schema` (`response_format.json_schema`), and `openai_guided_json`
  (`guided_json` + `guided_decoding_backend`). See [architecture](docs/architecture.md).
- Test path setup belongs in `src/app/tests/conftest.py`.

## Model workflow constraints

- Model workflows are grouped by family in `src/models/qwen/` and
  `src/models/medgemma/`; see [src/README.md](src/README.md) for job mappings.
  Their Python files are Databricks source notebooks with `# COMMAND ----------` separators and
  `# MAGIC %md` markdown cells, except `src/models/medgemma/xgrammar/model.py`.
- `model.py` is the MLflow artifact loaded at serving time. Keep it co-located
  with `01_register.py` and use code-based `python_model` logging; bitsandbytes
  model objects cannot be serialized by MLflow.
- `src/models/medgemma/finetune.py` reuses `xgrammar/model.py` relative to its
  notebook directory. Update both logging paths if that module ever moves.
- The model selects bf16 on SM >= 8.0 (A10/A100) and NF4 on SM 7.5 (T4).
  `load_context` builds an XGrammar compiler; inference compiles and caches
  `params.custom_inputs["schema"]`. An omitted schema leaves generation unconstrained.
- Keep JSON grammar whitespace bounded so it cannot exhaust the token budget.
  Temperature zero uses greedy decoding; positive temperatures use sampling.
  Registration includes the vitals prompt that previously stalled on whitespace.
- Keep notebook widget names aligned with job `base_parameters` in `resources/`.
  MedGemma endpoint names append a workload suffix (`-a10` by default); keep the
  workload type aligned across deployment, benchmark, and UDF jobs.
- Create `<catalog>.<schema>.model_weights` before registration or SGC batch
  inference. The bundle does not create that Unity Catalog volume.
- Batch notebooks set the `databricks-uc` registry URI; avoid duplicate resets.
  Batch and benchmark notebooks use `batch_experiment_name`; interactive runs
  default to the current user's experiment. App evaluations use their bound experiment.
- MedGemma downloads use `hf_secret_scope` / `hf_secret_key` Databricks secrets.
- The UDF job uses two `g5.2xlarge` workers by default; Serverless GPU jobs
  request `GPU_1xA10` directly.
- The classic job uses 17.3 LTS ML GPU and checks one SQL endpoint request before
  loading model weights. DBR 16.4's `ai_query` rejected this ChatModel's chat response.
  Its local model instance disables automatic forward-pass compilation after
  PyTorch 2.7 repeatedly captured CUDA graphs for changing prompt lengths.

## Shared and local files

- Keep `docs/`, `showcase/`, lockfiles, and `src/app/static/` in Git.
- `output/`, `docs/archive/`, export `.pages.json` metadata, editor settings,
  caches, and virtual environments are local and ignored.
- Bundle sync excludes documentation, showcase material, frontend source, and
  tests. Build the frontend locally before deployment; the app serves `static/`.
