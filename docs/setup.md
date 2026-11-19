# Setup and operations

## Prerequisites

- Databricks CLI 1.26.0 (used to verify these commands), or a compatible newer
  release, authenticated to the target workspace.
- Python 3.12 and Node.js 20 or 22 for local development.
- An existing Unity Catalog catalog and schema; defaults are `main.mlops`.
- For the app: a Lakebase project, branch, and database, a SQL warehouse for evaluation,
  and the serving endpoints attached in `resources/clinical_extractor.app.yml`.
- For MedGemma downloads and fine-tuning: access to the Hugging Face model and a
  Databricks secret configured by `hf_secret_scope` and `hf_secret_key`. The public
  Qwen model and NCBI Disease dataset do not require that token.

The deployer must be able to create/manage the app and grant its resource bindings.
Evaluation users need `CAN_USE` on the SQL warehouse plus `USE CATALOG`,
`USE SCHEMA`, and `SELECT` on the chosen Unity Catalog data. Creating the example
table below also requires `CREATE TABLE` in that schema. The app needs the
Postgres connection/create permission, AI Extract function execution, and model
query permissions declared in `resources/clinical_extractor.app.yml`.

Check that every serving endpoint retained in that resource file exists in your
workspace. For a managed-only installation, remove unavailable custom endpoint
bindings and their entries in `src/app/models.json`. Keep the environment binding
in `src/app/app.yaml` aligned if you replace its default model. Configure your
workspace's AI Gateway policies for the endpoints you use.

## Start from a fresh clone

```bash
git clone https://github.com/databricks-field-eng/clinical-extractor.git
cd clinical-extractor
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r src/app/requirements-dev.txt
npm --prefix src/app/frontend ci
```

The runtime and development requirements are locked. Use your organization's
configured package mirror if direct package access is restricted.

## Authenticate

The `dev` target resolves its workspace from your CLI profile or authentication
environment. Set your workspace URL and a local profile name, then authenticate:

```bash
export DATABRICKS_HOST="https://<your-workspace-host>"
export DATABRICKS_CONFIG_PROFILE="clinical-extractor"
databricks auth login --host "$DATABRICKS_HOST" --profile "$DATABRICKS_CONFIG_PROFILE"
```

Keep these environment variables set for the commands below. If you already use a
named profile, set `DATABRICKS_CONFIG_PROFILE` to that name; the profile supplies the
host when `DATABRICKS_HOST` is unset. A command-line `--profile` also selects it.

## Configure the bundle

`databricks.yml` contains variables and targets; `resources/*.yml` contains resource
definitions. The bundle name remains `medgemma`, and job and app resource keys remain
stable so existing deployment state continues to address the same resources.

| Variable | Purpose |
|---|---|
| `app_name` | App name; defaults to `clinical-` plus the deployer's normalized identity |
| `catalog`, `schema` | Existing Unity Catalog namespace for models and batch tables |
| `lakebase_project`, `lakebase_branch`, `lakebase_database` | Existing Lakebase resource IDs |
| `evaluation_warehouse_id` | SQL warehouse used for evaluation dataset queries |
| `evaluation_experiment_name` | MLflow experiment created by the bundle for evaluations |
| `batch_experiment_name` | MLflow experiment for Serverless GPU batch runs |
| `medgemma_*`, `qwen3_finetuned_*`, `finetune_registered_model_name` | Registered model and serving endpoint names |
| `gpu_node_type`, `gpu_spark_version` | Classic UDF cluster: two `g5.2xlarge` workers on 17.3 LTS ML GPU |
| `hf_secret_scope`, `hf_secret_key` | Databricks secret holding the Hugging Face token |

`lakebase_project`, `lakebase_branch`, and `evaluation_warehouse_id` are required
workspace-specific values. Save them in the ignored
`.databricks/bundle/dev/variable-overrides.json`, together with any other overrides.
Replace the placeholders in this example before using it:

```json
{
  "app_name": "clinical-extractor",
  "catalog": "<catalog>",
  "schema": "<schema>",
  "lakebase_project": "<project-id>",
  "lakebase_branch": "<branch-id>",
  "lakebase_database": "<database-id>",
  "evaluation_warehouse_id": "<warehouse-id>",
  "hf_secret_scope": "<secret-scope>"
}
```

For CI, supply the same settings through `BUNDLE_VAR_<variable-name>` environment
variables. Keep secret values in Databricks secret scopes; the bundle contains only
the scope and key names. A managed-only app can omit `hf_secret_scope`; it is used
by the optional MedGemma jobs.

Use the current app name when continuing an existing deployment. For a new app,
use a distinct name containing only lowercase
letters, digits, and hyphens, at most 30 characters long. Shorten `app_name` locally if
the identity-based default exceeds that limit. New non-production targets sharing a
workspace should use distinct app names.

## Build and deploy the app

For a new workspace that needs custom models, deploy and run their job definitions
first, then attach the resulting endpoints to the app. For example, after supplying
the required bundle variables:

```bash
databricks bundle plan --select jobs.inference_vllm_qwen --target dev
databricks bundle deploy --select jobs.inference_vllm_qwen --target dev
databricks bundle run inference_vllm_qwen --target dev
```

Use `jobs.finetune_qwen3_disease` for the independent fine-tuning path, or select
the appropriate MedGemma jobs from the model-job table below. Bind each custom
endpoint only after it is ready. A targeted app deployment does not create model
endpoints, and a full deployment with bindings to missing endpoints can fail.

Run these commands from the repository root:

```bash
npm --prefix src/app/frontend ci
npm --prefix src/app/frontend run build
databricks bundle validate --strict --target dev
databricks bundle plan --select apps.clinical_extractor,experiments.clinical_extractor_evaluations --target dev
databricks bundle deploy --select apps.clinical_extractor,experiments.clinical_extractor_evaluations --target dev
databricks bundle run clinical_extractor --target dev
databricks bundle summary --target dev
```

The targeted deployment includes the experiment referenced by the app. Review the
plan before applying it. `bundle deploy` uploads the code and configuration;
`bundle run clinical_extractor` starts a fresh app deployment. To deploy the model-job
definitions as well, run `databricks bundle deploy --target dev`.
Deploying job definitions does not run the jobs.
Use a full bundle plan and deployment after changing notebook paths so existing
jobs reference the uploaded locations; a targeted app deployment skips job updates.

```bash
databricks bundle plan --target dev
databricks bundle deploy --target dev
```

The app's Postgres resource binding injects `PG*` connection settings;
`src/app/app.yaml` supplies `LAKEBASE_ENDPOINT` so the backend can mint database
credentials. The app initializes its tables and seeds built-in presets on startup.
It also binds AI Extract, the evaluation warehouse and experiment, and the serving
endpoints listed in `resources/clinical_extractor.app.yml`.

`src/app/models.json` defines the initial model selection, request contracts, and
default inference options. Attaching an endpoint with `CAN_QUERY` grants access;
selecting it in Model Settings chooses it for use. The app cannot grant permissions
by changing a selection.

Custom GPU endpoints can take several minutes to start after scaling to zero.
The app limits an interactive inference request, including retries, to at most
110 seconds so it can return a structured timeout before the Apps proxy closes
the connection. If a cold endpoint times out, let it finish starting and retry;
the endpoint's control-plane `READY` status does not guarantee a warm replica.

## Local development

Create and activate a virtual environment from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r src/app/requirements-dev.txt
npm --prefix src/app/frontend ci
```

Run the backend and Vite in separate terminals, with the virtual environment active
in the backend terminal:

```bash
cd src/app
uvicorn main:app --reload --port 8000
```

```bash
npm --prefix src/app/frontend run dev
```

Vite proxies `/api` to the backend on port 8000. Backend-only serving uses the
prebuilt `src/app/static/` directory, so rebuild after frontend changes. Edit
`frontend/src/` and `frontend/public/`; the build regenerates `static/` and copies
the public assets into it. Keep the built output in Git for Databricks Apps.

Without Lakebase connection settings, the app uses built-in presets and process-local
model settings. Preset writes and evaluation runs require Lakebase. For a configured
database, the backend accepts Databricks Apps' `PG*` variables plus `LAKEBASE_ENDPOINT`,
or the legacy `LAKEBASE_DB_*` variables including `LAKEBASE_DB_PASSWORD`.
Evaluations also require `DATABRICKS_WAREHOUSE_ID` and use `MLFLOW_EXPERIMENT_ID`
when supplied. In Databricks Apps these values come from the resource bindings.
Dataset queries use the signed-in user's forwarded access token and the app's `sql`
user scope.

## Run a scored evaluation

This one-row synthetic setup check uses `med_001` from the batch notebook. It
checks that discovery, inference, persistence, and metric reporting work together.
It is not an accuracy study. Use an available general-purpose endpoint for this
medication task; the disease-only fine-tuned Qwen model is trained for another
output contract.

1. In Databricks SQL, replace `<catalog>` and `<schema>` with the same values used
   by the bundle and run the statement below. The table is created only if absent.
   Choose a new table name if that example name already contains unrelated data.

```sql
CREATE TABLE IF NOT EXISTS <catalog>.<schema>.clinical_extractor_eval_demo
USING DELTA AS
SELECT
  'med_001' AS record_id,
  'Extract medication info: Prescribe Amoxicillin 500mg orally three times daily for 10 days to treat bacterial sinusitis.' AS input_text,
  '{"medication_name":"Amoxicillin","dosage":"500mg","frequency":"three times daily","route":"orally","indication":"bacterial sinusitis"}' AS ground_truth;
```

2. Create a custom preset in the app named **Medication Smoke Evaluation**. Use
   the input text above as its example and the following JSON schema:

```json
{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "medication_name": {"type": "string"},
    "dosage": {"type": "string"},
    "frequency": {"type": "string"},
    "route": {"type": "string"},
    "indication": {"type": "string"}
  },
  "required": ["medication_name", "dosage", "frequency", "route", "indication"]
}
```

   Use this system prompt: “Extract the five requested medication fields from the
   source. Preserve drug capitalization and dosage formatting. Return one JSON
   object matching the schema and do not add facts.”

3. Open **Evaluations [Preview]**, choose **Create evaluation**, and select the
   table above. Map **Clinical text column** to `input_text`, **JSON ground truth
   column** to `ground_truth`, and **Record ID column** to `record_id`. Set
   **Rows** to `1`, choose an available model, and select the new preset.
4. Choose **Validate dataset**. It should report one valid row and zero invalid
   rows before inference. Choose **Run evaluation** and inspect the completed
   report and its MLflow link.
5. Check the schema-valid percentage, field agreement, fully-correct-record
   percentage, and latency. The reference contains five scalar values. A matching
   prediction scores five of five fields and no extra fields; the score is an
   observed result, not a promised model outcome.

Field comparison canonicalizes object/array ordering and then compares JSON
values. Different wording or dose formatting can therefore count as a mismatch.
Use reviewed labels and explicit normalization rules for a larger held-out
evaluation. The app accepts at most 100 rows per run and does not turn the workload
packs' field-name checklists into ground-truth JSON automatically.

## Model jobs

Registration and Serverless GPU batch inference share model weights under
`/Volumes/<catalog>/<schema>/model_weights/`. Create that volume before running
these jobs; the app itself does not require it:

```sql
CREATE VOLUME IF NOT EXISTS <catalog>.<schema>.model_weights;
CREATE VOLUME IF NOT EXISTS <catalog>.<schema>.ray_data;
```

After deploying job definitions, choose a pipeline:

| Pipeline | Jobs in order |
|---|---|
| MedGemma + XGrammar | `medgemma_register` → `medgemma_deploy` → `medgemma_benchmark` |
| vLLM serving | `medgemma_vllm` or `inference_vllm_qwen` |
| Batch | `prepare_batch_input` → `batch_inference_udf` or `batch_inference_sgc` |
| Fine-tuning | `finetune_medgemma` or `finetune_qwen3_disease` |

Run each job with `databricks bundle run <job-key> --target dev`.
Batch inference requires `medgemma_register` first: the UDF path loads its registered
model under the `challenger` alias, and the SGC path reads its saved weights.
Keep `workload_type` aligned between endpoint deployment, benchmark, and UDF jobs;
the notebooks append the corresponding suffix to `medgemma_endpoint_name`.

`finetune_qwen3_disease` trains on a pinned public NCBI Disease Corpus revision,
registers a version of `qwen3_vllm_structured` under the `disease-finetuned` alias,
and deploys a separate fine-tuned vLLM endpoint. Redeploy and run the app after that
endpoint is ready to use its binding. Model defaults and training parameters live
in `resources/finetune.jobs.yml`.

## Shared files and local artifacts

Share `src/`, `resources/`, the bundle configuration, lockfiles, `docs/`, and
`showcase/`. These contain runnable code, setup instructions, model-workflow
references, and reusable synthetic demo inputs. Keep `src/app/static/` tracked
because the Python app serves that prebuilt frontend directly.

`.gitignore` keeps reports in `output/`, historical local files in `docs/archive/`,
editor settings, browser-test artifacts, export `.pages.json` metadata, virtual
environments, caches, and logs out of new commits. The previous `/info-backup`
presentation is preserved locally at `docs/archive/info-backup/index.html`.
The shared presentation source is `src/app/frontend/public/info/`.

Bundle sync excludes documentation, showcase files, frontend source, and tests
from workspace uploads; it deploys the built app assets and model notebooks.
Rebuild the frontend before deployment.

## Usage tracking

The Python app uses `dbdemos-tracker==0.1.11` with demo ID
`hls-clinical-extractor`. One tracker is initialized per app startup using the
configured workspace. Its manual API lets the middleware filter page views to
`@databricks.com` visitors before sending an event; the library hashes the email.
Only the app and presentation entry pages are counted. Query strings, request
bodies, API calls, static assets, and other email domains are excluded.

Delivery runs in the library's background worker. Initialization and tracking
errors are caught so the app keeps serving requests. Local runs without
`DATABRICKS_HOST` skip tracking.

## Python dependency locks

`src/app/requirements.in` and `requirements-dev.in` contain exact direct dependency
pins. The generated `requirements.txt` and `requirements-dev.txt` also pin every
transitive dependency, with platform markers for Python 3.12 on local and deployed
environments. Databricks Apps installs the runtime `requirements.txt` directly.
Notebook installation cells and model-serving requirements use their own exact
pins because their GPU runtimes differ from the web app.

After intentionally changing a direct pin, regenerate both locks in order with uv:

```bash
uv pip compile src/app/requirements.in --python-version 3.12 --universal --output-file src/app/requirements.txt
uv pip compile src/app/requirements-dev.in --python-version 3.12 --universal --output-file src/app/requirements-dev.txt
```

The development input constrains shared packages to the runtime lock. Install the
development lock into a clean Python 3.12 virtual environment and run the checks
below before sharing dependency updates.

## Validation

With the virtual environment active, run from the repository root:

```bash
python -m pytest tests src/app/tests -q
npm --prefix src/app/frontend run typecheck
npm --prefix src/app/frontend run build
databricks bundle validate --strict --target dev
```

Backend tests use fakes for Databricks and database calls. The model tests cover
training-label alignment, merged checkpoint export, driver model-loading
environment handling, and benchmark latency percentiles without downloading weights.
They also check bounded JSON whitespace and greedy generation at temperature zero.
Endpoint preflight tests preserve remote errors and reject missing or schema-invalid
chat responses before a batch starts.
Bundle validation requires a working workspace login; it checks configuration and
does not execute notebooks.

Validate model workflows by running the jobs above on Databricks. For smoke runs,
use a separate catalog/schema and distinct model and endpoint names. Prepare one
synthetic replication (20 rows), benchmark a few rows, and limit fine-tuning examples
and epochs. The SGC path also needs the `ray_data` volume for Parquet staging.
Check notebook results, schema-valid batch rows, MLflow metrics, and actual endpoint
inference; remove temporary GPU endpoints when done. Small training runs validate
execution and serving, not clinical extraction quality.

The classic GPU job uses dedicated access (`SINGLE_USER`) so Spark can read Unity
Catalog tables. Its SQL path sends chat messages and each row's schema in the
custom endpoint request `STRUCT`; `modelParameters` is for foundation model
endpoints.
The notebook checks one endpoint request before downloading model weights. DBR
16.4 rejected this ChatModel's chat response as a traditional prediction response;
the classic job therefore uses 17.3 LTS ML GPU.
Its local model load temporarily removes `WORLD_SIZE` and restores it afterward:
Transformers 4.53 otherwise treats the runtime's value as a request for tensor
parallelism, which can fail during Gemma3 warm-up. This path performs inference
on the driver's GPU. Import PyTorch before clearing the variable, because DBR
creates it during the first PyTorch import.
The local driver model also disables Transformers' automatic forward-pass
compilation. A 17.3 runtime stack trace caught repeated CUDA graph capture while
the GPU was mostly idle. This setting applies only to the loaded driver instance;
the packaged serving model keeps its own configuration.
If the cluster events report insufficient AWS Spot GPU capacity, retry with
`aws_attributes.availability: ON_DEMAND` and `aws_attributes.zone_id: auto` in the
job cluster configuration. This changes the compute cost, so choose it explicitly
for the run or deployment that needs it.

Keep the model dependency pins together when upgrading. The XGrammar serving
packages pair PyTorch 2.6.0 with torchvision 0.21.0; inheriting a different
torchvision from the serving image can prevent Gemma3 from importing. These
notebooks also pin Databricks SDK 0.96.0 for large model artifact uploads. The
MedGemma vLLM entrypoint disables unbounded JSON whitespace, which otherwise can
exhaust the token budget on nested schemas. Validate both complex preset examples
after changing the vLLM or XGrammar versions.
The ChatModel's schema compiler also disables unbounded whitespace after a batch
vitals response exhausted 512 tokens before closing its JSON object. Temperature
zero uses greedy decoding; positive temperatures use sampling. Run registration's
vitals regression cases and the batch jobs after changing these generation controls.
