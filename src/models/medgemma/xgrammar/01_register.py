# Databricks notebook source
# DBTITLE 1,Overview
# MAGIC %md
# MAGIC # MedGemma XGrammar — Download, Test & Register
# MAGIC
# MAGIC Downloads MedGemma from Hugging Face, logs the XGrammar-constrained
# MAGIC `model.py` as a code-based MLflow model, registers it to Unity Catalog,
# MAGIC and validates inference from the registry.
# MAGIC
# MAGIC Runs on **Serverless GPU Compute** (SGC) — no pre-provisioned cluster needed.
# MAGIC
# MAGIC **Output**: A registered model version in `{catalog}.{schema}.{registered_model_name}`
# MAGIC that can be deployed by `02_deploy_endpoint.py`.

# COMMAND ----------

# DBTITLE 1,Widgets
dbutils.widgets.removeAll()
dbutils.widgets.text("catalog", "pavan_naidu_catalog", "Catalog")
dbutils.widgets.text("schema", "mlops", "Schema")
dbutils.widgets.text("registered_model_name", "medgemma_xgrammar_structured", "Registered Model Name")
dbutils.widgets.text("hf_model_name", "google/medgemma-1.5-4b-it", "HuggingFace Model Name")

# COMMAND ----------

# DBTITLE 1,Install Dependencies
# MAGIC %pip install -U \
# MAGIC   "transformers==4.53.3" \
# MAGIC   "accelerate==1.4.0" \
# MAGIC   "bitsandbytes==0.46.1" \
# MAGIC   "xgrammar==0.1.31" \
# MAGIC   "huggingface_hub" \
# MAGIC   "hf_transfer" \
# MAGIC   "mlflow==3.12.0" \
# MAGIC   "databricks-sdk==0.96.0"

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# DBTITLE 1,GPU Runtime Workaround
import os

runtime_version = os.environ.get("DATABRICKS_RUNTIME_VERSION", "")
if "AIR_RUNTIME_VERSION" in os.environ and "-gpu" not in runtime_version:
    runtime_version = runtime_version or os.environ["AIR_RUNTIME_VERSION"]
    os.environ["DATABRICKS_RUNTIME_VERSION"] = f"{runtime_version}-gpu"

print(os.environ.get("DATABRICKS_RUNTIME_VERSION", "(not set)"))

# COMMAND ----------

# DBTITLE 1,Configuration
CATALOG = dbutils.widgets.get("catalog")
SCHEMA_NAME = dbutils.widgets.get("schema")
REGISTERED_MODEL_NAME = dbutils.widgets.get("registered_model_name")
HF_MODEL_NAME = dbutils.widgets.get("hf_model_name")

FULL_MODEL_NAME = f"{CATALOG}.{SCHEMA_NAME}.{REGISTERED_MODEL_NAME}"

import tempfile
pwd = tempfile.mkdtemp(prefix="medgemma_register_")
os.chdir(pwd)

LOCAL_MODEL_PATH = "medgemma_weights"

print(f"HF model    : {HF_MODEL_NAME}")
print(f"Register as : {FULL_MODEL_NAME}")
print(f"Working dir : {pwd}")

# COMMAND ----------

# DBTITLE 1,Download Model from HuggingFace
import pathlib
from huggingface_hub import snapshot_download

hf_token = dbutils.secrets.get(scope="pavan_naidu", key="hf_token")

snapshot_download(
    repo_id=HF_MODEL_NAME,
    local_dir=LOCAL_MODEL_PATH,
    token=hf_token,
)

assert (pathlib.Path(LOCAL_MODEL_PATH) / "config.json").exists(), f"config.json not found in {LOCAL_MODEL_PATH}"
print(f"✓ Model downloaded to {os.path.join(os.getcwd(), LOCAL_MODEL_PATH)}")

# COMMAND ----------

# DBTITLE 1,Save Weights to UC Volume
import shutil

VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA_NAME}/model_weights/{HF_MODEL_NAME}/bf16"

if os.path.exists(VOLUME_PATH) and (pathlib.Path(VOLUME_PATH) / "config.json").exists():
    print(f"✓ Volume weights already exist at {VOLUME_PATH}")
else:
    dbutils.fs.mkdirs(VOLUME_PATH)
    shutil.copytree(LOCAL_MODEL_PATH, VOLUME_PATH, dirs_exist_ok=True)
    print(f"✓ Weights saved to {VOLUME_PATH}")

# COMMAND ----------

# DBTITLE 1,Log & Register Model with MLflow
import mlflow

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(f"/Users/{dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()}/medgemma-xgrammar")

# Derive the path to model.py (co-located in the same workspace directory)
_notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
model_code_path = f"/Workspace{_notebook_path.rsplit('/', 1)[0]}/model.py"
print(f"Model code path: {model_code_path}")

with mlflow.start_run(run_name="medgemma-xgrammar-chatmodel") as run:
    model_info = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=model_code_path,
        artifacts={"model-weights": LOCAL_MODEL_PATH},
        metadata={"task": "llm/v1/chat"},
        pip_requirements=[
            "torch==2.6.0",
            "torchvision==0.21.0",
            "transformers==4.53.3",
            "accelerate==1.4.0",
            "xgrammar==0.1.31",
        ],
        infer_code_paths=False,
    )
    print(f"✓ Model logged (run_id: {run.info.run_id})")

model_version = mlflow.register_model(
    model_uri=f"runs:/{run.info.run_id}/model",
    name=FULL_MODEL_NAME,
    env_pack="databricks_model_serving",
)

client = mlflow.MlflowClient()
client.set_registered_model_alias(
    name=FULL_MODEL_NAME,
    alias="challenger",
    version=model_version.version
)

print(f"\n✓ Model registered: {FULL_MODEL_NAME} version {model_version.version}")
print(f"✓ Alias 'challenger' set on version {model_version.version}")
print(f"\nPass these to 02_deploy_endpoint.py:")
print(f"  registered_model_name = {REGISTERED_MODEL_NAME}")
print(f"  model_version         = {model_version.version}")

# Store for notebook chaining (no-op in interactive runs)
try:
    dbutils.jobs.taskValues.set(key="model_name", value=FULL_MODEL_NAME)
    dbutils.jobs.taskValues.set(key="model_version", value=model_version.version)
except Exception:
    pass

# COMMAND ----------

# DBTITLE 1,Chat Inference Test (from Registry)
import json
from jsonschema import validate

loaded_model = mlflow.pyfunc.load_model(f"models:/{FULL_MODEL_NAME}@challenger")

medication_schema = {
    "type": "object",
    "properties": {
        "medication_name": {"type": "string"},
        "dosage": {"type": "string"},
        "frequency": {"type": "string"},
        "route": {"type": "string"},
        "indication": {"type": "string"}
    },
    "required": ["medication_name", "dosage", "frequency"]
}

prompts = [
    "Extract medication info: Prescribe Amoxicillin 500mg orally three times daily for bacterial infection.",
    "Extract medication info: Patient taking Lisinopril 10mg once daily by mouth for hypertension.",
]

test_cases = [(prompt, medication_schema, 0.7) for prompt in prompts]
vitals_schema = {
    "type": "object",
    "properties": {
        "blood_pressure": {"type": "string"},
        "heart_rate": {"type": "integer"},
        "temperature_f": {"type": "number"},
        "respiratory_rate": {"type": "integer"},
        "spo2_percent": {"type": "number"},
    },
    "required": ["blood_pressure", "heart_rate", "temperature_f"],
}
# This prompt previously exhausted its token budget on JSON whitespace.
for temperature in [0.0, 0.7]:
    test_cases.append((
        "Extract patient vitals: Blood pressure 118/74, heart rate 72, temperature "
        "98.6 Fahrenheit, respiratory rate 14, oxygen saturation 99 percent on room air.",
        vitals_schema,
        temperature,
    ))

for i, (prompt, requested_schema, temperature) in enumerate(test_cases, 1):
    result = loaded_model.predict(
        {
            "messages": [{"role": "user", "content": prompt}],
            "custom_inputs": {"schema": json.dumps(requested_schema)},
            "temperature": temperature,
        }
    )
    print(f"Result {i}:")
    content = result["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    validate(instance=parsed, schema=requested_schema)
    print(json.dumps(parsed, indent=2))
    print(f"Usage: {result.get('usage', {})}")
    print()

print("✓ Chat inference test passed")
dbutils.notebook.exit(json.dumps({
    "model_name": FULL_MODEL_NAME,
    "model_version": str(model_version.version),
    "schema_valid_predictions": len(test_cases),
}))
