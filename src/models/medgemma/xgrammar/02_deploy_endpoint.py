# Databricks notebook source
# DBTITLE 1,Overview
# MAGIC %md
# MAGIC # MedGemma XGrammar — Serving Endpoint Deployment & Testing
# MAGIC
# MAGIC Creates or updates a Databricks Model Serving endpoint from a registered MLflow model,
# MAGIC then validates it with a REST API call.
# MAGIC
# MAGIC **Input**: A model registered in Unity Catalog by `01_register.py`.

# COMMAND ----------

# DBTITLE 1,Widgets
dbutils.widgets.removeAll()
dbutils.widgets.text("catalog", "pavan_naidu_catalog", "Catalog")
dbutils.widgets.text("schema", "mlops", "Schema")
dbutils.widgets.text("registered_model_name", "medgemma_xgrammar_structured", "Registered Model Name")
dbutils.widgets.text("model_version", "", "Model Version (leave blank for latest)")
dbutils.widgets.text("endpoint_name", "medgemma-xgrammar-endpoint", "Endpoint Name")
dbutils.widgets.dropdown("workload_size", "Small", ["Small", "Medium", "Large"], "Workload Size")
dbutils.widgets.dropdown("workload_type", "GPU_MEDIUM", ["GPU_SMALL", "GPU_MEDIUM", "GPU_LARGE", "CPU"], "Workload Type")
dbutils.widgets.dropdown("scale_to_zero", "false", ["true", "false"], "Scale to Zero")

# COMMAND ----------

# DBTITLE 1,Configuration
import mlflow

CATALOG = dbutils.widgets.get("catalog")
SCHEMA_NAME = dbutils.widgets.get("schema")
REGISTERED_MODEL_NAME = dbutils.widgets.get("registered_model_name")
ENDPOINT_NAME = dbutils.widgets.get("endpoint_name")
WORKLOAD_SIZE = dbutils.widgets.get("workload_size")
WORKLOAD_TYPE = dbutils.widgets.get("workload_type")
SCALE_TO_ZERO = dbutils.widgets.get("scale_to_zero") == "true"

FULL_MODEL_NAME = f"{CATALOG}.{SCHEMA_NAME}.{REGISTERED_MODEL_NAME}"

# Append GPU type to endpoint name so each workload type gets a distinct endpoint
_GPU_SUFFIX = {
    "GPU_SMALL":  "t4",
    "GPU_MEDIUM": "a10",
    "GPU_LARGE":  "a100",
    "CPU":        "cpu",
}
ENDPOINT_NAME = f"{ENDPOINT_NAME}-{_GPU_SUFFIX.get(WORKLOAD_TYPE, WORKLOAD_TYPE.lower())}"

# Resolve model version: use widget value or pull from task values (when run in a pipeline)
_version_widget = dbutils.widgets.get("model_version").strip()
if _version_widget:
    MODEL_VERSION = _version_widget
else:
    try:
        MODEL_VERSION = dbutils.jobs.taskValues.get(taskKey="register", key="model_version")
    except Exception:
        # Fall back to latest version
        mlflow.set_registry_uri("databricks-uc")
        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(f"name='{FULL_MODEL_NAME}'")
        if not versions:
            raise ValueError(f"No versions found for model '{FULL_MODEL_NAME}'")
        MODEL_VERSION = sorted(versions, key=lambda v: int(v.version))[-1].version

print(f"Model   : {FULL_MODEL_NAME} v{MODEL_VERSION}")
print(f"Endpoint: {ENDPOINT_NAME}")
print(f"Workload: {WORKLOAD_TYPE}/{WORKLOAD_SIZE}, scale_to_zero={SCALE_TO_ZERO}")

# COMMAND ----------

# DBTITLE 1,Deploy / Update Endpoint
import datetime, time, requests as _requests

_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
_host  = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
_headers = {"Authorization": f"Bearer {_token}", "Content-Type": "application/json"}

_served_entity = {
    "entity_name": FULL_MODEL_NAME,
    "entity_version": str(MODEL_VERSION),
    "workload_size": WORKLOAD_SIZE,
    "workload_type": WORKLOAD_TYPE,
    "scale_to_zero_enabled": SCALE_TO_ZERO,
}

_check = _requests.get(f"{_host}/api/2.0/serving-endpoints/{ENDPOINT_NAME}", headers=_headers)

_create_new = _check.status_code != 200

if _check.status_code == 200:
    # Wait for any in-progress update to settle before submitting a new config
    print(f"Endpoint '{ENDPOINT_NAME}' exists — waiting until READY before updating...")
    _deadline_pre = time.time() + 3600
    while time.time() < _deadline_pre:
        _s = _requests.get(f"{_host}/api/2.0/serving-endpoints/{ENDPOINT_NAME}", headers=_headers).json().get("state", {})
        if _s.get("ready") == "READY" and _s.get("config_update") == "NOT_UPDATING":
            break
        if "FAILED" in (_s.get("ready", "") + _s.get("config_update", "")):
            print(f"Endpoint in failed state: {_s} — deleting and recreating...")
            _del = _requests.delete(f"{_host}/api/2.0/serving-endpoints/{ENDPOINT_NAME}", headers=_headers)
            print(f"  delete status: {_del.status_code}")
            time.sleep(10)
            _create_new = True
            break
        print(f"  waiting: ready={_s.get('ready')}, update={_s.get('config_update')}")
        time.sleep(30)

    if not _create_new:
        print("Submitting config update...")
        _resp = _requests.put(
            f"{_host}/api/2.0/serving-endpoints/{ENDPOINT_NAME}/config",
            headers=_headers,
            json={"served_entities": [_served_entity]}
        )
        if not _resp.ok:
            raise Exception(f"Config update failed ({_resp.status_code}): {_resp.text}")
        # Wait for the endpoint state to transition to IN_PROGRESS before polling.
        # Without this sleep the first poll can see the old READY+NOT_UPDATING state
        # (before Databricks registers the update) and exit prematurely.
        print("✓ Config update submitted — waiting 60s for state transition...")
        time.sleep(60)
        print("Polling until ready...")

if _create_new:
    print(f"Creating endpoint '{ENDPOINT_NAME}'...")
    _resp = _requests.post(
        f"{_host}/api/2.0/serving-endpoints",
        headers=_headers,
        json={"name": ENDPOINT_NAME, "config": {"served_entities": [_served_entity]}}
    )
    if not _resp.ok:
        raise Exception(f"Endpoint creation failed ({_resp.status_code}): {_resp.text}")
    # Brief pause before first poll so the endpoint transitions from its initial state
    time.sleep(30)
    print("✓ Endpoint creation submitted — polling until ready...")

# Poll until ready (up to 60 min)
import time
_deadline = time.time() + 3600
while time.time() < _deadline:
    _s = _requests.get(f"{_host}/api/2.0/serving-endpoints/{ENDPOINT_NAME}", headers=_headers).json().get("state", {})
    _ready = _s.get("ready", "")
    _update = _s.get("config_update", "")
    print(f"  state: ready={_ready}, update={_update}")
    if _ready == "READY" and _update == "NOT_UPDATING":
        print("✓ Endpoint is READY")
        break
    if "FAILED" in (_ready + _update):
        # Fetch endpoint JSON then try to get served-model logs while entity still exists
        _ep = _requests.get(f"{_host}/api/2.0/serving-endpoints/{ENDPOINT_NAME}", headers=_headers).json()
        _diag = [f"Endpoint state: {_ep.get('state')}"]
        _all_entities = []
        for _cfg_key in ("config", "pending_config"):
            _cfg = _ep.get(_cfg_key) or {}
            for _k in ("served_models", "served_entities"):
                _all_entities.extend(_cfg.get(_k, []))
        for _sm in _all_entities:
            # Try both "name" (auto-generated) and entity_name-version slug
            _names_to_try = []
            if _sm.get("name"):
                _names_to_try.append(_sm["name"])
            _entity = _sm.get("entity_name", "").split(".")[-1]
            _ver = _sm.get("entity_version", "")
            if _entity and _ver:
                _names_to_try.append(f"{_entity}-{_ver}")
            for _sm_name in dict.fromkeys(_names_to_try):  # dedup, preserve order
                _log_resp = _requests.get(
                    f"{_host}/api/2.0/serving-endpoints/{ENDPOINT_NAME}/served-models/{_sm_name}/logs",
                    headers=_headers,
                )
                _diag.append(f"\n--- logs/{_sm_name} (status={_log_resp.status_code}) ---")
                _diag.append(_log_resp.text[-4000:] if _log_resp.ok else _log_resp.text[:300])
        raise Exception("Endpoint deployment failed\n" + "\n".join(_diag))
    time.sleep(60)
else:
    raise TimeoutError(f"Endpoint {ENDPOINT_NAME} did not become ready within 60 minutes")

print(f"\nEndpoint URL: {_host}/ml/endpoints/{ENDPOINT_NAME}")
dbutils.jobs.taskValues.set(key="endpoint_name", value=ENDPOINT_NAME)

# COMMAND ----------

# DBTITLE 1,Validate Structured Endpoint Inference
import json
from jsonschema import validate

smoke_schema = {
    "type": "object",
    "properties": {"medication": {"type": "string"}, "dosage": {"type": "string"}},
    "required": ["medication", "dosage"],
    "additionalProperties": False,
}
smoke_response = _requests.post(
    f"{_host}/serving-endpoints/{ENDPOINT_NAME}/invocations",
    headers=_headers,
    json={
        "messages": [{"role": "user", "content": "Extract medication and dosage: Amoxicillin 500mg orally three times daily."}],
        "max_tokens": 128,
        "custom_inputs": {"schema": json.dumps(smoke_schema)},
    },
    timeout=180,
)
smoke_response.raise_for_status()
smoke_result = json.loads(smoke_response.json()["choices"][0]["message"]["content"])
validate(instance=smoke_result, schema=smoke_schema)
dbutils.notebook.exit(json.dumps({
    "endpoint_name": ENDPOINT_NAME,
    "model_name": FULL_MODEL_NAME,
    "model_version": str(MODEL_VERSION),
    "structured_smoke_test": smoke_result,
}))
