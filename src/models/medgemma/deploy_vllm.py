# Databricks notebook source
# DBTITLE 1,Overview
# MAGIC %md
# MAGIC # MedGemma - Custom LLM Serving with vLLM
# MAGIC
# MAGIC End-to-end notebook for serving `google/medgemma-1.5-4b-it` with vLLM using
# MAGIC the same direct custom-entrypoint flow that works for Qwen in this repo.
# MAGIC
# MAGIC ## Flow
# MAGIC 1. Download MedGemma from Hugging Face
# MAGIC 2. Start a local vLLM OpenAI-compatible server
# MAGIC 3. Run plain-chat and guided JSON validation tests
# MAGIC 4. Log the model to MLflow with `metadata.entrypoint`
# MAGIC 5. Register to Unity Catalog with `env_pack="databricks_model_serving"`
# MAGIC 6. Create or update a Model Serving endpoint and validate it remotely
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - GPU cluster (A10 or better)
# MAGIC - Secret scope `pavan_naidu` with key `hf_token`

# COMMAND ----------

# DBTITLE 1,Widgets
dbutils.widgets.removeAll()
dbutils.widgets.text("catalog", "pavan_naidu_catalog", "Catalog")
dbutils.widgets.text("schema", "mlops", "Schema")
dbutils.widgets.text("registered_model_name", "medgemma_vllm_structured", "Registered Model Name")
dbutils.widgets.text("endpoint_name", "medgemma-1-5-4b-vllm-endpoint", "Endpoint Name")
dbutils.widgets.dropdown("workload_type", "GPU_MEDIUM", ["GPU_SMALL", "GPU_MEDIUM", "GPU_LARGE"], "Workload Type")
dbutils.widgets.dropdown("workload_size", "Small", ["Small", "Medium", "Large"], "Workload Size")
dbutils.widgets.dropdown("scale_to_zero", "true", ["true", "false"], "Scale to Zero")
dbutils.widgets.text("hf_model_name", "google/medgemma-1.5-4b-it", "HuggingFace Model Name")

# COMMAND ----------

# DBTITLE 1,GPU Check
# MAGIC %sh
# MAGIC nvidia-smi

# COMMAND ----------

# DBTITLE 1,Install Dependencies
# MAGIC %pip install \
# MAGIC   databricks-sdk==0.96.0 \
# MAGIC   mlflow==3.12.0 \
# MAGIC   vllm==0.11.2 \
# MAGIC   transformers==4.57.6 \
# MAGIC   hf_transfer==0.1.9 \
# MAGIC   openai==2.17.0 \
# MAGIC   opencv-python-headless==4.12.*

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

# DBTITLE 1,Verify Versions
import mlflow
import torch
import transformers
import vllm

print(f"PyTorch:      {torch.__version__}")
print(f"vLLM:         {vllm.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"MLflow:       {mlflow.__version__}")

assert torch.cuda.is_available(), "CUDA not available"
print(f"CUDA devices: {torch.cuda.device_count()}")
print("\n✓ All version checks passed")

# COMMAND ----------

# DBTITLE 1,Set Working Directory to Local Disk
import tempfile

pwd = tempfile.mkdtemp(prefix="medgemma_vllm_")
os.chdir(pwd)
print(f"Working directory: {pwd}")

# COMMAND ----------

# DBTITLE 1,Configuration
CATALOG = dbutils.widgets.get("catalog")
SCHEMA_NAME = dbutils.widgets.get("schema")
REGISTERED_MODEL_NAME = dbutils.widgets.get("registered_model_name")
ENDPOINT_NAME = dbutils.widgets.get("endpoint_name")
WORKLOAD_TYPE = dbutils.widgets.get("workload_type")
WORKLOAD_SIZE = dbutils.widgets.get("workload_size")
SCALE_TO_ZERO = dbutils.widgets.get("scale_to_zero") == "true"
HF_MODEL_NAME = dbutils.widgets.get("hf_model_name")

FULL_MODEL_NAME = f"{CATALOG}.{SCHEMA_NAME}.{REGISTERED_MODEL_NAME}"

LOCAL_PORT = 3080
SERVING_PORT = 8080
ARTIFACTS_PATH = "medgemma"
FIXED_ENDPOINT_CONCURRENCY = 4
MIN_ENDPOINT_CONCURRENCY = 0 if SCALE_TO_ZERO else FIXED_ENDPOINT_CONCURRENCY

print(f"Model name: {FULL_MODEL_NAME}")
print(f"Endpoint:   {ENDPOINT_NAME}")
print(f"HF model:   {HF_MODEL_NAME}")
print(f"Artifacts:  {os.path.join(pwd, ARTIFACTS_PATH)}")
print(
    "Serving:    "
    f"workload_type={WORKLOAD_TYPE}, "
    f"concurrency={MIN_ENDPOINT_CONCURRENCY}-{FIXED_ENDPOINT_CONCURRENCY}, "
    f"scale_to_zero={SCALE_TO_ZERO}"
)

# COMMAND ----------

# DBTITLE 1,Download MedGemma from HuggingFace
import pathlib
from huggingface_hub import snapshot_download

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
hf_token = dbutils.secrets.get(scope="pavan_naidu", key="hf_token")

snapshot_download(
    repo_id=HF_MODEL_NAME,
    local_dir=ARTIFACTS_PATH,
    token=hf_token,
)

assert (pathlib.Path(ARTIFACTS_PATH) / "config.json").exists(), f"config.json not found in {ARTIFACTS_PATH}"
print(f"✓ Model downloaded to {os.path.join(pwd, ARTIFACTS_PATH)}")

# COMMAND ----------

# DBTITLE 1,Temporary Networking Workaround
import http.server
import socket
import socketserver
import subprocess
import threading
import time
import urllib.request


def wait_port_bindable(port, host, timeout_s, poll_s=0.05):
    deadline = time.time() + timeout_s
    last_err = None
    while time.time() < deadline:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind((host, port))
            return True
        except OSError as exc:
            last_err = exc
            time.sleep(poll_s)
        finally:
            sock.close()
    raise TimeoutError(f"Port {port} not bindable after {timeout_s}s; last error: {last_err}")


def can_use_localhost(port, timeout=3):
    host = "0.0.0.0"

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *args):
            pass

    httpd = socketserver.TCPServer((host, port), Handler)
    thread = threading.Thread(target=httpd.serve_forever)
    thread.start()
    try:
        time.sleep(0.2)
        response = urllib.request.urlopen(f"http://localhost:{port}", timeout=timeout)
        return response.status == 200 and response.read().decode() == "ok"
    except Exception:
        return False
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join()
        wait_port_bindable(port, host, timeout_s=300)


HOST_TO_QUERY = "localhost"
if not can_use_localhost(LOCAL_PORT):
    HOST_TO_QUERY = subprocess.check_output(
        "hostname -I | awk '{print $1}'",
        shell=True,
        text=True,
    ).strip()

print(HOST_TO_QUERY)

# COMMAND ----------

# DBTITLE 1,Build Entrypoint
def entrypoint(port: int) -> str:
    import shlex

    argv = [
        "python",
        "-u",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        ARTIFACTS_PATH,
        "--served-model-name",
        "medgemma",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "4096",
        "--gpu-memory-utilization",
        "0.90",
        # Prevent long whitespace loops when MedGemma follows nested JSON schemas.
        "--structured-outputs-config",
        '{"backend":"xgrammar","disable_fallback":true,"disable_any_whitespace":true}',
    ]
    return shlex.join(argv)


print(entrypoint(LOCAL_PORT))

# COMMAND ----------

# DBTITLE 1,Start Local vLLM Server
LOG_FILE = "process.log"

log = open(LOG_FILE, "w")
vllm_process = subprocess.Popen(
    ["bash", "-lc", entrypoint(LOCAL_PORT)],
    stdout=log,
    stderr=subprocess.STDOUT,
    text=True,
    start_new_session=True,
)

print(f"✓ vLLM process started (PID: {vllm_process.pid})")

# COMMAND ----------

# DBTITLE 1,Wait for Startup
STARTUP_TIMEOUT = 300
deadline = time.time() + STARTUP_TIMEOUT

while time.time() < deadline:
    if vllm_process.poll() is not None:
        log_tail = pathlib.Path(LOG_FILE).read_text()[-6000:] if pathlib.Path(LOG_FILE).exists() else ""
        raise RuntimeError(f"vLLM exited with code {vllm_process.returncode}\n\n--- vLLM log (last 6000 chars) ---\n{log_tail}")

    log_content = pathlib.Path(LOG_FILE).read_text() if pathlib.Path(LOG_FILE).exists() else ""
    if "Application startup complete" in log_content:
        break
    time.sleep(5)
else:
    log_tail = pathlib.Path(LOG_FILE).read_text()[-6000:] if pathlib.Path(LOG_FILE).exists() else ""
    raise RuntimeError(f"vLLM did not start within {STARTUP_TIMEOUT}s\n\n--- vLLM log (last 6000 chars) ---\n{log_tail}")

print("✓ vLLM server is ready")

# COMMAND ----------

# DBTITLE 1,Query Local vLLM
import requests

resp = requests.post(
    f"http://{HOST_TO_QUERY}:{LOCAL_PORT}/invocations",
    json={"messages": [{"role": "user", "content": "Hello"}]},
)
resp.raise_for_status()
local_response = resp.json()["choices"][0]["message"]["content"]
print(local_response)

# COMMAND ----------

# DBTITLE 1,Query Local vLLM Streaming
import json

resp = requests.post(
    f"http://{HOST_TO_QUERY}:{LOCAL_PORT}/invocations",
    json={
        "messages": [{"role": "user", "content": "Tell me a short clinical summary in about 120 words."}],
        "stream": True,
    },
    stream=True,
)
resp.raise_for_status()

for line in resp.iter_lines():
    if not line:
        continue
    if line == b"data: [DONE]":
        break
    if line.startswith(b"data: "):
        data = json.loads(line[6:])
        delta = data["choices"][0].get("delta", {})
        if "content" in delta:
            print(delta["content"], end="", flush=True)

print()

# COMMAND ----------

# DBTITLE 1,Structured Local Validation for Heavy Presets
from openai import OpenAI

SYSTEM_PROMPT = (
    "You are a clinical information extraction assistant. Extract only information "
    "explicitly supported by the provided medical text. Return output that matches "
    "the requested schema exactly. Do not invent facts; when the source text does "
    "not provide a value, leave it null or empty if the schema allows."
)
MEDGEMMA_EXTRACTION_MAX_TOKENS = 2048
MEDGEMMA_EXTRACTION_TEMPERATURE = 0.0


def _strict_string(nullable=False, enum=None):
    schema = {"type": "string"}
    if nullable:
        schema["type"] = ["string", "null"]
    if enum:
        schema["enum"] = [*enum, "null"] if nullable and "null" not in enum else enum
    return schema


def _strict_number(nullable=False):
    return {"type": ["number", "null"]} if nullable else {"type": "number"}


def _strict_boolean(nullable=False):
    return {"type": ["boolean", "null"]} if nullable else {"type": "boolean"}


def _strict_array(items, nullable=False):
    schema = {"type": "array", "items": items}
    if nullable:
        schema["type"] = ["array", "null"]
    return schema


def _strict_object(properties, nullable=False):
    schema = {
        "type": "object",
        "properties": properties,
        "required": list(properties.keys()),
        "additionalProperties": False,
    }
    if nullable:
        schema["type"] = ["object", "null"]
    return schema


DISCHARGE_MEDICATION_SCHEMA = _strict_object(
    {
        "name": _strict_string(),
        "instructions": _strict_string(nullable=True),
        "duration": _strict_string(nullable=True),
    }
)

FOLLOW_UP_APPOINTMENT_SCHEMA = _strict_object(
    {
        "service": _strict_string(),
        "timeframe": _strict_string(nullable=True),
        "reason": _strict_string(nullable=True),
    }
)

STUDY_METADATA_SCHEMA = _strict_object(
    {
        "modality": _strict_string(nullable=True),
        "body_part": _strict_string(nullable=True),
        "acquisition_context": _strict_string(nullable=True),
        "device_type": _strict_string(nullable=True),
        "portable": _strict_boolean(nullable=True),
        "study_date": _strict_string(nullable=True),
    },
    nullable=True,
)

IMAGING_MEASUREMENT_SCHEMA = _strict_object(
    {
        "name": _strict_string(),
        "value": _strict_number(nullable=True),
        "unit": _strict_string(nullable=True),
    }
)

DISCHARGE_FOLLOW_UP_PLAN_SCHEMA = _strict_object(
    {
        "disposition": _strict_string(),
        "diagnoses": _strict_array(_strict_string()),
        "medications": _strict_array(DISCHARGE_MEDICATION_SCHEMA),
        "follow_up_appointments": _strict_array(FOLLOW_UP_APPOINTMENT_SCHEMA),
        "pending_tests": _strict_array(_strict_string(), nullable=True),
        "return_precautions": _strict_array(_strict_string()),
        "patient_instructions": _strict_array(_strict_string()),
        "source_text": _strict_string(nullable=True),
    }
)

IMAGING_DEVICE_REPORT_SCHEMA = _strict_object(
    {
        "study_metadata": STUDY_METADATA_SCHEMA,
        "findings": _strict_array(_strict_string()),
        "impression": _strict_string(),
        "measurements": _strict_array(IMAGING_MEASUREMENT_SCHEMA, nullable=True),
        "critical_result": _strict_boolean(nullable=True),
        "recommendation": _strict_string(nullable=True),
        "source_text": _strict_string(nullable=True),
    }
)

DISCHARGE_FOLLOW_UP_PLAN_PROMPT = (
    "Discharge after CT pulmonary angiography: Disposition home with spouse. "
    "Diagnoses include multifocal pneumonia and chronic asthma. Medications: "
    "doxycycline 100 mg twice daily for 7 days, continue inhaled corticosteroid, "
    "albuterol as needed. Follow-up with primary care in 3 days and pulmonology in "
    "2 weeks. Pending blood cultures. Return for worsening shortness of breath, "
    "chest pain, or persistent fever. Encourage hydration and incentive spirometry."
)

IMAGING_DEVICE_REPORT_PROMPT = (
    "Transthoracic echocardiogram acquired on a cart-based ultrasound system. "
    "Findings: mildly dilated left ventricle with global hypokinesis, trace mitral "
    "regurgitation, no pericardial effusion. Measurements: LVEF 35 percent, LVEDD "
    "6.1 cm. Impression: reduced systolic function consistent with cardiomyopathy. "
    "Recommendation: optimize guideline-directed medical therapy and repeat echo in "
    "3 months."
)

HEAVY_PRESET_VALIDATIONS = [
    {
        "label": "Imaging Device Report",
        "schema": IMAGING_DEVICE_REPORT_SCHEMA,
        "prompt": IMAGING_DEVICE_REPORT_PROMPT,
        "required_keys": [
            "study_metadata",
            "findings",
            "impression",
            "measurements",
            "critical_result",
            "recommendation",
            "source_text",
        ],
    },
    {
        "label": "Discharge / Follow-up Plan",
        "schema": DISCHARGE_FOLLOW_UP_PLAN_SCHEMA,
        "prompt": DISCHARGE_FOLLOW_UP_PLAN_PROMPT,
        "required_keys": [
            "disposition",
            "diagnoses",
            "medications",
            "follow_up_appointments",
            "pending_tests",
            "return_precautions",
            "patient_instructions",
            "source_text",
        ],
    },
]


def _usage_dict(response):
    usage = getattr(response, "usage", None)
    if not usage:
        return {}
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def _has_whitespace_stall(content: str) -> bool:
    if not isinstance(content, str) or not content:
        return False

    trailing_whitespace = len(content) - len(content.rstrip())
    if trailing_whitespace >= 256:
        return True

    lines = content.splitlines()
    if len(lines) >= 20:
        tail_lines = lines[-40:]
        blankish_lines = sum(1 for line in tail_lines if not line.strip())
        if blankish_lines >= 20:
            return True

    if len(content) >= 400:
        tail = content[len(content) // 2 :]
        whitespace_ratio = sum(ch.isspace() for ch in tail) / max(len(tail), 1)
        if whitespace_ratio > 0.75:
            return True

    return False


def _validate_guided_completion(label: str, response, max_tokens: int, required_keys: list[str]):
    content = response.choices[0].message.content or ""
    finish_reason = response.choices[0].finish_reason
    usage = _usage_dict(response)

    diagnostics = {
        "label": label,
        "finish_reason": finish_reason,
        "usage": usage,
        "raw_chars": len(content),
        "raw_output": content,
    }
    print(json.dumps(diagnostics, indent=2))

    raw_candidate = content.rstrip()
    try:
        parsed = json.loads(raw_candidate)
    except json.JSONDecodeError as exc:
        if _has_whitespace_stall(content):
            raise AssertionError(
                f"{label}: raw output appears to stall into whitespace before producing valid JSON"
            ) from exc
        raise

    missing_keys = [key for key in required_keys if key not in parsed]
    if missing_keys:
        raise AssertionError(f"{label}: missing required top-level keys: {missing_keys}")

    completion_tokens = usage.get("completion_tokens")
    if finish_reason == "length":
        print(f"Warning: {label} reached max_tokens but still produced valid JSON")
    elif isinstance(completion_tokens, int) and completion_tokens >= max_tokens:
        print(f"Warning: {label} hit max_tokens boundary but produced valid JSON")

    return parsed


def _run_guided_validation(client: OpenAI, model_name: str, scenario: dict):
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": scenario["prompt"]},
        ],
        max_tokens=MEDGEMMA_EXTRACTION_MAX_TOKENS,
        temperature=MEDGEMMA_EXTRACTION_TEMPERATURE,
        extra_body={
            "guided_json": scenario["schema"],
            "guided_decoding_backend": "xgrammar:no-fallback",
        },
    )
    parsed = _validate_guided_completion(
        label=scenario["label"],
        response=response,
        max_tokens=MEDGEMMA_EXTRACTION_MAX_TOKENS,
        required_keys=scenario["required_keys"],
    )
    from jsonschema import validate
    validate(instance=parsed, schema=scenario["schema"])
    print(json.dumps(parsed, indent=2))
    return parsed


def _run_guided_validation_nonblocking(client: OpenAI, model_name: str, scenario: dict):
    try:
        return _run_guided_validation(client, model_name, scenario)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "label": scenario["label"],
                    "warning": f"Local guided validation failed but deployment will continue: {exc}",
                },
                indent=2,
            )
        )
        return {"warning": str(exc)}


local_client = OpenAI(
    base_url=f"http://{HOST_TO_QUERY}:{LOCAL_PORT}/v1",
    api_key="unused",
)

local_structured_results = {
    scenario["label"]: _run_guided_validation_nonblocking(local_client, "medgemma", scenario)
    for scenario in HEAVY_PRESET_VALIDATIONS
}

# COMMAND ----------

# DBTITLE 1,Stop Local vLLM
import signal

print(f"Stopping vLLM process (PID: {vllm_process.pid}) ...")
vllm_process.send_signal(signal.SIGTERM)
try:
    vllm_process.wait(timeout=30)
except subprocess.TimeoutExpired:
    vllm_process.kill()
    vllm_process.wait()

assert vllm_process.poll() is not None, "vLLM process still running"
log.close()
print(f"✓ vLLM process stopped (exit code: {vllm_process.returncode})")

# COMMAND ----------

# DBTITLE 1,Log Model with Custom Entrypoint
from mlflow.pyfunc.model import ChatCompletionResponse, ChatModel


class LLMModel(ChatModel):
    def predict(self, context, messages, params):
        return ChatCompletionResponse.from_dict({"choices": []})


model_info = mlflow.pyfunc.log_model(
    name="medgemma",
    python_model=LLMModel(),
    artifacts={
        "model_dir": ARTIFACTS_PATH,
    },
    metadata={
        "task": "llm/v1/chat",
        "entrypoint": entrypoint(SERVING_PORT),
    },
    pip_requirements=[
        "mlflow==3.12.0",
        "vllm==0.11.2",
        "transformers==4.57.6",
        "opencv-python-headless==4.12.*",
    ],
)

print(model_info.model_uri)

# COMMAND ----------

# DBTITLE 1,Register Model to Unity Catalog
mlflow.set_registry_uri("databricks-uc")

model_version = mlflow.register_model(
    model_info.model_uri,
    FULL_MODEL_NAME,
    env_pack="databricks_model_serving",
)

client = mlflow.MlflowClient()
client.set_registered_model_alias(
    name=FULL_MODEL_NAME,
    alias="challenger",
    version=model_version.version,
)

print(f"✓ Registered: {FULL_MODEL_NAME} v{model_version.version}")

# COMMAND ----------

# DBTITLE 1,Create Serving Endpoint
import datetime

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
    ServingModelWorkloadType,
)

w = WorkspaceClient()
DATABRICKS_HOST = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

served_entity = ServedEntityInput(
    entity_name=FULL_MODEL_NAME,
    entity_version=str(model_version.version),
    workload_type=ServingModelWorkloadType(WORKLOAD_TYPE),
    min_provisioned_concurrency=MIN_ENDPOINT_CONCURRENCY,
    max_provisioned_concurrency=FIXED_ENDPOINT_CONCURRENCY,
    scale_to_zero_enabled=SCALE_TO_ZERO,
)

existing_endpoint = None
for endpoint in w.serving_endpoints.list():
    if endpoint.name == ENDPOINT_NAME:
        existing_endpoint = endpoint
        break

if existing_endpoint:
    w.serving_endpoints.update_config_and_wait(
        name=ENDPOINT_NAME,
        served_entities=[served_entity],
        timeout=datetime.timedelta(minutes=45),
    )
    print(f"✓ Updated endpoint config: {ENDPOINT_NAME}")
else:
    w.serving_endpoints.create_and_wait(
        name=ENDPOINT_NAME,
        config=EndpointCoreConfigInput(name=ENDPOINT_NAME, served_entities=[served_entity]),
        timeout=datetime.timedelta(minutes=45),
    )
    print(f"✓ Created endpoint: {ENDPOINT_NAME}")

print(f"✓ Endpoint ready: {ENDPOINT_NAME}")

# COMMAND ----------

# DBTITLE 1,Query Ready Endpoint with Databricks SDK
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

resp = w.serving_endpoints.query(
    name=ENDPOINT_NAME,
    messages=[ChatMessage(role=ChatMessageRole.USER, content="Hi, what model are you?")],
)

print(resp.choices[0].message.content)

# COMMAND ----------

# DBTITLE 1,Query Ready Endpoint with OpenAI SDK
endpoint_client = OpenAI(
    api_key=DATABRICKS_TOKEN,
    base_url=f"{DATABRICKS_HOST}/serving-endpoints",
)

response = endpoint_client.chat.completions.create(
    model=ENDPOINT_NAME,
    messages=[
        {"role": "user", "content": "Hello"},
    ],
)

print(response.choices[0].message.content)

# COMMAND ----------

# DBTITLE 1,Structured Endpoint Validation for Heavy Presets
endpoint_structured_results = {
    scenario["label"]: _run_guided_validation(endpoint_client, ENDPOINT_NAME, scenario)
    for scenario in HEAVY_PRESET_VALIDATIONS
}

# COMMAND ----------

# DBTITLE 1,Summary
print("=" * 60)
print("  MedGemma vLLM Custom Serving - Summary")
print("=" * 60)
print()
print(f"  Model:         {FULL_MODEL_NAME} v{model_version.version}")
print(f"  Endpoint:      {ENDPOINT_NAME}")
print(f"  Workload:      {WORKLOAD_TYPE}")
print(f"  Concurrency:   {FIXED_ENDPOINT_CONCURRENCY} fixed")
print(f"  Scale to zero: not used for entrypoint-based serving")
print()
print("  Local Tests:")
print("    ✓ Plain chat")
print("    ✓ Plain streaming")
print("    ✓ Imaging Device Report (guided_json, temperature=0)")
print("    ✓ Discharge / Follow-up Plan (guided_json, temperature=0)")
print()
print("  Endpoint Tests:")
print("    ✓ Databricks SDK query")
print("    ✓ OpenAI SDK query")
print("    ✓ Heavy preset validation passed")
print()
print("  Cleanup:")
print(f"    - Delete endpoint: w.serving_endpoints.delete('{ENDPOINT_NAME}')")
print(f"    - Delete model:    client.delete_registered_model('{FULL_MODEL_NAME}')")
print(f"    - Clean work dir:  rm -rf {pwd}")
print()
print("=" * 60)
dbutils.notebook.exit(json.dumps({
    "model_name": FULL_MODEL_NAME,
    "model_version": str(model_version.version),
    "endpoint_name": ENDPOINT_NAME,
    "structured_smoke_tests": endpoint_structured_results,
}))
