# Databricks notebook source
# DBTITLE 1,Overview
# MAGIC %md
# MAGIC # Qwen3 — Custom LLM Serving with vLLM + XGrammar
# MAGIC
# MAGIC A stricter clone of
# MAGIC `reference/[GOLD] Vllm working e2e with port range.ipynb`, adapted for this
# MAGIC repository and bundle deployment.
# MAGIC
# MAGIC The notebook keeps the reference flow and only adds parameterization for
# MAGIC catalog, schema, model name, endpoint name, workload sizing, and
# MAGIC schema-guided XGrammar tests.

# COMMAND ----------

# DBTITLE 1,Widgets
dbutils.widgets.removeAll()
dbutils.widgets.text("catalog", "pavan_naidu_catalog", "Catalog")
dbutils.widgets.text("schema", "mlops", "Schema")
dbutils.widgets.text("registered_model_name", "qwen3_vllm_structured", "Registered Model Name")
dbutils.widgets.text("endpoint_name", "qwen3-4b-vllm-endpoint", "Endpoint Name")
dbutils.widgets.dropdown("workload_type", "GPU_MEDIUM", ["GPU_SMALL", "GPU_MEDIUM", "GPU_LARGE"], "Workload Type")
dbutils.widgets.dropdown("workload_size", "Small", ["Small", "Medium", "Large"], "Workload Size")
dbutils.widgets.dropdown("scale_to_zero", "false", ["true", "false"], "Scale to Zero")
dbutils.widgets.text("hf_model_name", "Qwen/Qwen3-4B-Instruct-2507", "HuggingFace Model Name")

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

# DBTITLE 1,Set Working Directory to Local Disk
import tempfile

pwd = tempfile.mkdtemp(prefix="qwen3_vllm_")
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

# In the notebook we have allowlisted ports 3000-3999. When actually logging or
# serving the model, Model Serving requires port 8080, so we copy the entrypoint
# command with it.
LOCAL_PORT = 3080
SERVING_PORT = 8080

ARTIFACTS_PATH = "qwen3"
FIXED_ENDPOINT_CONCURRENCY = 4

print(f"Model name: {FULL_MODEL_NAME}")
print(f"Endpoint:   {ENDPOINT_NAME}")
print(f"HF model:   {HF_MODEL_NAME}")
print(f"Artifacts:  {os.path.join(pwd, ARTIFACTS_PATH)}")
print(
    "Serving:    "
    f"workload_type={WORKLOAD_TYPE}, "
    f"fixed_concurrency={FIXED_ENDPOINT_CONCURRENCY}, "
    f"scale_to_zero_requested={SCALE_TO_ZERO}"
)

# COMMAND ----------

# DBTITLE 1,Download Qwen from HuggingFace
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=HF_MODEL_NAME,
    local_dir=ARTIFACTS_PATH,
)

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
    argv = [
        "python",
        "-u",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        ARTIFACTS_PATH,
        "--served-model-name",
        "qwen",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--dtype",
        "float16",
        "--max-model-len",
        "8192",
        "--gpu-memory-utilization",
        "0.85",
        "--guided-decoding-backend",
        "xgrammar:no-fallback",
        # "--enforce-eager",
    ]
    return " ".join(argv)


print(entrypoint(LOCAL_PORT))

# COMMAND ----------

# DBTITLE 1,Start Local vLLM Server
import subprocess

log = open("process.log", "w")
subprocess.Popen(
    ["bash", "-lc", entrypoint(LOCAL_PORT)],
    stdout=log,
    stderr=subprocess.STDOUT,
    text=True,
    start_new_session=True,
)

# COMMAND ----------

# DBTITLE 1,Wait for Startup
import pathlib

STARTUP_TIMEOUT = 300
deadline = time.time() + STARTUP_TIMEOUT

while time.time() < deadline:
    log_content = pathlib.Path("process.log").read_text() if pathlib.Path("process.log").exists() else ""
    if "Application startup complete" in log_content:
        break
    time.sleep(5)
else:
    log_tail = pathlib.Path("process.log").read_text()[-6000:] if pathlib.Path("process.log").exists() else ""
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
        "messages": [{"role": "user", "content": "Tell me a story that is about 300 words!"}],
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

# DBTITLE 1,Structured Local Query with XGrammar
from openai import OpenAI

medication_schema = {
    "type": "object",
    "properties": {
        "medications": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "dosage": {"type": "string"},
                    "route": {"type": "string"},
                    "frequency": {"type": "string"},
                    "duration": {"type": "string"},
                    "indication": {"type": "string"},
                },
                "required": ["name", "dosage"],
            },
        }
    },
    "required": ["medications"],
}

medication_prompt = (
    "Extract medication details as structured JSON.\n\n"
    "Prescribe Amoxicillin 500mg orally three times daily for 10 days to treat "
    "bacterial sinusitis. Also continue Lisinopril 10mg daily for hypertension."
)

local_client = OpenAI(
    base_url=f"http://{HOST_TO_QUERY}:{LOCAL_PORT}/v1",
    api_key="unused",
)

response = local_client.chat.completions.create(
    model="qwen",
    messages=[{"role": "user", "content": medication_prompt}],
    max_tokens=512,
    temperature=0.1,
    extra_body={
        "guided_json": medication_schema,
        "guided_decoding_backend": "xgrammar:no-fallback",
    },
)

structured_local = json.loads(response.choices[0].message.content)
assert "medications" in structured_local, structured_local
print(json.dumps(structured_local, indent=2))

# COMMAND ----------

# DBTITLE 1,Structured Local Streaming with XGrammar
stream = local_client.chat.completions.create(
    model="qwen",
    messages=[{"role": "user", "content": medication_prompt}],
    max_tokens=512,
    temperature=0.1,
    stream=True,
    extra_body={
        "guided_json": medication_schema,
        "guided_decoding_backend": "xgrammar:no-fallback",
    },
)

chunks = []
for event in stream:
    delta = event.choices[0].delta
    if delta.content:
        chunks.append(delta.content)

structured_stream = json.loads("".join(chunks))
assert "medications" in structured_stream, structured_stream
print(json.dumps(structured_stream, indent=2))

# COMMAND ----------

# DBTITLE 1,Stop Local vLLM
# MAGIC %sh
# MAGIC pkill -f vllm.entrypoints.openai.api_server

# COMMAND ----------

# DBTITLE 1,Log Model with Custom Entrypoint
import mlflow
from mlflow.pyfunc.model import ChatCompletionResponse, ChatModel


class LLMModel(ChatModel):
    def predict(self, context, messages, params):
        return ChatCompletionResponse.from_dict({"choices": []})


model_info = mlflow.pyfunc.log_model(
    name="qwen",
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
    min_provisioned_concurrency=FIXED_ENDPOINT_CONCURRENCY,
    max_provisioned_concurrency=FIXED_ENDPOINT_CONCURRENCY,
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
from openai import OpenAI

client = OpenAI(
    api_key=DATABRICKS_TOKEN,
    base_url=f"{DATABRICKS_HOST}/serving-endpoints",
)

response = client.chat.completions.create(
    model=ENDPOINT_NAME,
    messages=[
        {"role": "user", "content": "Hello"},
    ],
)

print(response.choices[0].message.content)

# COMMAND ----------

# DBTITLE 1,Structured Endpoint Query with XGrammar
response = client.chat.completions.create(
    model=ENDPOINT_NAME,
    messages=[{"role": "user", "content": medication_prompt}],
    max_tokens=512,
    temperature=0.1,
    extra_body={
        "guided_json": medication_schema,
        "guided_decoding_backend": "xgrammar:no-fallback",
    },
)

structured_endpoint = json.loads(response.choices[0].message.content)
from jsonschema import validate
validate(instance=structured_endpoint, schema=medication_schema)
print(json.dumps(structured_endpoint, indent=2))

# COMMAND ----------

# DBTITLE 1,Stream from Ready Endpoint
stream = client.chat.completions.create(
    model=ENDPOINT_NAME,
    messages=[
        {"role": "user", "content": "Hello, tell me a 200 word story"},
    ],
    stream=True,
)

for event in stream:
    delta = event.choices[0].delta
    print(delta.content, end="|")

print()
dbutils.notebook.exit(json.dumps({
    "model_name": FULL_MODEL_NAME,
    "model_version": str(model_version.version),
    "endpoint_name": ENDPOINT_NAME,
    "structured_smoke_test": structured_endpoint,
}))
