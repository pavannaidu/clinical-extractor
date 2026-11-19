# Databricks notebook source
# MAGIC %md
# MAGIC # Qwen3-4B fine-tuning for biomedical disease extraction
# MAGIC
# MAGIC Adapt `Qwen/Qwen3-4B-Instruct-2507` with QLoRA on the expert-annotated
# MAGIC [NCBI Disease Corpus](https://huggingface.co/datasets/ncbi/ncbi_disease).
# MAGIC The corpus contains PubMed text and disease mention spans, not patient records.
# MAGIC Its dataset card includes the original NCBI public-domain notice and citation.
# MAGIC
# MAGIC This follows the Databricks [Serverless GPU SFT tutorial](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/examples/tutorials/sgc-finetune-qwen3.5-0.8b):
# MAGIC train with TRL, merge the adapter, register a model in Unity Catalog with
# MAGIC `env_pack="databricks_model_serving"`, and serve the merged weights with vLLM.
# MAGIC QLoRA lets the 4B model train on one A10. The endpoint accepts the app's
# MAGIC `guided_json` contract and constrains output with XGrammar.

# COMMAND ----------

# DBTITLE 1,Parameters
dbutils.widgets.text("catalog", "pavan_naidu_catalog")
dbutils.widgets.text("schema", "mlops")
dbutils.widgets.text("registered_model_name", "qwen3_vllm_structured")
dbutils.widgets.text("endpoint_name", "qwen3-4b-finetuned-vllm-endpoint")
dbutils.widgets.text("max_train_examples", "2048")
dbutils.widgets.text("max_eval_examples", "256")
dbutils.widgets.text("num_epochs", "1")
dbutils.widgets.text("learning_rate", "2e-4")
dbutils.widgets.text("max_seq_length", "1024")

# COMMAND ----------

# DBTITLE 1,Dependencies
# Keep training on CUDA 12.8: bitsandbytes 0.49.2 has no CUDA 13.2 binary.
# MAGIC %pip install "mlflow==3.12.0" "databricks-sdk==0.96.0" \
# MAGIC   "torch==2.9.0" "torchvision==0.24.0" \
# MAGIC   "transformers==4.57.6" "trl==0.24.0" \
# MAGIC   "peft==0.18.0" "bitsandbytes==0.49.2" "datasets>=4,<5" \
# MAGIC   "accelerate>=1.10,<2" "huggingface_hub>=0.34,<2"

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# DBTITLE 1,Initialize runtime
import gc
import json
import os
import tempfile
from datetime import timedelta

import mlflow
import torch
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
    ServingModelWorkloadType,
)
from datasets import load_dataset
from huggingface_hub import hf_hub_download, snapshot_download
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

runtime_version = os.environ.get("DATABRICKS_RUNTIME_VERSION", "")
if "AIR_RUNTIME_VERSION" in os.environ and "-gpu" not in runtime_version:
    runtime_version = runtime_version or os.environ["AIR_RUNTIME_VERSION"]
    os.environ["DATABRICKS_RUNTIME_VERSION"] = f"{runtime_version}-gpu"

assert torch.cuda.is_available(), "This notebook requires Serverless GPU compute"

CATALOG = dbutils.widgets.get("catalog")
SCHEMA_NAME = dbutils.widgets.get("schema")
REGISTERED_MODEL_NAME = dbutils.widgets.get("registered_model_name")
ENDPOINT_NAME = dbutils.widgets.get("endpoint_name")
MAX_TRAIN_EXAMPLES = int(dbutils.widgets.get("max_train_examples"))
MAX_EVAL_EXAMPLES = int(dbutils.widgets.get("max_eval_examples"))
NUM_EPOCHS = int(dbutils.widgets.get("num_epochs"))
LEARNING_RATE = float(dbutils.widgets.get("learning_rate"))
MAX_SEQ_LENGTH = int(dbutils.widgets.get("max_seq_length"))

FULL_MODEL_NAME = f"{CATALOG}.{SCHEMA_NAME}.{REGISTERED_MODEL_NAME}"
BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
BASE_MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
DATASET_ID = "ncbi/ncbi_disease"
DATASET_REVISION = "ddf78b541d0004431f0eee441e439b7881e6213d"
WORKDIR = tempfile.mkdtemp(prefix="qwen3_disease_finetune_")
BASE_PATH = os.path.join(WORKDIR, "base_model")
ADAPTER_PATH = os.path.join(WORKDIR, "lora_adapter")
ARTIFACTS_PATH = "qwen3_finetuned"
MERGED_PATH = os.path.join(WORKDIR, ARTIFACTS_PATH)
os.environ["HF_HOME"] = os.path.join(WORKDIR, "hf_cache")
os.chdir(WORKDIR)

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Model: {FULL_MODEL_NAME}")
print(f"Endpoint: {ENDPOINT_NAME}")
print(f"Dataset: {DATASET_ID}@{DATASET_REVISION}")

# COMMAND ----------

# DBTITLE 1,Load the pinned model and corpus
snapshot_download(
    repo_id=BASE_MODEL_ID,
    revision=BASE_MODEL_REVISION,
    local_dir=BASE_PATH,
)

data_files = {
    split: hf_hub_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        revision=DATASET_REVISION,
        filename=f"ncbi_disease/{split}/0000.parquet",
    )
    for split in ("train", "validation", "test")
}
corpus = load_dataset("parquet", data_files=data_files)
tokenizer = AutoTokenizer.from_pretrained(BASE_PATH)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

DISEASE_SCHEMA = {
    "type": "object",
    "properties": {"diseases": {"type": "array", "items": {"type": "string"}}},
    "required": ["diseases"],
    "additionalProperties": False,
}
SYSTEM_PROMPT = (
    "Extract disease mentions from biomedical text. Return exactly one JSON object "
    "with a diseases array containing verbatim mentions in order of first appearance. "
    "Use only text that appears in the source. Return an empty array when none are present."
)


def disease_mentions(tokens, tags):
    if len(tokens) != len(tags):
        raise ValueError("Token and tag counts differ")
    mentions = []
    current = []
    for token, tag in zip(tokens, tags):
        if tag == 1:
            if current:
                mentions.append(" ".join(current))
            current = [token]
        elif tag == 2:
            current.append(token)
        elif tag == 0:
            if current:
                mentions.append(" ".join(current))
            current = []
        else:
            raise ValueError(f"Unexpected NER tag: {tag}")
    if current:
        mentions.append(" ".join(current))
    return list(dict.fromkeys(mentions))


def source_messages(tokens):
    sentence = " ".join(tokens)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Biomedical text:\n{sentence}\n\n"
                f"Return JSON matching this schema:\n{json.dumps(DISEASE_SCHEMA)}"
            ),
        },
    ]


def training_example(row):
    prompt = tokenizer.apply_chat_template(
        source_messages(row["tokens"]),
        tokenize=False,
        add_generation_prompt=True,
    )
    answer = {"diseases": disease_mentions(row["tokens"], row["ner_tags"])}
    return {
        "prompt": prompt,
        "completion": json.dumps(answer, ensure_ascii=False) + tokenizer.eos_token,
    }


def within_context(example):
    return len(
        tokenizer(
            example["prompt"] + example["completion"],
            add_special_tokens=False,
        )["input_ids"]
    ) <= MAX_SEQ_LENGTH


def prepare_split(name, limit):
    source = corpus[name].shuffle(seed=42)
    if limit > 0:
        source = source.select(range(min(limit, len(source))))
    prepared = source.map(training_example, remove_columns=source.column_names)
    prepared = prepared.filter(within_context)
    if len(prepared) == 0:
        raise ValueError(f"No {name} examples fit the context window")
    return prepared


train_dataset = prepare_split("train", MAX_TRAIN_EXAMPLES)
eval_dataset = prepare_split("validation", MAX_EVAL_EXAMPLES)
print(f"Train examples: {len(train_dataset)}")
print(f"Validation examples: {len(eval_dataset)}")
print(train_dataset[0]["completion"])

# COMMAND ----------

# DBTITLE 1,QLoRA fine-tuning
quantization = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    BASE_PATH,
    quantization_config=quantization,
    device_map={"": 0},
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
)
model.config.use_cache = False
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model = get_peft_model(
    model,
    LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
        ],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    ),
)
model.print_trainable_parameters()

mlflow.set_registry_uri("databricks-uc")
workspace_user = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
)
mlflow.set_experiment(f"/Users/{workspace_user}/qwen3-4b-disease-extraction")
run = mlflow.start_run(run_name="qwen3-4b-ncbi-disease-qlora")
mlflow.log_params(
    {
        "base_model": BASE_MODEL_ID,
        "base_revision": BASE_MODEL_REVISION,
        "dataset": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "train_examples": len(train_dataset),
        "validation_examples": len(eval_dataset),
        "epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "max_seq_length": MAX_SEQ_LENGTH,
        "lora_rank": 16,
    }
)

trainer = SFTTrainer(
    model=model,
    args=SFTConfig(
        output_dir=os.path.join(WORKDIR, "training_output"),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        max_length=MAX_SEQ_LENGTH,
        completion_only_loss=True,
        packing=False,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=10,
        report_to=[],
    ),
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
)
train_result = trainer.train()
eval_metrics = trainer.evaluate()
mlflow.log_metric("train_loss", train_result.training_loss)
mlflow.log_metric("validation_loss", eval_metrics["eval_loss"])
trainer.save_model(ADAPTER_PATH)
tokenizer.save_pretrained(ADAPTER_PATH)
print(f"Training steps: {trainer.state.global_step}")
print(f"Train loss: {train_result.training_loss:.4f}")
print(f"Validation loss: {eval_metrics['eval_loss']:.4f}")

del trainer
del model
gc.collect()
torch.cuda.empty_cache()

# COMMAND ----------

# DBTITLE 1,Merge the adapter for vLLM
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_PATH,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
    attn_implementation="sdpa",
)
merged_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH).merge_and_unload()
merged_model.save_pretrained(MERGED_PATH, safe_serialization=True)
tokenizer.save_pretrained(MERGED_PATH)
print(f"Merged weights: {MERGED_PATH}")

del merged_model
del base_model
gc.collect()
torch.cuda.empty_cache()

# COMMAND ----------

# DBTITLE 1,Register the vLLM serving artifact
from mlflow.pyfunc.model import ChatCompletionResponse, ChatModel
from mlflow.utils.databricks_utils import stage_model_for_databricks_model_serving
from mlflow.utils.env_pack import pack_env_for_databricks_model_serving

if not os.environ.get("DATABRICKS_ACCELERATOR"):
    raise RuntimeError("Model registration must run on Serverless GPU AI Runtime")


class EntrypointChatModel(ChatModel):
    def predict(self, context, messages, params):
        return ChatCompletionResponse.from_dict({"choices": []})


entrypoint = (
    "python -u -m vllm.entrypoints.openai.api_server "
    f"--model {ARTIFACTS_PATH} --served-model-name qwen3-4b-finetuned "
    "--host 0.0.0.0 --port 8080 --dtype float16 "
    "--max-model-len 8192 --gpu-memory-utilization 0.85 "
    "--guided-decoding-backend xgrammar:no-fallback"
)
model_info = mlflow.pyfunc.log_model(
    name="qwen3_finetuned",
    python_model=EntrypointChatModel(),
    artifacts={"model_dir": ARTIFACTS_PATH},
    metadata={"task": "llm/v1/chat", "entrypoint": entrypoint},
    pip_requirements=[
        "mlflow==3.12.0",
        "vllm==0.11.2",
        "transformers==4.57.6",
        "opencv-python-headless==4.12.*",
    ],
)
model_client = mlflow.MlflowClient()
from mlflow.exceptions import MlflowException

try:
    model_client.create_registered_model(FULL_MODEL_NAME)
except MlflowException as exc:
    if exc.error_code != "RESOURCE_ALREADY_EXISTS":
        raise
with pack_env_for_databricks_model_serving(
    model_info.model_uri,
    enforce_pip_requirements=True,
) as artifacts_path:
    model_version = model_client._create_model_version(
        name=FULL_MODEL_NAME,
        source=model_info.model_uri,
        model_id=model_info.model_id,
        local_model_path=artifacts_path,
        await_creation_for=900,
    )
try:
    stage_model_for_databricks_model_serving(
        model_name=FULL_MODEL_NAME,
        model_version=str(model_version.version),
    )
except Exception as exc:
    print(f"Serving staging deferred: {exc}")
model_client.set_registered_model_alias(
    name=FULL_MODEL_NAME,
    alias="disease-finetuned",
    version=model_version.version,
)
mlflow.end_run()
print(f"Registered {FULL_MODEL_NAME} v{model_version.version}")

# COMMAND ----------

# DBTITLE 1,Deploy the fine-tuned endpoint
workspace = WorkspaceClient()
served_entity = ServedEntityInput(
    entity_name=FULL_MODEL_NAME,
    entity_version=str(model_version.version),
    workload_type=ServingModelWorkloadType.GPU_MEDIUM,
    min_provisioned_concurrency=4,
    max_provisioned_concurrency=4,
)
existing_names = {endpoint.name for endpoint in workspace.serving_endpoints.list()}
if ENDPOINT_NAME in existing_names:
    workspace.serving_endpoints.update_config_and_wait(
        name=ENDPOINT_NAME,
        served_entities=[served_entity],
        timeout=timedelta(minutes=75),
    )
else:
    workspace.serving_endpoints.create_and_wait(
        name=ENDPOINT_NAME,
        config=EndpointCoreConfigInput(
            name=ENDPOINT_NAME,
            served_entities=[served_entity],
        ),
        timeout=timedelta(minutes=75),
    )

endpoint = workspace.serving_endpoints.get(ENDPOINT_NAME)
assert endpoint.state.ready.value == "READY", endpoint.state
assert endpoint.state.config_update.value == "NOT_UPDATING", endpoint.state
print(f"Endpoint ready: {ENDPOINT_NAME}")

# COMMAND ----------

# DBTITLE 1,Verify the app request contract
test_row = next(
    row for row in corpus["test"]
    if disease_mentions(row["tokens"], row["ner_tags"])
)
response = workspace.api_client.do(
    method="POST",
    path="/serving-endpoints/chat/completions",
    body={
        "model": ENDPOINT_NAME,
        "messages": source_messages(test_row["tokens"]),
        "max_tokens": 256,
        "temperature": 0.0,
        "guided_json": DISEASE_SCHEMA,
        "guided_decoding_backend": "xgrammar:no-fallback",
    },
)
extracted = json.loads(response["choices"][0]["message"]["content"])
assert set(extracted) == {"diseases"}, extracted
assert isinstance(extracted["diseases"], list), extracted
assert all(isinstance(mention, str) for mention in extracted["diseases"]), extracted
print(f"Structured extraction: {json.dumps(extracted)}")

dbutils.notebook.exit(
    json.dumps(
        {
            "model": FULL_MODEL_NAME,
            "model_version": str(model_version.version),
            "endpoint": ENDPOINT_NAME,
            "training_examples": len(train_dataset),
            "validation_loss": eval_metrics["eval_loss"],
            "structured_smoke_test": extracted,
        }
    )
)
