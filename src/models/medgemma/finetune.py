# Databricks notebook source
# DBTITLE 1,Overview
# MAGIC %md
# MAGIC # MedGemma - QLoRA Fine-Tuning for Clinical Extraction
# MAGIC
# MAGIC End-to-end notebook for domain-adapting `google/medgemma-1.5-4b-it` via QLoRA
# MAGIC fine-tuning, then registering and deploying the merged model.
# MAGIC
# MAGIC ## Flow
# MAGIC 1. Download MedGemma from Hugging Face
# MAGIC 2. Generate synthetic clinical extraction training data
# MAGIC 3. QLoRA fine-tune with PEFT + TRL SFTTrainer
# MAGIC 4. Merge LoRA adapter into base model
# MAGIC 5. Validate merged model locally with XGrammar constrained generation
# MAGIC 6. Log & register to Unity Catalog (reuses existing `model.py`)
# MAGIC 7. Create or update a Model Serving endpoint and validate remotely
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - GPU cluster (A10 or better — QLoRA 4-bit fits in ~15 GB VRAM)
# MAGIC - A Hugging Face token in the configured `hf_secret_scope` / `hf_secret_key` secret

# COMMAND ----------

# DBTITLE 1,Widgets
dbutils.widgets.removeAll()
dbutils.widgets.text("catalog", "main", "Catalog")
dbutils.widgets.text("schema", "mlops", "Schema")
dbutils.widgets.text("registered_model_name", "medgemma_xgrammar_finetuned", "Registered Model Name")
dbutils.widgets.text("endpoint_name", "medgemma-finetuned-endpoint", "Endpoint Name")
dbutils.widgets.dropdown("workload_type", "GPU_MEDIUM", ["GPU_SMALL", "GPU_MEDIUM", "GPU_LARGE"], "Workload Type")
dbutils.widgets.dropdown("scale_to_zero", "true", ["true", "false"], "Scale to Zero")
dbutils.widgets.text("hf_model_name", "google/medgemma-1.5-4b-it", "HuggingFace Model Name")
dbutils.widgets.text("hf_secret_scope", "clinical-extractor", "Hugging Face Secret Scope")
dbutils.widgets.text("hf_secret_key", "hf_token", "Hugging Face Secret Key")
dbutils.widgets.text("lora_rank", "16", "LoRA Rank")
dbutils.widgets.text("lora_alpha", "32", "LoRA Alpha")
dbutils.widgets.text("num_epochs", "3", "Training Epochs")
dbutils.widgets.text("learning_rate", "2e-4", "Learning Rate")
dbutils.widgets.text("max_seq_length", "2048", "Max Sequence Length")
dbutils.widgets.text("num_training_examples", "50", "Examples per Category")

# COMMAND ----------

# DBTITLE 1,GPU Check
# MAGIC %sh
# MAGIC nvidia-smi

# COMMAND ----------

# DBTITLE 1,Install Dependencies
# MAGIC %pip install "peft==0.14.0" "trl==0.16.1" "bitsandbytes==0.46.1" "datasets==3.6.0" \
# MAGIC   "transformers==4.53.3" "accelerate==1.4.0" "xgrammar==0.1.31" \
# MAGIC   "huggingface_hub==0.36.2" "hf_transfer==0.1.9" "mlflow==3.12.0" "databricks-sdk==0.96.0" "openai==2.17.0"

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
import peft
import trl

print(f"PyTorch:      {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"PEFT:         {peft.__version__}")
print(f"TRL:          {trl.__version__}")
print(f"MLflow:       {mlflow.__version__}")

assert torch.cuda.is_available(), "CUDA not available"
print(f"CUDA devices: {torch.cuda.device_count()}")
print(f"GPU:          {torch.cuda.get_device_name()}")
print(f"VRAM:         {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print("\n✓ All version checks passed")

# COMMAND ----------

# DBTITLE 1,Configuration
import tempfile

CATALOG = dbutils.widgets.get("catalog")
SCHEMA_NAME = dbutils.widgets.get("schema")
REGISTERED_MODEL_NAME = dbutils.widgets.get("registered_model_name")
ENDPOINT_NAME = dbutils.widgets.get("endpoint_name")
WORKLOAD_TYPE = dbutils.widgets.get("workload_type")
SCALE_TO_ZERO = dbutils.widgets.get("scale_to_zero") == "true"
HF_MODEL_NAME = dbutils.widgets.get("hf_model_name")

LORA_RANK = int(dbutils.widgets.get("lora_rank"))
LORA_ALPHA = int(dbutils.widgets.get("lora_alpha"))
NUM_EPOCHS = int(dbutils.widgets.get("num_epochs"))
LEARNING_RATE = float(dbutils.widgets.get("learning_rate"))
MAX_SEQ_LENGTH = int(dbutils.widgets.get("max_seq_length"))
NUM_TRAINING_EXAMPLES = int(dbutils.widgets.get("num_training_examples"))

FULL_MODEL_NAME = f"{CATALOG}.{SCHEMA_NAME}.{REGISTERED_MODEL_NAME}"
ARTIFACTS_PATH = "medgemma_weights"

pwd = tempfile.mkdtemp(prefix="medgemma_finetune_")
os.chdir(pwd)

print(f"Model name:      {FULL_MODEL_NAME}")
print(f"Endpoint:        {ENDPOINT_NAME}")
print(f"HF model:        {HF_MODEL_NAME}")
print(f"Working dir:     {pwd}")
print(f"LoRA:            rank={LORA_RANK}, alpha={LORA_ALPHA}")
print(f"Training:        epochs={NUM_EPOCHS}, lr={LEARNING_RATE}, max_seq={MAX_SEQ_LENGTH}")
print(f"Examples/cat:    {NUM_TRAINING_EXAMPLES}")

# COMMAND ----------

# DBTITLE 1,Download MedGemma from HuggingFace
import pathlib
from huggingface_hub import snapshot_download

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
hf_token = dbutils.secrets.get(
    scope=dbutils.widgets.get("hf_secret_scope"),
    key=dbutils.widgets.get("hf_secret_key"),
)

snapshot_download(
    repo_id=HF_MODEL_NAME,
    local_dir=ARTIFACTS_PATH,
    token=hf_token,
)

assert (pathlib.Path(ARTIFACTS_PATH) / "config.json").exists(), f"config.json not found in {ARTIFACTS_PATH}"
print(f"✓ Model downloaded to {os.path.join(pwd, ARTIFACTS_PATH)}")

# COMMAND ----------

# DBTITLE 1,Generate Synthetic Training Data
# MAGIC %md
# MAGIC Generate chat-formatted SFT training examples for ~12 clinical extraction
# MAGIC categories. Each example: system prompt + user message (clinical text + schema) +
# MAGIC assistant response (ground-truth JSON).

# COMMAND ----------

import json
import random

random.seed(42)

SYSTEM_PROMPT = (
    "You are a clinical information extraction assistant. Extract only information "
    "explicitly supported by the provided medical text. Return output that matches "
    "the requested schema exactly. Do not invent facts; when the source text does "
    "not provide a value, leave it null or empty if the schema allows."
)

# ── Schema definitions (matching presets.py structure) ──────────────────────

MEDICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "medications": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "dosage": {"type": "string"},
                    "route": {"type": ["string", "null"]},
                    "frequency": {"type": ["string", "null"]},
                    "duration": {"type": ["string", "null"]},
                    "indication": {"type": ["string", "null"]},
                },
                "required": ["name", "dosage", "route", "frequency", "duration", "indication"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["medications"],
    "additionalProperties": False,
}

DIAGNOSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "diagnoses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "condition": {"type": "string"},
                    "certainty": {"type": "string", "enum": ["confirmed", "suspected", "ruled_out"]},
                    "severity": {"type": ["string", "null"]},
                    "onset": {"type": ["string", "null"]},
                    "notes": {"type": ["string", "null"]},
                },
                "required": ["condition", "certainty", "severity", "onset", "notes"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["diagnoses"],
    "additionalProperties": False,
}

VITALS_SCHEMA = {
    "type": "object",
    "properties": {
        "blood_pressure": {"type": ["string", "null"]},
        "heart_rate": {"type": ["integer", "null"]},
        "respiratory_rate": {"type": ["integer", "null"]},
        "spo2_percent": {"type": ["number", "null"]},
        "temperature_f": {"type": ["number", "null"]},
        "oxygen_support": {"type": ["string", "null"]},
    },
    "required": ["blood_pressure", "heart_rate", "respiratory_rate", "spo2_percent", "temperature_f", "oxygen_support"],
    "additionalProperties": False,
}

LAB_SCHEMA = {
    "type": "object",
    "properties": {
        "panels": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "panel_name": {"type": "string"},
                    "results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "test_name": {"type": "string"},
                                "value": {"type": ["number", "null"]},
                                "unit": {"type": ["string", "null"]},
                                "flag": {"type": ["string", "null"], "enum": ["normal", "high", "low", "critical", None]},
                            },
                            "required": ["test_name", "value", "unit", "flag"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["panel_name", "results"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["panels"],
    "additionalProperties": False,
}

ALLERGY_SCHEMA = {
    "type": "object",
    "properties": {
        "allergies": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "allergen": {"type": "string"},
                    "category": {"type": ["string", "null"], "enum": ["drug", "food", "environment", "contrast", "latex", "other", None]},
                    "reaction": {"type": ["string", "null"]},
                    "severity": {"type": ["string", "null"], "enum": ["mild", "moderate", "severe", "anaphylaxis", None]},
                },
                "required": ["allergen", "category", "reaction", "severity"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["allergies"],
    "additionalProperties": False,
}

PROCEDURE_SCHEMA = {
    "type": "object",
    "properties": {
        "procedures": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "procedure_name": {"type": "string"},
                    "body_site": {"type": ["string", "null"]},
                    "status": {"type": ["string", "null"], "enum": ["completed", "planned", "cancelled", "in_progress", None]},
                    "indication": {"type": ["string", "null"]},
                    "outcome": {"type": ["string", "null"]},
                },
                "required": ["procedure_name", "body_site", "status", "indication", "outcome"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["procedures"],
    "additionalProperties": False,
}

DISCHARGE_SCHEMA = {
    "type": "object",
    "properties": {
        "disposition": {"type": "string"},
        "diagnoses": {"type": "array", "items": {"type": "string"}},
        "medications": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "instructions": {"type": ["string", "null"]},
                    "duration": {"type": ["string", "null"]},
                },
                "required": ["name", "instructions", "duration"],
                "additionalProperties": False,
            },
        },
        "follow_up_appointments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "service": {"type": "string"},
                    "timeframe": {"type": ["string", "null"]},
                    "reason": {"type": ["string", "null"]},
                },
                "required": ["service", "timeframe", "reason"],
                "additionalProperties": False,
            },
        },
        "return_precautions": {"type": "array", "items": {"type": "string"}},
        "patient_instructions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["disposition", "diagnoses", "medications", "follow_up_appointments", "return_precautions", "patient_instructions"],
    "additionalProperties": False,
}

IMAGING_SCHEMA = {
    "type": "object",
    "properties": {
        "modality": {"type": "string"},
        "body_part": {"type": "string"},
        "findings": {"type": "array", "items": {"type": "string"}},
        "impression": {"type": "string"},
        "critical_result": {"type": ["boolean", "null"]},
        "recommendation": {"type": ["string", "null"]},
    },
    "required": ["modality", "body_part", "findings", "impression", "critical_result", "recommendation"],
    "additionalProperties": False,
}

# ── Data pools for randomization ────────────────────────────────────────────

MEDICATIONS = [
    ("Metoprolol Succinate", "25 mg", "oral", "once daily", "heart rate control"),
    ("Aspirin", "81 mg", "oral", "once daily", "antiplatelet therapy"),
    ("Lisinopril", "10 mg", "oral", "once daily", "hypertension"),
    ("Metformin", "500 mg", "oral", "twice daily", "type 2 diabetes"),
    ("Atorvastatin", "40 mg", "oral", "at bedtime", "hyperlipidemia"),
    ("Amoxicillin", "500 mg", "oral", "three times daily", "bacterial infection"),
    ("Furosemide", "20 mg", "oral", "once daily", "volume overload"),
    ("Omeprazole", "20 mg", "oral", "once daily", "GERD"),
    ("Amlodipine", "5 mg", "oral", "once daily", "hypertension"),
    ("Prednisone", "10 mg", "oral", "once daily", "inflammation"),
    ("Doxycycline", "100 mg", "oral", "twice daily", "pneumonia"),
    ("Warfarin", "5 mg", "oral", "once daily", "anticoagulation"),
    ("Albuterol", "2 puffs", "inhaled", "every 4 hours as needed", "bronchospasm"),
    ("Insulin Glargine", "18 units", "subcutaneous", "at bedtime", "diabetes"),
    ("Ceftriaxone", "1 g", "intravenous", "once daily", "sepsis"),
    ("Vancomycin", "1 g", "intravenous", "every 12 hours", "MRSA infection"),
]

DURATIONS = ["7 days", "10 days", "14 days", "5 days", "30 days", "ongoing", None]

CONDITIONS = [
    ("community-acquired pneumonia", "confirmed", "moderate", "2 days ago"),
    ("acute coronary syndrome", "confirmed", "severe", "acute onset"),
    ("urinary tract infection", "confirmed", "mild", "3 days ago"),
    ("pulmonary embolism", "suspected", None, "acute onset"),
    ("type 2 diabetes mellitus", "confirmed", "moderate", "chronic"),
    ("congestive heart failure", "confirmed", "moderate", "exacerbation 1 week ago"),
    ("cellulitis", "confirmed", "moderate", "4 days ago"),
    ("atrial fibrillation", "confirmed", None, "chronic"),
    ("deep vein thrombosis", "suspected", None, "1 day ago"),
    ("chronic kidney disease", "confirmed", "stage 3", "chronic"),
]

ALLERGENS = [
    ("Penicillin", "drug", "anaphylaxis", "severe"),
    ("Sulfa drugs", "drug", "rash and hives", "moderate"),
    ("Shellfish", "food", "urticaria", "mild"),
    ("Iodinated contrast", "contrast", "throat tightness and hives", "severe"),
    ("Latex", "latex", "localized itching", "mild"),
    ("Codeine", "drug", "nausea and vomiting", "moderate"),
    ("Peanuts", "food", "anaphylaxis", "anaphylaxis"),
    ("Amoxicillin", "drug", "maculopapular rash", "mild"),
]

PROCEDURES = [
    ("Laparoscopic appendectomy", "right lower quadrant", "completed", "acute appendicitis", "successful removal"),
    ("Right thoracentesis", "right hemithorax", "completed", "symptomatic pleural effusion", "900 mL fluid removed"),
    ("Central venous catheter placement", "right internal jugular", "completed", "IV access", "confirmed by chest x-ray"),
    ("Percutaneous coronary intervention", "LAD", "completed", "acute MI", "successful revascularization"),
    ("Lumbar puncture", "L3-L4 interspace", "completed", "meningitis workup", "CSF sent for analysis"),
    ("Endotracheal intubation", "trachea", "completed", "respiratory failure", "ETT confirmed at 22 cm"),
]

IMAGING_STUDIES = [
    ("X-ray", "Chest", ["bibasilar opacities", "small bilateral pleural effusions"], "worsening pulmonary edema", False, "correlate with fluid balance"),
    ("CT", "Abdomen/Pelvis", ["7 mm appendiceal diameter", "periappendiceal fat stranding"], "acute appendicitis", True, "surgical consultation recommended"),
    ("MRI", "Brain", ["no acute intracranial abnormality", "age-appropriate volume loss"], "negative for acute stroke", False, None),
    ("Ultrasound", "Right upper quadrant", ["gallbladder wall thickening", "pericholecystic fluid", "positive sonographic Murphy sign"], "acute cholecystitis", True, "surgical consultation"),
    ("CT", "Chest", ["segmental filling defect in right lower lobe pulmonary artery"], "acute pulmonary embolism", True, "initiate anticoagulation"),
    ("X-ray", "Left ankle", ["non-displaced distal fibula fracture"], "lateral malleolus fracture", False, "orthopedic follow-up in 1 week"),
]

LAB_PANELS = [
    ("BMP", [
        ("Sodium", 134, "mmol/L", "low"), ("Potassium", 4.2, "mmol/L", "normal"),
        ("Creatinine", 1.6, "mg/dL", "high"), ("Glucose", 186, "mg/dL", "high"),
        ("BUN", 28, "mg/dL", "high"), ("Chloride", 101, "mmol/L", "normal"),
    ]),
    ("CBC", [
        ("WBC", 12.4, "K/uL", "high"), ("Hemoglobin", 13.2, "g/dL", "normal"),
        ("Platelets", 245, "K/uL", "normal"), ("Neutrophils", 78, "%", "high"),
    ]),
    ("Cardiac", [
        ("Troponin I", 0.04, "ng/mL", "normal"), ("BNP", 890, "pg/mL", "high"),
        ("CK-MB", 3.2, "ng/mL", "normal"),
    ]),
    ("Hepatic", [
        ("ALT", 62, "U/L", "high"), ("AST", 48, "U/L", "high"),
        ("Alkaline Phosphatase", 120, "U/L", "normal"), ("Total Bilirubin", 1.8, "mg/dL", "high"),
        ("Albumin", 3.2, "g/dL", "low"),
    ]),
    ("Coagulation", [
        ("PT", 14.2, "seconds", "normal"), ("INR", 1.1, None, "normal"),
        ("PTT", 32, "seconds", "normal"),
    ]),
]


def _jitter_value(val, pct=0.15):
    """Add random noise to a numeric value."""
    if val is None:
        return None
    delta = val * pct
    return round(val + random.uniform(-delta, delta), 1)


def _make_example(category, schema, prompt_text, ground_truth):
    """Build a single chat-formatted training example."""
    user_content = f"{prompt_text}\n\nExtract according to this JSON schema:\n{json.dumps(schema)}"
    return {
        "category": category,
        "messages": json.dumps([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": json.dumps(ground_truth)},
        ]),
    }


# ── Generate examples per category ─────────────────────────────────────────

training_examples = []

for _ in range(NUM_TRAINING_EXAMPLES):
    # Medication extraction
    n_meds = random.randint(1, 4)
    meds = random.sample(MEDICATIONS, n_meds)
    prompt_parts = []
    gt_meds = []
    for name, dose, route, freq, indication in meds:
        dur = random.choice(DURATIONS)
        dur_text = f" for {dur}" if dur else ""
        prompt_parts.append(f"{name} {dose} {route} {freq}{dur_text} for {indication}")
        gt_meds.append({"name": name, "dosage": dose, "route": route, "frequency": freq, "duration": dur, "indication": indication})
    prompt_text = random.choice([
        "Extract medication info: ",
        "Parse the following medication orders: ",
        "Identify all medications from this note: ",
        "Medication reconciliation: ",
    ]) + ". ".join(prompt_parts) + "."
    training_examples.append(_make_example("medication", MEDICATION_SCHEMA, prompt_text, {"medications": gt_meds}))

    # Diagnosis extraction
    n_dx = random.randint(1, 3)
    dxs = random.sample(CONDITIONS, n_dx)
    prompt_parts = []
    gt_dxs = []
    for cond, cert, sev, onset in dxs:
        sev_text = f", {sev} severity" if sev else ""
        prompt_parts.append(f"{cond} ({cert}{sev_text}, onset {onset})")
        gt_dxs.append({"condition": cond, "certainty": cert, "severity": sev, "onset": onset, "notes": None})
    prompt_text = random.choice([
        "Extract diagnoses: ",
        "Clinical impression: ",
        "Assessment and diagnoses: ",
    ]) + ". ".join(prompt_parts) + "."
    training_examples.append(_make_example("diagnosis", DIAGNOSIS_SCHEMA, prompt_text, {"diagnoses": gt_dxs}))

    # Vitals extraction
    sys_bp = random.randint(90, 180)
    dia_bp = random.randint(55, 110)
    hr = random.randint(55, 130)
    rr = random.randint(12, 28)
    spo2 = random.randint(88, 100)
    temp = round(random.uniform(97.0, 103.5), 1)
    o2_support = random.choice([None, "room air", "2L nasal cannula", "4L nasal cannula", "6L face mask", "15L non-rebreather"])
    o2_text = f", {o2_support}" if o2_support else ""
    prompt_text = random.choice([
        "Extract vital signs: ",
        "Patient vitals: ",
        "Bedside monitoring: ",
    ]) + f"BP {sys_bp}/{dia_bp} mmHg, HR {hr} bpm, RR {rr}, SpO2 {spo2}%, Temp {temp}F{o2_text}."
    training_examples.append(_make_example("vitals", VITALS_SCHEMA, prompt_text, {
        "blood_pressure": f"{sys_bp}/{dia_bp}",
        "heart_rate": hr,
        "respiratory_rate": rr,
        "spo2_percent": spo2,
        "temperature_f": temp,
        "oxygen_support": o2_support,
    }))

    # Lab results extraction
    panel_name, base_results = random.choice(LAB_PANELS)
    gt_results = []
    result_texts = []
    for test_name, val, unit, flag in base_results:
        jval = _jitter_value(val)
        unit_text = f" {unit}" if unit else ""
        result_texts.append(f"{test_name} {jval}{unit_text} ({flag})")
        gt_results.append({"test_name": test_name, "value": jval, "unit": unit, "flag": flag})
    prompt_text = random.choice([
        "Extract lab results: ",
        "Lab panel report: ",
        "Chemistry analyzer results: ",
    ]) + f"{panel_name} panel: " + ", ".join(result_texts) + "."
    training_examples.append(_make_example("lab", LAB_SCHEMA, prompt_text, {"panels": [{"panel_name": panel_name, "results": gt_results}]}))

    # Allergy extraction
    n_alg = random.randint(1, 3)
    algs = random.sample(ALLERGENS, n_alg)
    prompt_parts = []
    gt_algs = []
    for allergen, cat, reaction, severity in algs:
        prompt_parts.append(f"{allergen}: {reaction} ({severity})")
        gt_algs.append({"allergen": allergen, "category": cat, "reaction": reaction, "severity": severity})
    prompt_text = random.choice([
        "Extract allergy information: ",
        "Allergy review: ",
        "Allergies documented: ",
    ]) + ". ".join(prompt_parts) + "."
    training_examples.append(_make_example("allergy", ALLERGY_SCHEMA, prompt_text, {"allergies": gt_algs}))

    # Procedure extraction
    proc_name, body_site, status, indication, outcome = random.choice(PROCEDURES)
    prompt_text = random.choice([
        "Extract procedure details: ",
        "Procedure note: ",
        "Operative summary: ",
    ]) + f"{proc_name} performed on {body_site}. Indication: {indication}. Outcome: {outcome}."
    training_examples.append(_make_example("procedure", PROCEDURE_SCHEMA, prompt_text, {
        "procedures": [{"procedure_name": proc_name, "body_site": body_site, "status": status, "indication": indication, "outcome": outcome}]
    }))

    # Imaging extraction
    modality, body_part, findings, impression, critical, rec = random.choice(IMAGING_STUDIES)
    findings_text = "; ".join(findings)
    rec_text = f" Recommendation: {rec}." if rec else ""
    critical_text = " This is a critical result." if critical else ""
    prompt_text = random.choice([
        "Extract imaging findings: ",
        "Radiology report: ",
        "Imaging study interpretation: ",
    ]) + f"{modality} of {body_part}. Findings: {findings_text}. Impression: {impression}.{critical_text}{rec_text}"
    training_examples.append(_make_example("imaging", IMAGING_SCHEMA, prompt_text, {
        "modality": modality, "body_part": body_part, "findings": findings,
        "impression": impression, "critical_result": critical, "recommendation": rec,
    }))

    # Discharge plan extraction
    n_meds = random.randint(1, 3)
    dc_meds = random.sample(MEDICATIONS, n_meds)
    dc_dx = random.sample(CONDITIONS, random.randint(1, 2))
    disposition = random.choice(["home", "home with home health", "skilled nursing facility", "rehabilitation facility"])
    gt_meds = [{"name": m[0], "instructions": f"{m[1]} {m[3]}", "duration": random.choice(DURATIONS)} for m in dc_meds]
    gt_appts = [{"service": random.choice(["Primary Care", "Cardiology", "Pulmonology", "Endocrinology"]),
                 "timeframe": random.choice(["3 days", "1 week", "2 weeks"]),
                 "reason": "follow-up"}]
    gt_precautions = [random.choice([
        "Return for worsening shortness of breath",
        "Return for chest pain or pressure",
        "Return for persistent fever over 101F",
        "Return for signs of bleeding",
    ])]
    gt_instructions = [random.choice([
        "Encourage hydration",
        "Resume normal diet as tolerated",
        "Activity as tolerated",
        "Incentive spirometry 10 times per hour while awake",
    ])]
    med_text = "; ".join([f"{m[0]} {m[1]} {m[3]}" for m in dc_meds])
    dx_text = ", ".join([d[0] for d in dc_dx])
    prompt_text = f"Discharge summary: Disposition {disposition}. Diagnoses: {dx_text}. Medications: {med_text}. Follow-up with {gt_appts[0]['service']} in {gt_appts[0]['timeframe']}. {gt_precautions[0]}. {gt_instructions[0]}."
    training_examples.append(_make_example("discharge", DISCHARGE_SCHEMA, prompt_text, {
        "disposition": disposition,
        "diagnoses": [d[0] for d in dc_dx],
        "medications": gt_meds,
        "follow_up_appointments": gt_appts,
        "return_precautions": gt_precautions,
        "patient_instructions": gt_instructions,
    }))

random.shuffle(training_examples)

categories = {}
for ex in training_examples:
    categories[ex["category"]] = categories.get(ex["category"], 0) + 1

print(f"✓ Generated {len(training_examples)} training examples")
for cat, count in sorted(categories.items()):
    print(f"  {cat}: {count}")

# COMMAND ----------

# DBTITLE 1,Preview Training Examples
sample = json.loads(training_examples[0]["messages"])
print("System:", sample[0]["content"][:80], "...")
print()
print("User:", sample[1]["content"][:200], "...")
print()
print("Assistant:", sample[2]["content"][:200], "...")

# COMMAND ----------

# DBTITLE 1,Load Tokenizer and Prepare Dataset
from datasets import Dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(ARTIFACTS_PATH)

dataset = Dataset.from_list(training_examples)


def formatting_func(example):
    messages = json.loads(example["messages"])
    return tokenizer.apply_chat_template(messages, tokenize=False)


# Verify formatting on one example
sample_formatted = formatting_func(training_examples[0])
print(f"Formatted length: {len(sample_formatted)} chars")
print(sample_formatted[:500])

# COMMAND ----------

# DBTITLE 1,Load Base Model with QLoRA
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    ARTIFACTS_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="sdpa",
    trust_remote_code=True,
)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type=TaskType.CAUSAL_LM,
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print(f"\n✓ Model loaded with QLoRA (rank={LORA_RANK}, alpha={LORA_ALPHA})")
print(f"  VRAM used: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

# COMMAND ----------

# DBTITLE 1,Train with SFTTrainer
from trl import SFTConfig, SFTTrainer
from transformers import DataCollatorForLanguageModeling

mlflow.set_experiment(
    f"/Users/{dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()}/medgemma-finetune"
)


# Gemma3 multimodal wrapper requires token_type_ids during training to
# distinguish image vs text tokens. For text-only fine-tuning we inject
# all-zero token_type_ids (= text) via a custom data collator that also
# handles padding for variable-length sequences.
class Gemma3TextCollator:
    def __init__(self, tokenizer, max_length=None):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features):
        # Pad input_ids, attention_mask, and labels
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]

        batch = self.tokenizer.pad(
            {"input_ids": input_ids},
            padding=True,
            return_tensors="pt",
        )
        # Pad labels with -100 (ignore index)
        max_len = batch["input_ids"].shape[1]
        padded_labels = []
        for lbl in labels:
            pad_len = max_len - len(lbl)
            padding = [-100] * pad_len
            padded_labels.append(
                padding + lbl if self.tokenizer.padding_side == "left" else lbl + padding
            )
        batch["labels"] = torch.tensor(padded_labels)
        # Inject token_type_ids (all zeros = text tokens)
        batch["token_type_ids"] = torch.zeros_like(batch["input_ids"])
        return batch


data_collator = Gemma3TextCollator(tokenizer=tokenizer, max_length=MAX_SEQ_LENGTH)

sft_config = SFTConfig(
    output_dir="./sft_output",
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    report_to="mlflow",
    dataset_kwargs={"skip_prepare_dataset": True},
)

# Pre-tokenize the dataset since we're using a custom collator
def tokenize_func(example):
    messages = json.loads(example["messages"])
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    tokens = tokenizer(text, truncation=True, max_length=MAX_SEQ_LENGTH)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(tokenize_func, remove_columns=dataset.column_names)

trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
    args=sft_config,
)

print("Starting training...")
train_result = trainer.train()

print(f"\n✓ Training complete")
print(f"  Total steps:  {trainer.state.global_step}")
print(f"  Final loss:   {train_result.training_loss:.4f}")
print(f"  Peak VRAM:    {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")

# COMMAND ----------

# DBTITLE 1,Save LoRA Adapter
ADAPTER_PATH = "./lora_adapter"

trainer.save_model(ADAPTER_PATH)
tokenizer.save_pretrained(ADAPTER_PATH)

print(f"✓ Adapter saved to {ADAPTER_PATH}")

# Free quantized model from GPU
del model
del trainer
torch.cuda.empty_cache()
import gc
gc.collect()

print(f"  VRAM after cleanup: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

# COMMAND ----------

# DBTITLE 1,Merge Adapter into Base Model
from peft import PeftModel

print("Loading base model in bf16 for merge...")
base_model = AutoModelForCausalLM.from_pretrained(
    ARTIFACTS_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

print("Loading adapter and merging...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model = model.merge_and_unload()

MERGED_PATH = "merged_model"
# Gemma3's nested output head shares storage with the token embeddings.
# Give safetensors an independent copy while preserving the model's weights.
def merged_state_for_save(model):
    state = model.state_dict()
    for name, tensor in state.items():
        if name.endswith("lm_head.weight"):
            state[name] = tensor.clone()
    return state


merged_state = merged_state_for_save(model)
model.save_pretrained(MERGED_PATH, state_dict=merged_state)
del merged_state
tokenizer.save_pretrained(MERGED_PATH)

print(f"✓ Merged model saved to {MERGED_PATH}")

del model
del base_model
torch.cuda.empty_cache()
gc.collect()

# COMMAND ----------

# DBTITLE 1,Validate Merged Model Locally
import xgrammar as xgr

print("Loading merged model for validation...")
val_model = AutoModelForCausalLM.from_pretrained(
    MERGED_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
val_tokenizer = AutoTokenizer.from_pretrained(MERGED_PATH)

# Build XGrammar compiler
vocab_size = getattr(
    val_model.config,
    "vocab_size",
    getattr(getattr(val_model.config, "text_config", val_model.config), "vocab_size", None),
)
tokenizer_info = xgr.TokenizerInfo.from_huggingface(val_tokenizer, vocab_size=vocab_size)
compiler = xgr.GrammarCompiler(tokenizer_info)

# Test with a medication extraction prompt
test_schema = MEDICATION_SCHEMA
compiled_grammar = compiler.compile_json_schema(json.dumps(test_schema))
logits_processor = xgr.contrib.hf.LogitsProcessor(compiled_grammar)

test_messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "Extract medication info: Prescribe Amoxicillin 500mg orally three times daily for bacterial sinusitis."},
]
prompt_text = val_tokenizer.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)
inputs = val_tokenizer(prompt_text, return_tensors="pt").to(val_model.device)

with torch.no_grad():
    outputs = val_model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.1,
        do_sample=True,
        logits_processor=[logits_processor],
        pad_token_id=val_tokenizer.eos_token_id,
    )

new_tokens = outputs[:, inputs["input_ids"].shape[1]:]
result_text = val_tokenizer.decode(new_tokens[0], skip_special_tokens=True)
parsed = json.loads(result_text)
print(json.dumps(parsed, indent=2))
print("\n✓ Local validation passed — merged model produces valid constrained JSON")

del val_model
torch.cuda.empty_cache()
gc.collect()

# COMMAND ----------

# DBTITLE 1,Log & Register Model with MLflow
import mlflow

mlflow.set_registry_uri("databricks-uc")

# Reuse the XGrammar serving module beside the registration notebook.
_notebook_path = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
)
_notebook_dir = _notebook_path.rsplit("/", 1)[0]
model_code_path = f"/Workspace{_notebook_dir}/xgrammar/model.py"
print(f"Model code path: {model_code_path}")

with mlflow.start_run(run_name="medgemma-qlora-finetuned") as run:
    mlflow.log_params({
        "base_model": HF_MODEL_NAME,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "max_seq_length": MAX_SEQ_LENGTH,
        "training_examples": len(training_examples),
    })

    model_info = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=model_code_path,
        artifacts={"model-weights": MERGED_PATH},
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
    version=model_version.version,
)

print(f"\n✓ Model registered: {FULL_MODEL_NAME} version {model_version.version}")

# COMMAND ----------

# DBTITLE 1,Create Serving Endpoint (best-effort)
import time
import requests as _requests

DATABRICKS_HOST = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
HEADERS = {"Authorization": f"Bearer {DATABRICKS_TOKEN}", "Content-Type": "application/json"}

ENDPOINT_READY = False

served_entity = {
    "entity_name": FULL_MODEL_NAME,
    "entity_version": str(model_version.version),
    "workload_type": WORKLOAD_TYPE,
    "workload_size": "Small",
    "scale_to_zero_enabled": SCALE_TO_ZERO,
}


def _fetch_serving_logs(ep_json):
    """Fetch served-model logs from all entities for diagnostics."""
    diag = [f"Endpoint state: {ep_json.get('state')}"]
    all_entities = []
    for cfg_key in ("config", "pending_config"):
        cfg = ep_json.get(cfg_key) or {}
        for k in ("served_models", "served_entities"):
            all_entities.extend(cfg.get(k, []))
    for sm in all_entities:
        names_to_try = []
        if sm.get("name"):
            names_to_try.append(sm["name"])
        entity = sm.get("entity_name", "").split(".")[-1]
        ver = sm.get("entity_version", "")
        if entity and ver:
            names_to_try.append(f"{entity}-{ver}")
        for sm_name in dict.fromkeys(names_to_try):
            log_resp = _requests.get(
                f"{DATABRICKS_HOST}/api/2.0/serving-endpoints/{ENDPOINT_NAME}/served-models/{sm_name}/logs",
                headers=HEADERS,
            )
            diag.append(f"\n--- logs/{sm_name} (status={log_resp.status_code}) ---")
            diag.append(log_resp.text[-4000:] if log_resp.ok else log_resp.text[:300])
    return "\n".join(diag)


try:
    create_new = False
    check = _requests.get(
        f"{DATABRICKS_HOST}/api/2.0/serving-endpoints/{ENDPOINT_NAME}",
        headers=HEADERS,
        timeout=120,
    )

    if check.status_code == 200:
        # If endpoint exists, check if it's in a failed state and needs cleanup
        ep_state = check.json().get("state", {})
        ready = ep_state.get("ready", "")
        update = ep_state.get("config_update", "")
        if "FAILED" in f"{ready} {update}":
            print(f"Endpoint in failed state: {ep_state} — deleting and recreating...")
            _requests.delete(
                f"{DATABRICKS_HOST}/api/2.0/serving-endpoints/{ENDPOINT_NAME}",
                headers=HEADERS,
            )
            time.sleep(10)
            create_new = True
        elif ready == "READY" and update == "NOT_UPDATING":
            # Submit config update
            print("Submitting config update...")
            resp = _requests.put(
                f"{DATABRICKS_HOST}/api/2.0/serving-endpoints/{ENDPOINT_NAME}/config",
                headers=HEADERS,
                json={"served_entities": [served_entity]},
                timeout=120,
            )
            resp.raise_for_status()
            print("✓ Config update submitted — waiting 60s for state transition...")
            time.sleep(60)
        else:
            # Endpoint exists but is in progress — wait for it to settle
            print(f"Endpoint in-progress: ready={ready}, update={update} — waiting...")
            settle_deadline = time.time() + 600
            while time.time() < settle_deadline:
                s = _requests.get(
                    f"{DATABRICKS_HOST}/api/2.0/serving-endpoints/{ENDPOINT_NAME}",
                    headers=HEADERS,
                ).json().get("state", {})
                if s.get("ready") == "READY" and s.get("config_update") == "NOT_UPDATING":
                    break
                if "FAILED" in (s.get("ready", "") + s.get("config_update", "")):
                    print(f"Endpoint failed while waiting — deleting...")
                    _requests.delete(
                        f"{DATABRICKS_HOST}/api/2.0/serving-endpoints/{ENDPOINT_NAME}",
                        headers=HEADERS,
                    )
                    time.sleep(10)
                    create_new = True
                    break
                time.sleep(30)
            if not create_new:
                resp = _requests.put(
                    f"{DATABRICKS_HOST}/api/2.0/serving-endpoints/{ENDPOINT_NAME}/config",
                    headers=HEADERS,
                    json={"served_entities": [served_entity]},
                    timeout=120,
                )
                resp.raise_for_status()
                print("✓ Config update submitted")
                time.sleep(60)
    else:
        create_new = True

    if create_new:
        print(f"Creating endpoint '{ENDPOINT_NAME}'...")
        resp = _requests.post(
            f"{DATABRICKS_HOST}/api/2.0/serving-endpoints",
            headers=HEADERS,
            json={"name": ENDPOINT_NAME, "config": {"served_entities": [served_entity]}},
            timeout=120,
        )
        resp.raise_for_status()
        print("✓ Endpoint creation submitted")
        time.sleep(30)

    # Poll until ready (up to 30 min)
    deadline = time.time() + 1800
    while time.time() < deadline:
        ep_resp = _requests.get(
            f"{DATABRICKS_HOST}/api/2.0/serving-endpoints/{ENDPOINT_NAME}",
            headers=HEADERS,
            timeout=120,
        )
        ep_resp.raise_for_status()
        ep_json = ep_resp.json()
        state = ep_json.get("state", {})
        ready_state = state.get("ready", "")
        update_state = state.get("config_update", "")
        print(f"  state: ready={ready_state}, update={update_state}")
        if ready_state == "READY" and update_state == "NOT_UPDATING":
            ENDPOINT_READY = True
            break
        if "FAILED" in f"{ready_state} {update_state}":
            print(f"\n⚠ Endpoint deployment failed. Fetching logs...")
            print(_fetch_serving_logs(ep_json))
            print("\n  The model is registered and can be deployed manually.")
            break
        time.sleep(60)
    else:
        print(f"⚠ Endpoint not ready within 30 minutes — model is registered for manual deploy")
except Exception as e:
    print(f"⚠ Endpoint deployment error: {e}")
    print("  The model is registered and can be deployed manually.")

if ENDPOINT_READY:
    print(f"✓ Endpoint ready: {ENDPOINT_NAME}")

# COMMAND ----------

# DBTITLE 1,Query Endpoint (if ready)
if ENDPOINT_READY:
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

    w = WorkspaceClient()
    resp = w.serving_endpoints.query(
        name=ENDPOINT_NAME,
        messages=[ChatMessage(role=ChatMessageRole.USER, content="Hi, what model are you?")],
    )
    print(resp.choices[0].message.content)

    from openai import OpenAI

    endpoint_client = OpenAI(
        api_key=DATABRICKS_TOKEN,
        base_url=f"{DATABRICKS_HOST}/serving-endpoints",
    )

    test_prompt = (
        "Discharge medication reconciliation: Start metoprolol succinate 25 mg tablet "
        "by mouth once daily for heart-rate control. Continue aspirin 81 mg daily. "
        "Hold metformin 500 mg twice daily for 48 hours after contrast. "
        "Use albuterol inhaler 2 puffs every 4 hours as needed for wheeze."
    )

    response = endpoint_client.chat.completions.create(
        model=ENDPOINT_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": test_prompt},
        ],
        max_tokens=512,
        temperature=0.0,
        extra_body={
            "custom_inputs": {"schema": json.dumps(MEDICATION_SCHEMA)},
        },
    )

    content = response.choices[0].message.content
    parsed = json.loads(content)
    print(json.dumps(parsed, indent=2))
    print(f"\n✓ Structured endpoint validation passed ({len(parsed.get('medications', []))} medications extracted)")
else:
    print("Endpoint not ready — skipping remote validation.")
    print(f"Deploy manually: databricks bundle run medgemma_deploy --params model_version={model_version.version}")

# COMMAND ----------

# DBTITLE 1,Summary
print("=" * 60)
print("  MedGemma QLoRA Fine-Tuning - Summary")
print("=" * 60)
print()
print(f"  Base model:    {HF_MODEL_NAME}")
print(f"  LoRA:          rank={LORA_RANK}, alpha={LORA_ALPHA}")
print(f"  Training:      {len(training_examples)} examples, {NUM_EPOCHS} epochs")
print(f"  Registered:    {FULL_MODEL_NAME} v{model_version.version}")
print(f"  Endpoint:      {ENDPOINT_NAME} ({'READY' if ENDPOINT_READY else 'NOT DEPLOYED'})")
print(f"  Workload:      {WORKLOAD_TYPE}")
print()
print("  Validations:")
print("    ✓ Local merged model inference (XGrammar constrained)")
if ENDPOINT_READY:
    print("    ✓ Endpoint query (Databricks SDK + OpenAI SDK)")
    print("    ✓ Structured extraction (medication schema)")
else:
    print("    - Endpoint validation skipped (deploy manually)")
print()
print("  Next steps:")
print(f"    - Deploy:  use existing medgemma_deploy job with registered_model_name={REGISTERED_MODEL_NAME}")
print(f"    - Compare: use clinical-extractor app UI to compare base vs fine-tuned")
print(f"    - Benchmark: run medgemma_benchmark with endpoint_name={ENDPOINT_NAME}")
print()
print("  Cleanup:")
print(f"    - Delete endpoint: w.serving_endpoints.delete('{ENDPOINT_NAME}')")
print(f"    - Delete model:    client.delete_registered_model('{FULL_MODEL_NAME}')")
print(f"    - Clean work dir:  rm -rf {pwd}")
print()
print("=" * 60)

if not ENDPOINT_READY:
    raise RuntimeError(f"Model registered, but endpoint {ENDPOINT_NAME} was not deployed successfully")
from jsonschema import validate

validate(instance=parsed, schema=MEDICATION_SCHEMA)
dbutils.notebook.exit(json.dumps({
    "registered_model": FULL_MODEL_NAME,
    "model_version": str(model_version.version),
    "endpoint_name": ENDPOINT_NAME,
    "training_examples": len(training_examples),
    "training_loss": train_result.training_loss,
    "structured_smoke_test": parsed,
}))
