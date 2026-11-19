# Databricks notebook source
# DBTITLE 1,Overview
# MAGIC %md
# MAGIC # MedGemma Batch Input Preparation
# MAGIC
# MAGIC Creates a synthetic batch input Delta table used by both UDF and SGC
# MAGIC inference notebooks so their benchmarks run against the same records.

# COMMAND ----------

# DBTITLE 1,Widgets
dbutils.widgets.removeAll()
dbutils.widgets.text("catalog", "main", "Catalog")
dbutils.widgets.text("schema", "mlops", "Schema")
dbutils.widgets.text("input_table", "medgemma_batch_input", "Input Delta Table")
dbutils.widgets.text("synthetic_replications", "50", "Synthetic input replications")

# COMMAND ----------

# DBTITLE 1,Configuration
import json
import pandas as pd

CATALOG = dbutils.widgets.get("catalog")
SCHEMA_NAME = dbutils.widgets.get("schema")
INPUT_TABLE = dbutils.widgets.get("input_table")
SYNTHETIC_REPLICATIONS = int(dbutils.widgets.get("synthetic_replications"))

INPUT_TABLE_FQN = f"{CATALOG}.{SCHEMA_NAME}.{INPUT_TABLE}"

print(f"Input table            : {INPUT_TABLE_FQN}")
print(f"Synthetic replications : {SYNTHETIC_REPLICATIONS}")

# COMMAND ----------

# DBTITLE 1,Define Schemas for Examples
medication_schema = {
    "type": "object",
    "properties": {
        "medication_name": {"type": "string"},
        "dosage": {"type": "string"},
        "frequency": {"type": "string"},
        "route": {"type": "string"},
        "indication": {"type": "string"},
    },
    "required": ["medication_name", "dosage", "frequency"],
}

condition_schema = {
    "type": "object",
    "properties": {
        "condition": {"type": "string"},
        "symptoms": {"type": "array", "items": {"type": "string"}},
        "severity": {"type": "string", "enum": ["mild", "moderate", "severe"]},
    },
    "required": ["condition", "symptoms"],
}

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

lab_schema = {
    "type": "object",
    "properties": {
        "test_name": {"type": "string"},
        "value": {"type": "string"},
        "unit": {"type": "string"},
        "reference_range": {"type": "string"},
        "flag": {"type": "string", "enum": ["normal", "low", "high", "critical"]},
    },
    "required": ["test_name", "value", "unit", "flag"],
}

allergy_schema = {
    "type": "object",
    "properties": {
        "allergen": {"type": "string"},
        "reaction_type": {"type": "string"},
        "severity": {"type": "string", "enum": ["mild", "moderate", "severe", "life-threatening"]},
        "onset": {"type": "string"},
        "documented_by": {"type": "string"},
    },
    "required": ["allergen", "reaction_type", "severity"],
}

procedure_schema = {
    "type": "object",
    "properties": {
        "procedure_name": {"type": "string"},
        "cpt_code": {"type": "string"},
        "body_site": {"type": "string"},
        "performing_provider": {"type": "string"},
        "complications": {"type": "array", "items": {"type": "string"}},
        "outcome": {"type": "string"},
    },
    "required": ["procedure_name", "body_site"],
}

discharge_schema = {
    "type": "object",
    "properties": {
        "admission_diagnosis": {"type": "string"},
        "discharge_diagnosis": {"type": "string"},
        "length_of_stay_days": {"type": "integer"},
        "discharge_condition": {
            "type": "string",
            "enum": ["stable", "improved", "unchanged", "declined", "expired"],
        },
        "follow_up_instructions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["admission_diagnosis", "discharge_diagnosis", "discharge_condition"],
}

# COMMAND ----------

# DBTITLE 1,Create Synthetic Input Rows
sample_rows = [
    {
        "id": "med_001",
        "category": "medication",
        "prompt": "Extract medication info: Prescribe Amoxicillin 500mg orally three times daily for 10 days to treat bacterial sinusitis.",
        "schema": json.dumps(medication_schema),
    },
    {
        "id": "med_002",
        "category": "medication",
        "prompt": "Extract medication info: Patient is taking Lisinopril 10mg once daily by mouth for management of essential hypertension.",
        "schema": json.dumps(medication_schema),
    },
    {
        "id": "med_003",
        "category": "medication",
        "prompt": "Extract medication info: Administer Metformin 1000mg twice daily with meals for glycaemic control in type 2 diabetes mellitus.",
        "schema": json.dumps(medication_schema),
    },
    {
        "id": "med_004",
        "category": "medication",
        "prompt": "Extract medication info: Start Atorvastatin 40mg by mouth at bedtime for hyperlipidaemia per lipid panel results.",
        "schema": json.dumps(medication_schema),
    },
    {
        "id": "cond_001",
        "category": "condition",
        "prompt": "Extract medical condition: Patient presents with moderate asthma exacerbation characterised by wheezing, shortness of breath, and use of accessory muscles.",
        "schema": json.dumps(condition_schema),
    },
    {
        "id": "cond_002",
        "category": "condition",
        "prompt": "Extract medical condition: Diagnosed with severe community-acquired pneumonia. Symptoms include high-grade fever, productive cough with purulent sputum, pleuritic chest pain, and oxygen saturation of 88% on room air.",
        "schema": json.dumps(condition_schema),
    },
    {
        "id": "cond_003",
        "category": "condition",
        "prompt": "Extract medical condition: Mild urinary tract infection presenting with dysuria, urinary frequency, and suprapubic discomfort. No systemic signs of infection.",
        "schema": json.dumps(condition_schema),
    },
    {
        "id": "cond_004",
        "category": "condition",
        "prompt": "Extract medical condition: Patient has moderate major depressive disorder with persistent low mood, anhedonia, fatigue, and insomnia for more than 8 weeks.",
        "schema": json.dumps(condition_schema),
    },
    {
        "id": "vit_001",
        "category": "vitals",
        "prompt": "Extract patient vitals: BP 142/88 mmHg, HR 96 bpm, Temp 101.3°F, RR 18 breaths/min, SpO2 97% on 2L nasal cannula.",
        "schema": json.dumps(vitals_schema),
    },
    {
        "id": "vit_002",
        "category": "vitals",
        "prompt": "Extract patient vitals: Blood pressure 118/74, heart rate 72, temperature 98.6 Fahrenheit, respiratory rate 14, oxygen saturation 99 percent on room air.",
        "schema": json.dumps(vitals_schema),
    },
    {
        "id": "vit_003",
        "category": "vitals",
        "prompt": "Extract patient vitals: Vital signs notable for hypotension at 88/52 mmHg, tachycardia 124 bpm, fever 103.1°F, RR 24, SpO2 91% on 6L O2 via face mask.",
        "schema": json.dumps(vitals_schema),
    },
    {
        "id": "lab_001",
        "category": "lab",
        "prompt": "Extract lab result: Serum potassium returned at 3.1 mEq/L (reference 3.5-5.0 mEq/L), flagged as low.",
        "schema": json.dumps(lab_schema),
    },
    {
        "id": "lab_002",
        "category": "lab",
        "prompt": "Extract lab result: HbA1c measured at 9.2% which is above the normal range of <5.7%, flagged as high per institutional guidelines.",
        "schema": json.dumps(lab_schema),
    },
    {
        "id": "lab_003",
        "category": "lab",
        "prompt": "Extract lab result: Troponin I critically elevated at 4.8 ng/mL, reference range <0.04 ng/mL, critical high value reported to provider.",
        "schema": json.dumps(lab_schema),
    },
    {
        "id": "alg_001",
        "category": "allergy",
        "prompt": "Extract allergy: Patient has a documented severe allergy to Penicillin resulting in anaphylaxis. Reaction first noted in 2018 and confirmed by allergist.",
        "schema": json.dumps(allergy_schema),
    },
    {
        "id": "alg_002",
        "category": "allergy",
        "prompt": "Extract allergy: Mild shellfish allergy causing urticaria and mild angioedema. Onset reported as childhood. No anaphylaxis on record.",
        "schema": json.dumps(allergy_schema),
    },
    {
        "id": "proc_001",
        "category": "procedure",
        "prompt": "Extract procedure details: Laparoscopic appendectomy performed on the right lower quadrant by Dr. Chen. No intraoperative complications. Patient tolerated procedure well and was transferred to recovery in stable condition.",
        "schema": json.dumps(procedure_schema),
    },
    {
        "id": "proc_002",
        "category": "procedure",
        "prompt": "Extract procedure details: Percutaneous coronary intervention (PCI) with drug-eluting stent placement in the LAD. Procedure performed by Dr. Patel. Complications included minor access-site haematoma. Outcome: successful revascularisation.",
        "schema": json.dumps(procedure_schema),
    },
    {
        "id": "disch_001",
        "category": "discharge",
        "prompt": "Extract discharge summary: Admitted for NSTEMI, discharged with diagnosis of acute coronary syndrome. Length of stay 4 days. Discharged in stable condition with instructions to follow up with cardiology in 1 week, continue dual antiplatelet therapy, and attend cardiac rehab.",
        "schema": json.dumps(discharge_schema),
    },
    {
        "id": "disch_002",
        "category": "discharge",
        "prompt": "Extract discharge summary: Patient admitted for diabetic ketoacidosis, discharged with diagnosis of DKA secondary to medication non-compliance. LOS 3 days. Condition improved at discharge. Follow-up with endocrinology in 2 weeks and diabetes education programme.",
        "schema": json.dumps(discharge_schema),
    },
]

replicated_rows = []
for rep in range(SYNTHETIC_REPLICATIONS):
    for row in sample_rows:
        new_row = dict(row)
        new_row["id"] = f"{row['id']}_{rep}"
        replicated_rows.append(new_row)

print(f"Creating input table: {INPUT_TABLE_FQN} ({len(replicated_rows)} rows)")
sdf = spark.createDataFrame(pd.DataFrame(replicated_rows))
(
    sdf.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(INPUT_TABLE_FQN)
)
print(f"Created {INPUT_TABLE_FQN} with {len(replicated_rows)} rows")

# COMMAND ----------

# DBTITLE 1,Preview Input Data
display(spark.table(INPUT_TABLE_FQN))
