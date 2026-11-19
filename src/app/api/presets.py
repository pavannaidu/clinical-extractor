import json
from copy import deepcopy
from typing import Any


def _field(field_type: str, *, nullable: bool = False, enum: list[Any] | None = None) -> dict[str, Any]:
    """Build a sparse built-in field; missing values are omitted, not returned as null."""
    schema: dict[str, Any] = {
        "type": field_type,
    }
    if enum is not None:
        schema["enum"] = enum
    return schema


def _string(*, nullable: bool = False, enum: list[str] | None = None) -> dict[str, Any]:
    return _field("string", nullable=nullable, enum=enum)


def _integer(*, nullable: bool = False) -> dict[str, Any]:
    return _field("integer", nullable=nullable)


def _number(*, nullable: bool = False) -> dict[str, Any]:
    return _field("number", nullable=nullable)


def _boolean(*, nullable: bool = False) -> dict[str, Any]:
    return _field("boolean", nullable=nullable)


def _object(properties: dict[str, Any], *, nullable: bool = False) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
    }


def _array(items: dict[str, Any], *, nullable: bool = False) -> dict[str, Any]:
    return {
        "type": "array",
        "items": items,
    }


_DESCRIPTION_OVERRIDES = {
    "age": "Patient age.",
    "age_years": "Age in years.",
    "allergies": "Allergy entries.",
    "analytes": "Analyte results.",
    "assertion": "Mention assertion.",
    "assessment": "Clinical assessments.",
    "bmi": "Body mass index.",
    "body_site": "Anatomical location.",
    "changed": "Changed medications.",
    "certainty": "Clinical certainty.",
    "collected_at": "Collection time.",
    "continued": "Continued medications.",
    "critical_result": "Whether the result is critical.",
    "diagnoses": "Diagnosis entries.",
    "diseases": "Disease mentions.",
    "dose_regimen": "Reported dose regimen.",
    "ecog_status": "ECOG performance status.",
    "entities": "Clinical mentions.",
    "entity_type": "Clinical entity category.",
    "exclusion_flags": "Met exclusion criteria.",
    "experiencer": "Person experiencing the finding.",
    "fio2_percent": "Inspired oxygen percentage.",
    "flow_rate_lpm": "Oxygen flow rate.",
    "follow_up": "Follow-up plan.",
    "history_of_present_illness": "Current illness history.",
    "icd_code": "Reported ICD code.",
    "inclusion_flags": "Met inclusion criteria.",
    "key_labs": "Key lab results.",
    "laterality": "Anatomical side.",
    "measurements": "Reported measurements.",
    "medications": "Medication entries.",
    "onset_timing": "Event onset timing.",
    "oxygen_saturation_percent": "Oxygen saturation percentage.",
    "present_on_admission": "Present on admission status.",
    "prior_conservative_therapy": "Prior conservative treatment.",
    "prior_therapy_summary": "Prior therapy summary.",
    "procedures": "Procedure entries.",
    "prn": "Whether use is as needed.",
    "prn_reason": "As-needed use reason.",
    "rechallenge_performed": "Whether rechallenge occurred.",
    "reference_high": "Reference upper bound.",
    "reference_low": "Reference lower bound.",
    "reference_range": "Reference interval.",
    "result_time": "Result time.",
    "results": "Lab results.",
    "source_text": "Verbatim supporting text.",
    "spo2": "Oxygen saturation.",
    "study_date": "Study date.",
    "studies": "Imaging studies.",
    "started": "Started medications.",
    "stopped": "Discontinued medications.",
    "temperature_f": "Temperature in Fahrenheit.",
    "temporality": "Mention timing context.",
    "timestamp": "Observation time.",
    "value_text": "Reported nonnumeric value.",
    "vitals": "Vital measurements.",
}

_ARRAY_ITEM_DESCRIPTIONS = {
    "allergies": "One allergy.",
    "analytes": "One analyte result.",
    "assessment": "One assessment.",
    "changed": "One changed medication.",
    "continued": "One continued medication.",
    "diagnoses": "One diagnosis.",
    "diseases": "One disease mention.",
    "entities": "One clinical mention.",
    "exclusion_flags": "One exclusion criterion.",
    "findings": "One imaging finding.",
    "follow_up_appointments": "One follow-up appointment.",
    "inclusion_flags": "One inclusion criterion.",
    "key_labs": "One key lab result.",
    "measurements": "One measurement.",
    "medications": "One medication.",
    "past_medical_history": "One history item.",
    "pending_tests": "One pending test.",
    "physical_exam": "One exam finding.",
    "procedures": "One procedure.",
    "reaction": "One reaction.",
    "results": "One lab result.",
    "return_precautions": "One return precaution.",
    "review_of_systems": "One review finding.",
    "started": "One started medication.",
    "stopped": "One stopped medication.",
    "studies": "One imaging study.",
}


def _short_description(field_name: str) -> str:
    override = _DESCRIPTION_OVERRIDES.get(field_name)
    if override:
        return override
    return f"Reported {field_name.replace('_', ' ')}."


def _add_descriptions(schema: dict[str, Any], root_description: str) -> dict[str, Any]:
    described = deepcopy(schema)
    described["description"] = root_description

    def visit(node: dict[str, Any], field_name: str | None = None) -> None:
        if field_name is not None:
            node.setdefault("description", _short_description(field_name))

        properties = node.get("properties")
        if isinstance(properties, dict):
            for child_name, child in properties.items():
                if isinstance(child, dict):
                    visit(child, child_name)

        items = node.get("items")
        if isinstance(items, dict):
            if field_name is not None:
                items.setdefault(
                    "description",
                    _ARRAY_ITEM_DESCRIPTIONS.get(field_name, f"One {field_name.replace('_', ' ')} item."),
                )
            visit(items)

    visit(described)
    return described


def _preset(
    preset_id: str,
    name: str,
    description: str,
    schema: dict[str, Any],
    example_prompt: str,
    system_prompt: str,
) -> dict[str, str]:
    described_schema = _add_descriptions(schema, description)
    return {
        "id": preset_id,
        "name": name,
        "description": description,
        "schema_json": json.dumps(described_schema, indent=2),
        "example_prompt": example_prompt,
        "system_prompt": system_prompt,
    }


def _system_prompt(workflow: str, guidance: str) -> str:
    """Build focused instructions that remain portable across serving runtimes."""
    return (
        f"You are extracting {workflow} from clinical text. "
        "Use only information explicitly supported by the source text and return one valid JSON "
        "object that matches the requested schema exactly. "
        f"{guidance} "
        "Do not invent facts or resolve ambiguity silently. Omit fields that are not explicitly "
        "supported by the source text; do not use null or empty placeholders for missing information."
    )


MEDICATION_ITEM = _object(
    {
        "name": _string(),
        "strength": _string(nullable=True),
        "dosage": _string(nullable=True),
        "form": _string(nullable=True),
        "route": _string(nullable=True),
        "frequency": _string(nullable=True),
        "duration": _string(nullable=True),
        "start_date": _string(nullable=True),
        "end_date": _string(nullable=True),
        "status": _string(nullable=True, enum=["active", "continued", "started", "held", "stopped", "changed"]),
        "prn": _boolean(nullable=True),
        "prn_reason": _string(nullable=True),
        "indication": _string(nullable=True),
    }
)

DIAGNOSIS_ITEM = _object(
    {
        "condition": _string(),
        "icd_code": _string(nullable=True),
        "certainty": _string(nullable=True, enum=["confirmed", "suspected", "ruled_out"]),
        "severity": _string(nullable=True),
        "onset": _string(nullable=True),
        "status": _string(nullable=True),
        "present_on_admission": _string(nullable=True, enum=["yes", "no", "unknown"]),
        "notes": _string(nullable=True),
        "source_text": _string(nullable=True),
    }
)

BP_COMPONENT = _object(
    {
        "systolic": _integer(nullable=True),
        "diastolic": _integer(nullable=True),
        "unit": _string(nullable=True),
    },
    nullable=True,
)

MEASURED_COMPONENT_INT = _object(
    {
        "value": _integer(nullable=True),
        "unit": _string(nullable=True),
    },
    nullable=True,
)

MEASURED_COMPONENT_NUM = _object(
    {
        "value": _number(nullable=True),
        "unit": _string(nullable=True),
    },
    nullable=True,
)

OXYGEN_SUPPORT = _object(
    {
        "device": _string(nullable=True),
        "flow_rate_lpm": _number(nullable=True),
        "fio2_percent": _number(nullable=True),
    },
    nullable=True,
)

PHYSICAL_EXAM_ITEM = _object(
    {
        "system": _string(),
        "finding": _string(),
        "severity": _string(nullable=True),
    }
)

ASSESSMENT_ITEM = _object(
    {
        "diagnosis": _string(),
        "rationale": _string(nullable=True),
        "priority": _string(nullable=True, enum=["primary", "secondary", "rule_out", "monitor"]),
    }
)

LAB_RESULT_ITEM = _object(
    {
        "panel_name": _string(),
        "category": _string(nullable=True),
        "test_name": _string(),
        "value": _number(nullable=True),
        "value_text": _string(nullable=True),
        "unit": _string(nullable=True),
        "reference_low": _number(nullable=True),
        "reference_high": _number(nullable=True),
        "reference_range": _string(nullable=True),
        "flag": _string(nullable=True, enum=["normal", "high", "low", "critical", "abnormal", "pending"]),
        "interpretation": _string(nullable=True),
    }
)

ALLERGY_ITEM = _object(
    {
        "allergen": _string(),
        "category": _string(nullable=True, enum=["drug", "food", "environment", "contrast", "latex", "other"]),
        "reaction": _array(_string(), nullable=True),
        "severity": _string(nullable=True, enum=["mild", "moderate", "severe", "anaphylaxis", "unknown"]),
        "status": _string(nullable=True, enum=["active", "historical", "entered-in-error"]),
        "onset": _string(nullable=True),
        "certainty": _string(nullable=True, enum=["confirmed", "reported", "suspected"]),
    }
)

PROCEDURE_ITEM = _object(
    {
        "procedure_name": _string(),
        "body_site": _string(nullable=True),
        "laterality": _string(nullable=True, enum=["left", "right", "bilateral", "midline", "unknown"]),
        "procedure_date": _string(nullable=True),
        "status": _string(nullable=True, enum=["completed", "planned", "cancelled", "in_progress"]),
        "indication": _string(nullable=True),
        "outcome": _string(nullable=True),
        "device_used": _string(nullable=True),
        "source_text": _string(nullable=True),
    }
)

DEVICE_METADATA = _object(
    {
        "device_type": _string(nullable=True),
        "portable": _boolean(nullable=True),
        "contrast_used": _string(nullable=True),
    },
    nullable=True,
)

IMAGING_STUDY_ITEM = _object(
    {
        "modality": _string(),
        "body_part": _string(),
        "laterality": _string(nullable=True, enum=["left", "right", "bilateral", "none"]),
        "acquisition_context": _string(nullable=True),
        "findings": _array(_string(), nullable=True),
        "impression": _string(),
        "critical_result": _boolean(nullable=True),
        "recommendation": _string(nullable=True),
        "device_metadata": DEVICE_METADATA,
        "source_text": _string(nullable=True),
    }
)

MED_RECON_ITEM = _object(
    {
        "name": _string(),
        "prior_regimen": _string(nullable=True),
        "current_regimen": _string(nullable=True),
        "route": _string(nullable=True),
        "frequency": _string(nullable=True),
        "rationale": _string(nullable=True),
    }
)

DISCHARGE_MEDICATION = _object(
    {
        "name": _string(),
        "instructions": _string(nullable=True),
        "duration": _string(nullable=True),
    }
)

FOLLOW_UP_APPOINTMENT = _object(
    {
        "service": _string(),
        "timeframe": _string(nullable=True),
        "reason": _string(nullable=True),
    }
)

VET_ANALYTE_ITEM = _object(
    {
        "test_name": _string(),
        "value": _number(nullable=True),
        "value_text": _string(nullable=True),
        "unit": _string(nullable=True),
        "reference_range": _string(nullable=True),
        "flag": _string(nullable=True, enum=["normal", "high", "low", "critical", "borderline"]),
        "interpretation": _string(nullable=True),
    }
)

STUDY_METADATA = _object(
    {
        "modality": _string(nullable=True),
        "body_part": _string(nullable=True),
        "acquisition_context": _string(nullable=True),
        "device_type": _string(nullable=True),
        "portable": _boolean(nullable=True),
        "study_date": _string(nullable=True),
    },
    nullable=True,
)

IMAGING_MEASUREMENT = _object(
    {
        "name": _string(),
        "value": _number(nullable=True),
        "unit": _string(nullable=True),
    }
)


PRESETS = [
    _preset(
        "medication_extraction",
        "Medication Extraction",
        "Extract detailed medication orders and therapy context from clinical text.",
        _object(
            {
                "medications": _array(MEDICATION_ITEM),
            }
        ),
        "Discharge medication reconciliation after coronary CT angiography: Start metoprolol succinate 25 mg tablet by mouth once daily for heart-rate control. Continue aspirin 81 mg chewable tablet daily. Hold metformin 500 mg twice daily for 48 hours after contrast exposure. Use albuterol inhaler 2 puffs every 4 hours as needed for wheeze.",
        _system_prompt(
            "medication orders and therapy details",
            "Preserve the medication name, dose, form, route, frequency, duration, status, PRN context, and indication as stated; do not normalize or calculate doses.",
        ),
    ),
    _preset(
        "diagnosis_extraction",
        "Diagnosis Extraction",
        "Extract diagnosis lists with certainty, severity, and supporting context.",
        _object(
            {
                "diagnoses": _array(DIAGNOSIS_ITEM),
            }
        ),
        "Impression from advanced imaging workup: Acute right lower lobe pneumonia, confirmed, moderate severity. Suspected small parapneumonic pleural effusion. Chronic systolic heart failure remains stable. Rule out pulmonary embolism after CTA showed no filling defect.",
        _system_prompt(
            "diagnoses and their clinical context",
            "Keep confirmed, suspected, and ruled-out diagnoses distinct, and quote concise supporting source text when requested by the schema.",
        ),
    ),
    _preset(
        "vitals_extraction",
        "Vitals Extraction",
        "Extract bedside vital signs, anthropometrics, oxygen support, and timing.",
        _object(
            {
                "vitals": _object(
                    {
                        "blood_pressure": BP_COMPONENT,
                        "heart_rate": MEASURED_COMPONENT_INT,
                        "respiratory_rate": MEASURED_COMPONENT_INT,
                        "spo2": MEASURED_COMPONENT_INT,
                        "temperature": MEASURED_COMPONENT_NUM,
                        "weight": MEASURED_COMPONENT_NUM,
                        "height": MEASURED_COMPONENT_NUM,
                        "bmi": MEASURED_COMPONENT_NUM,
                        "pain_score": _integer(nullable=True),
                        "oxygen_support": OXYGEN_SUPPORT,
                    }
                ),
                "timestamp": _string(nullable=True),
            }
        ),
        "Bedside monitor integration at 08:30: BP 142/88 mmHg, HR 78 bpm, RR 16/min, SpO2 97% on 2 L nasal cannula, temp 98.6 F oral, weight 82 kg, height 178 cm, BMI 25.9, pain score 3/10.",
        _system_prompt(
            "vital signs, measurements, and oxygen support",
            "Keep each observed value with its stated unit and timestamp; do not convert units or infer unreported measurements.",
        ),
    ),
    _preset(
        "clinical_summary",
        "Clinical Summary",
        "Extract a structured encounter summary with findings, assessment, and plan.",
        _object(
            {
                "chief_complaint": _string(),
                "patient_demographics": _object(
                    {
                        "age": _integer(nullable=True),
                        "sex": _string(nullable=True),
                    },
                    nullable=True,
                ),
                "history_of_present_illness": _string(nullable=True),
                "past_medical_history": _array(_string(), nullable=True),
                "review_of_systems": _array(_string(), nullable=True),
                "physical_exam": _array(PHYSICAL_EXAM_ITEM, nullable=True),
                "assessment": _array(ASSESSMENT_ITEM, nullable=True),
                "plan": _array(_string()),
                "follow_up": _array(_string(), nullable=True),
            }
        ),
        "48-year-old female seen after emergency department evaluation for dyspnea. HPI: worsening shortness of breath for 2 days with orthopnea. PMH includes asthma and hypertension. ROS positive for cough and fatigue, negative for chest pain. Exam: diminished breath sounds at bases, mild bilateral edema. Assessment: community-acquired pneumonia and possible volume overload. Plan: start ceftriaxone and azithromycin, obtain transthoracic echo, continue diuresis, follow up with pulmonary clinic in 1 week.",
        _system_prompt(
            "a structured clinical encounter summary",
            "Separate history, observed findings, assessment, plan, and follow-up; preserve uncertainty and avoid adding clinical recommendations.",
        ),
    ),
    _preset(
        "lab_results",
        "Lab Results",
        "Extract structured lab panels, analyzer context, and flagged analytes.",
        _object(
            {
                "specimen_metadata": _object(
                    {
                        "specimen_type": _string(nullable=True),
                        "collected_at": _string(nullable=True),
                        "result_time": _string(nullable=True),
                        "analyzer_name": _string(nullable=True),
                    },
                    nullable=True,
                ),
                "results": _array(LAB_RESULT_ITEM),
                "overall_interpretation": _string(nullable=True),
            }
        ),
        "Chemistry analyzer report, serum collected 2026-03-19 07:40, resulted 08:05. BMP panel: Sodium 134 mmol/L low, Potassium 4.2 mmol/L normal, Creatinine 1.6 mg/dL high, Glucose 186 mg/dL high. Cardiac panel: Troponin I 0.01 ng/mL normal. Interpretation: mild AKI with hyperglycemia.",
        _system_prompt(
            "laboratory panels and analyte results",
            "Preserve specimen and analyzer context, record the source panel on every analyte, and capture numeric values, units, reference ranges, flags, and the overall interpretation exactly as reported; never derive a missing flag.",
        ),
    ),
    _preset(
        "allergy_extraction",
        "Allergy Extraction",
        "Extract allergy and intolerance details, including reactions and certainty.",
        _object(
            {
                "allergies": _array(ALLERGY_ITEM),
            }
        ),
        "Allergies reviewed prior to contrast-enhanced MRI: iodinated contrast causes hives and throat tightness, severity severe, documented by prior emergency visit. Penicillin listed as rash in childhood, reported history only. Latex causes localized itching from adhesive leads.",
        _system_prompt(
            "allergies and intolerances",
            "Distinguish allergen, reaction, severity, status, onset, and certainty, and do not upgrade a reported intolerance to a confirmed allergy.",
        ),
    ),
    _preset(
        "procedure_extraction",
        "Procedure Extraction",
        "Extract procedures, device usage, and outcome details from operative or procedural notes.",
        _object(
            {
                "procedures": _array(PROCEDURE_ITEM),
            }
        ),
        "Procedure note: Ultrasound-guided right thoracentesis completed on 2026-03-18 for symptomatic pleural effusion. 900 mL straw-colored fluid removed. No immediate complication. Portable ultrasound probe used for localization. Follow-up chest radiograph showed no pneumothorax.",
        _system_prompt(
            "procedures and procedural outcomes",
            "Capture procedure, body site, laterality, date, status, indication, device use, and outcome without inferring steps that are not documented; include concise source evidence when requested.",
        ),
    ),
    _preset(
        "imaging_findings",
        "Imaging Findings",
        "Extract imaging studies, findings, impression, and recommended follow-up.",
        _object(
            {
                "studies": _array(IMAGING_STUDY_ITEM),
            }
        ),
        "Portable AP chest radiograph obtained in the ICU using a bedside digital system. Findings: bibasilar opacities and small bilateral pleural effusions. Impression: worsening pulmonary edema. Critical result: no. Recommendation: correlate with fluid balance and repeat film after diuresis if respiratory status worsens.",
        _system_prompt(
            "imaging findings and impressions",
            "Keep modality and acquisition context separate from findings, impression, critical-result status, and recommendation; do not add a diagnosis not stated by the reader.",
        ),
    ),
    _preset(
        "medication_reconciliation",
        "Medication Reconciliation",
        "Extract continued, started, stopped, and changed medications across transitions of care.",
        _object(
            {
                "continued": _array(MED_RECON_ITEM),
                "started": _array(MED_RECON_ITEM),
                "stopped": _array(MED_RECON_ITEM),
                "changed": _array(MED_RECON_ITEM),
            }
        ),
        "Medication reconciliation after observation stay: Continue atorvastatin 40 mg nightly and lisinopril 10 mg daily. Start furosemide 20 mg daily for 5 days for volume management. Stop ibuprofen because of acute kidney injury. Change insulin glargine from 18 units nightly to 22 units nightly due to persistent hyperglycemia.",
        _system_prompt(
            "medication reconciliation across a transition of care",
            "Place every medication in exactly one continued, started, stopped, or changed collection and preserve prior/current regimens and the documented rationale.",
        ),
    ),
    _preset(
        "discharge_follow_up_plan",
        "Discharge / Follow-up Plan",
        "Extract disposition, discharge medications, follow-up, pending tests, and patient instructions.",
        _object(
            {
                "disposition": _string(),
                "diagnoses": _array(_string()),
                "medications": _array(DISCHARGE_MEDICATION),
                "follow_up_appointments": _array(FOLLOW_UP_APPOINTMENT),
                "pending_tests": _array(_string(), nullable=True),
                "return_precautions": _array(_string()),
                "patient_instructions": _array(_string()),
            }
        ),
        "Discharge after CT pulmonary angiography: Disposition home with spouse. Diagnoses include multifocal pneumonia and chronic asthma. Medications: doxycycline 100 mg twice daily for 7 days, continue inhaled corticosteroid, albuterol as needed. Follow-up with primary care in 3 days and pulmonology in 2 weeks. Pending blood cultures. Return for worsening shortness of breath, chest pain, or persistent fever. Encourage hydration and incentive spirometry.",
        _system_prompt(
            "a discharge and follow-up plan",
            "Separate disposition, diagnoses, medication instructions, appointments, pending tests, return precautions, and patient instructions; do not generate new advice.",
        ),
    ),
    _preset(
        "veterinary_lab_panel",
        "Veterinary Lab Panel",
        "Extract structured veterinary diagnostics from analyzer and pathology narratives.",
        _object(
            {
                "patient_name": _string(nullable=True),
                "species": _string(),
                "breed": _string(nullable=True),
                "sex": _string(nullable=True),
                "age_years": _number(nullable=True),
                "specimen_metadata": _object(
                    {
                        "specimen_type": _string(nullable=True),
                        "collected_at": _string(nullable=True),
                        "analyzer_name": _string(nullable=True),
                        "panel_name": _string(nullable=True),
                    },
                    nullable=True,
                ),
                "analytes": _array(VET_ANALYTE_ITEM),
                "overall_interpretation": _string(nullable=True),
                "recommendation": _string(nullable=True),
            }
        ),
        "Canine chemistry and CBC summary for Bella, 8-year-old spayed female Labrador retriever. Serum collected from a small-animal in-clinic analyzer. Panel: ALT 162 U/L high, ALP 410 U/L high, BUN 34 mg/dL borderline high, creatinine 1.3 mg/dL normal. CBC: neutrophils 15.2 K/uL high, platelets adequate. Interpretation: inflammatory leukogram with hepatocellular enzyme elevation. Recommendation: abdominal ultrasound and repeat chemistry in 2 weeks.",
        _system_prompt(
            "veterinary patient and diagnostic results",
            "Preserve species-specific patient context, analyzer metadata, analyte values, units, flags, interpretation, and recommendation without applying human reference ranges.",
        ),
    ),
    _preset(
        "imaging_device_report",
        "Imaging Device Report",
        "Extract structured findings from imaging or monitoring device reports.",
        _object(
            {
                "study_metadata": STUDY_METADATA,
                "findings": _array(_string()),
                "impression": _string(),
                "measurements": _array(IMAGING_MEASUREMENT, nullable=True),
                "critical_result": _boolean(nullable=True),
                "recommendation": _string(nullable=True),
            }
        ),
        "Transthoracic echocardiogram acquired on a cart-based ultrasound system. Findings: mildly dilated left ventricle with global hypokinesis, trace mitral regurgitation, no pericardial effusion. Measurements: LVEF 35 percent, LVEDD 6.1 cm. Impression: reduced systolic function consistent with cardiomyopathy. Recommendation: optimize guideline-directed medical therapy and repeat echo in 3 months.",
        _system_prompt(
            "imaging-device study metadata and report content",
            "Separate device/acquisition metadata, findings, measurements, impression, critical-result status, and recommendation; preserve reported measurement units.",
        ),
    ),
    _preset(
        "adverse_event_reporting",
        "Adverse Event Reporting",
        "Capture a safety narrative for pharmacovigilance review, including treatment and outcome.",
        _object(
            {
                "suspect_drug": _string(nullable=True),
                "dose_regimen": _string(nullable=True),
                "indication": _string(nullable=True),
                "adverse_event": _string(nullable=True),
                "severity": _string(nullable=True),
                "onset_timing": _string(nullable=True),
                "action_taken": _string(nullable=True),
                "outcome": _string(nullable=True),
                "rechallenge_performed": _boolean(nullable=True),
            }
        ),
        "Safety narrative: Patient started trimethoprim-sulfamethoxazole 800/160 mg twice daily for urinary tract infection on 2026-04-02. On day 3 the patient developed a diffuse rash, lip swelling, and shortness of breath requiring emergency department evaluation. Event severity was severe. Drug was discontinued and the patient was treated with antihistamines and corticosteroids. Outcome: symptoms improved after treatment. No rechallenge was performed.",
        _system_prompt(
            "adverse-event reporting fields for pharmacovigilance",
            "Keep the suspect product, regimen, indication, event, onset, action taken, outcome, and rechallenge status grounded in the safety narrative; do not make a causality determination.",
        ),
    ),
    _preset(
        "clinical_trial_screening",
        "Clinical Trial Screening",
        "Structure eligibility-relevant disease, therapy, laboratory, and screening evidence.",
        _object(
            {
                "disease": _string(nullable=True),
                "ecog_status": _integer(nullable=True),
                "prior_therapy_summary": _string(nullable=True),
                "key_labs": _array(
                    _object(
                        {
                            "name": _string(),
                            "value": _string(nullable=True),
                        }
                    )
                ),
                "inclusion_flags": _array(_string()),
                "exclusion_flags": _array(_string()),
                "screening_status": _string(nullable=True),
            }
        ),
        "Screening note for oncology trial enrollment. Patient is a 58-year-old male with metastatic non-small cell lung cancer. ECOG performance status 1. Prior platinum therapy completed 8 months ago. No active brain metastases on MRI. Labs show ANC 2.1, platelets 168, AST 31, ALT 28, creatinine clearance 72 mL/min. Exclusion review: no autoimmune disease requiring systemic therapy and no uncontrolled infection. Eligible for screening based on available criteria pending central pathology confirmation.",
        _system_prompt(
            "clinical-trial screening evidence",
            "Capture eligibility-relevant facts and stated screening status, preserve negative findings, and do not decide final protocol eligibility beyond the source note.",
        ),
    ),
    _preset(
        "drug_interaction_screening",
        "Drug Interaction Screening",
        "Extract the active regimen and interaction concerns documented for pharmacy review.",
        _object(
            {
                "medications": _array(
                    _object(
                        {
                            "name": _string(),
                            "dosage": _string(nullable=True),
                            "route": _string(nullable=True),
                            "frequency": _string(nullable=True),
                            "indication": _string(nullable=True),
                        }
                    )
                ),
                "interaction_concerns": _array(_string()),
            }
        ),
        "Medication review prior to discharge: warfarin 5 mg by mouth nightly for atrial fibrillation, trimethoprim-sulfamethoxazole 800/160 mg twice daily for urinary tract infection, and amiodarone 200 mg daily for rhythm control. Pharmacy note warns that TMP-SMX and amiodarone may increase warfarin effect and recommends close INR monitoring.",
        _system_prompt(
            "medication regimens for drug-interaction screening",
            "Capture each stated medication and only the interaction concerns or monitoring actions explicitly documented; do not generate additional interaction warnings.",
        ),
    ),
    _preset(
        "ehr_note_extraction",
        "EHR Note Extraction",
        "Extract a compact cross-domain EHR record with diagnoses, medications, vitals, and follow-up.",
        _object(
            {
                "diagnoses": _array(
                    _object(
                        {
                            "condition": _string(),
                            "status": _string(nullable=True),
                        }
                    )
                ),
                "medications": _array(
                    _object(
                        {
                            "name": _string(),
                            "dosage": _string(nullable=True),
                            "frequency": _string(nullable=True),
                        }
                    )
                ),
                "vitals": _object(
                    {
                        "blood_pressure": _string(nullable=True),
                        "heart_rate": _integer(nullable=True),
                        "temperature_f": _number(nullable=True),
                        "oxygen_saturation_percent": _integer(nullable=True),
                    }
                ),
                "follow_up_actions": _array(_string()),
            }
        ),
        "63-year-old female seen for pneumonia follow-up after a recent inpatient stay. Symptoms have improved. Oxygen saturation is 96 percent on room air. Blood pressure 128/76, heart rate 82, temperature 98.7 F. Active diagnoses include community-acquired pneumonia improving and type 2 diabetes mellitus. Current medications are amoxicillin-clavulanate 875/125 mg twice daily for 5 days remaining, metformin 1000 mg twice daily, and albuterol inhaler 2 puffs every 6 hours as needed. Allergy: lisinopril causes cough. Plan: finish antibiotics, continue hydration, repeat chest x-ray in 6 weeks, and follow up with primary care next week.",
        _system_prompt(
            "diagnoses, medications, vitals, and follow-up actions from an EHR note",
            "Keep each domain separate, retain diagnosis status and medication regimen wording, and include only follow-up actions actually ordered or recommended.",
        ),
    ),
    _preset(
        "prior_authorization",
        "Prior Authorization",
        "Structure a prior-authorization request with service, diagnosis, conservative therapy, and rationale.",
        _object(
            {
                "requested_service": _string(nullable=True),
                "diagnosis": _string(nullable=True),
                "diagnosis_code": _string(nullable=True),
                "prior_conservative_therapy": _array(_string()),
                "supporting_rationale": _string(nullable=True),
                "urgency": _string(nullable=True),
            }
        ),
        "Prior authorization request for lumbar spine MRI. Ordering diagnosis is lumbar radiculopathy with persistent left leg pain after 6 weeks of conservative therapy including physical therapy and NSAIDs. Prior X-ray was nondiagnostic. Requested service: MRI lumbar spine without contrast. Clinical rationale: evaluate for disc herniation before procedural planning. No red flag symptoms requiring emergency referral.",
        _system_prompt(
            "prior-authorization request fields",
            "Preserve the requested service, diagnosis, stated code, completed conservative therapy, supporting rationale, and urgency; leave unstated coding details null.",
        ),
    ),
    _preset(
        "disease_mention_extraction",
        "Disease Mention Extraction",
        "Extract verbatim disease mentions from biomedical research text (NCBI Disease Corpus format).",
        _object(
            {
                "diseases": _array(
                    _object(
                        {
                            "mention": _string(),
                            "category": _string(
                                enum=["disease", "syndrome", "complication", "comorbidity", "symptom", "risk_factor"]
                            ),
                            "negated": _boolean(nullable=True),
                            "body_system": _string(nullable=True),
                        }
                    )
                ),
            }
        ),
        "BACKGROUND: Sickle cell disease is an inherited hemoglobinopathy characterized by chronic hemolytic anemia, vaso-occlusive crises, and progressive end-organ damage including pulmonary hypertension, chronic kidney disease, and retinopathy. Hydroxyurea remains the mainstay of disease-modifying therapy, yet many patients continue to experience acute chest syndrome, stroke, and avascular necrosis of the femoral head. Recent advances in gene therapy targeting BCL11A have shown promise in transfusion-dependent beta-thalassemia and sickle cell anemia, but long-term safety data regarding myelodysplastic syndrome and secondary malignancy risk remain limited. METHODS: We conducted a multicenter retrospective cohort study of 2,847 patients with sickle cell disease across 14 centers to evaluate the incidence of silent cerebral infarcts, nephrotic syndrome, and leg ulcers in relation to hydroxyurea adherence. Patients with coexisting glucose-6-phosphate dehydrogenase deficiency were excluded due to confounding hemolysis risk. RESULTS: Over a median follow-up of 6.2 years, 34% of patients developed at least one vaso-occlusive crisis requiring hospitalization, 12% experienced acute chest syndrome, and 8% had radiographic evidence of silent cerebral infarcts. Pulmonary hypertension was identified in 18% of screened patients and was independently associated with chronic kidney disease (OR 2.4, 95% CI 1.8-3.1) and retinopathy (OR 1.9, 95% CI 1.4-2.6). Hydroxyurea adherence above 80% was associated with reduced incidence of stroke (HR 0.42, 95% CI 0.28-0.63) and acute chest syndrome (HR 0.51, 95% CI 0.38-0.69). CONCLUSIONS: Sickle cell disease confers substantial multi-organ morbidity. While hydroxyurea reduces vaso-occlusive complications, the burden of pulmonary hypertension, chronic kidney disease, and silent cerebral infarcts remains high. Emerging gene therapies may address the underlying hemoglobinopathy but require vigilance for myelodysplastic syndrome.",
        _system_prompt(
            "structured disease mentions",
            "For each mention provide the verbatim text, classify it as disease, syndrome, complication, comorbidity, symptom, or risk_factor, indicate whether it is negated in context, and note the affected body system when stated or clearly implied. Preserve order of first appearance and use only text from the source.",
        ),
    ),
    # Keep the legacy ID so Lakebase upserts the existing built-in instead of
    # leaving a stale SNOMED-mapping preset alongside the evidence-only version.
    _preset(
        "snomed_omop_mapping",
        "Clinical Concept Extraction",
        "Extract source-grounded clinical concept mentions with assertion, timing, and experiencer context.",
        _object(
            {
                "entities": _array(
                    _object(
                        {
                            "source_text": _string(),
                            "entity_type": _string(
                                enum=[
                                    "condition",
                                    "symptom_or_sign",
                                    "procedure",
                                    "medication",
                                    "test",
                                    "measurement",
                                    "observation",
                                ]
                            ),
                            "assertion": _string(
                                enum=["present", "absent", "possible", "conditional", "hypothetical"]
                            ),
                            "temporality": _string(
                                enum=["current", "historical", "planned", "future", "unknown"]
                            ),
                            "experiencer": _string(
                                enum=["patient", "family_member", "other", "unknown"]
                            ),
                            "body_site": _string(nullable=True),
                            "laterality": _string(
                                nullable=True,
                                enum=["left", "right", "bilateral", "midline"],
                            ),
                        }
                    )
                ),
            }
        ),
        "Emergency department note: A 67-year-old man reports 3 days of dyspnea, productive cough, and fever. He denies chest pain or hemoptysis. Chest CT shows right lower-lobe consolidation and small bilateral pleural effusions, with no pneumothorax. A prior echocardiogram from 2024 showed LVEF 35%. Assessment: community-acquired PNA, acute-on-chronic systolic heart failure, and new-onset atrial fibrillation. His mother had breast cancer; the patient has no known history of malignancy. Start ceftriaxone 1 g IV daily and furosemide 40 mg IV twice daily; continue apixaban 5 mg twice daily. Repeat echocardiogram tomorrow. Consider thoracentesis only if the pleural effusions worsen.",
        (
            "You extract clinical concept mentions explicitly supported by clinical source text and return "
            "one valid JSON object that matches the requested schema exactly. This is mention extraction, "
            "not terminology normalization or coding. For every clinically meaningful condition, symptom "
            "or sign, procedure, medication, test, measurement, or observation, copy source_text verbatim "
            "using the smallest sufficient contiguous span and preserve order of first appearance. Classify "
            "the mention's entity_type, assertion, temporality, and experiencer only from its local context. "
            "Absent means explicitly negated; possible means uncertain or suspected; conditional means it "
            "applies only under a stated condition; hypothetical means a risk or scenario that is not asserted "
            "to have occurred. Do not treat historical, planned, future, family-member, negated, uncertain, or "
            "conditional mentions as current patient facts. Record body_site and laterality only when the source "
            "supports them; otherwise use null. Do not infer a diagnosis from a medication, test, demographic, "
            "or general medical knowledge. Preserve abbreviations exactly in source_text; use surrounding text "
            "to classify one only when its meaning is unambiguous, and otherwise preserve the ambiguity. Do not "
            "generate, recall, infer, or output SNOMED CT, OMOP, ICD, LOINC, RxNorm, CPT, or other terminology "
            "identifiers or preferred terms. Terminology linking requires a separate, versioned vocabulary "
            "lookup and is intentionally outside this extraction task. Do not invent facts or resolve ambiguity "
            "silently. Use an empty entities array when the source contains no supported clinical mentions."
        ),
    ),
]
