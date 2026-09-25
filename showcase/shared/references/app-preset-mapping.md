# App Preset Mapping

The app in `src/app` already contains preset schemas and examples that map cleanly to the showcase taxonomy.

## Current Presets That Align to Showcase Workloads

- **EHR / clinical notes**
  - `clinical_summary`
  - `diagnosis_extraction`
  - `medication_extraction`
  - `vitals_extraction`
  - `lab_results`
  - `allergy_extraction`
  - `procedure_extraction`
  - `imaging_findings`
  - `medication_reconciliation`
  - `discharge_follow_up_plan`

- **Adverse event reporting**
  - `adverse_event_reporting`
  - supporting focused presets: `medication_extraction`, `diagnosis_extraction`, `allergy_extraction`

- **Clinical trial data capture**
  - `clinical_trial_screening`
  - supporting focused presets: `clinical_summary`, `lab_results`, `procedure_extraction`

- **Prior authorization**
  - `prior_authorization`
  - supporting focused presets: `diagnosis_extraction`, `procedure_extraction`, `medication_extraction`

- **Drug interaction screening**
  - `drug_interaction_screening`
  - supporting focused presets: `medication_extraction`, `allergy_extraction`

- **Compact EHR note extraction**
  - `ehr_note_extraction`

## Guidance

Every built-in preset carries a workflow-specific system prompt in addition to its example text and strict JSON Schema. Selecting a preset in the app applies all three together; users can then adapt the system prompt before running a comparison. Saved custom presets preserve the current system prompt as well.
