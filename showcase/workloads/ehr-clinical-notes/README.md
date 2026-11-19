# EHR / Clinical Notes

## Use Case

Extract diagnoses, medications, dosages, vitals, and follow-up actions from unstructured clinical notes.

## Why It Matters

This is the most direct way to show the structured-output gap: many customers already have note-heavy workflows and downstream systems that want typed records instead of free text.

## Recommended Demo Path

- FMAPI first for a low-complexity baseline
- MedGemma + XGrammar for a medical-model comparison against reviewed expected values
- Batch path for scaling across large note volumes

## Files

- `sample-note.txt`
- `schema.json`
- `system-prompt.txt`
- `expected-fields.md`

## Execution Layer

- App comparison cockpit: `src/app`
- Domain model path: `src/models/medgemma/xgrammar`
- Batch path: `src/models/medgemma/batch`
