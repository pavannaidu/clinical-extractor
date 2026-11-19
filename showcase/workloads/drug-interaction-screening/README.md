# Drug Interaction Screening

## Use Case

Extract medication regimen details needed for downstream interaction screening: drug, dosage, route, frequency, and indication context.

## Why It Matters

The extraction layer is often the prerequisite for an interaction engine. Clean structured medication records make the downstream safety workflow possible.

## Recommended Demo Path

- MedGemma + XGrammar for medication-focused clinical extraction
- FMAPI baseline when the use case is simple

## Files

- `sample-note.txt`
- `schema.json`
- `system-prompt.txt`
- `expected-fields.md`

## Execution Layer

- App comparison cockpit: `src/app`
- Domain model path: `src/models/medgemma/xgrammar`
