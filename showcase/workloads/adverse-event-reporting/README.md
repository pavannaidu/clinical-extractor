# Adverse Event Reporting

## Use Case

Extract FDA MedWatch-style fields from patient safety narratives: suspect drug, adverse event, severity, timing, and outcome.

## Why It Matters

This is a strong example of why unstructured outputs are not sufficient. Downstream safety workflows need consistent, typed records.

## Recommended Demo Path

- MedGemma + XGrammar for domain-aware extraction
- PEFT + XGrammar if the customer has internal pharmacovigilance terminology

## Files

- `sample-note.txt`
- `schema.json`
- `system-prompt.txt`
- `expected-fields.md`

## Execution Layer

- Domain model path: `src/medgemma`
- Fine-tuning path: `src/finetune`
