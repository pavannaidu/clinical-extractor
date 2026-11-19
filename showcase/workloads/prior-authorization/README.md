# Prior Authorization

## Use Case

Extract diagnosis codes, procedure details, requested service, and approval rationale from payer-facing clinical summaries.

## Why It Matters

Prior authorization workflows depend on consistent structured outputs that can be routed into utilization management systems or human review queues.

## Recommended Demo Path

- FMAPI baseline for low-complexity proof of concept
- MedGemma + XGrammar for stronger clinical reasoning on procedure context

## Files

- `sample-note.txt`
- `schema.json`
- `system-prompt.txt`
- `expected-fields.md`

## Execution Layer

- App comparison cockpit: `src/app`
- Domain model path: `src/models/medgemma/xgrammar`
