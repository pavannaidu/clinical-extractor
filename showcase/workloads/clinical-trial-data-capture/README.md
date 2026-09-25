# Clinical Trial Data Capture

## Use Case

Extract structured trial screening or visit information such as inclusion and exclusion criteria flags, endpoints, and protocol-relevant observations.

## Why It Matters

Clinical trial workflows often require repeatable extraction against protocol-defined fields, which makes schema-driven constrained decoding especially useful.

## Recommended Demo Path

- MedGemma + XGrammar for protocol-aware extraction
- PEFT + XGrammar if trial-specific language or sponsor-specific terminology is important
- batch inference for retrospective note review

## Files

- `sample-note.txt`
- `schema.json`
- `system-prompt.txt`
- `expected-fields.md`

## Execution Layer

- Domain model path: `src/medgemma`
- Fine-tuning path: `src/finetune`
- Batch path: `src/batch`
