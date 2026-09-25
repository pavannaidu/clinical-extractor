# Demo: Interactive Comparison Cockpit

## Story

Show the structured-output decision framework live by comparing:

- a Foundation Model API model
- a custom MedGemma endpoint with XGrammar
- optionally a second custom endpoint such as Qwen, or a fine-tuned variant after it has been attached to the app resource bindings

## Execution Layer

- App UI: `src/app`
- Endpoint catalog: `src/app/models.json`

## Suggested Steps

1. Deploy the bundle and open the `clinical-extractor` app.
2. Choose one workload pack from `../workloads`.
3. Paste the workload sample text and schema into the app.
4. Select one FMAPI model and one custom endpoint.
5. Run extraction and compare:
   - JSON validity
   - extraction quality
   - terminology handling
   - latency or throughput expectations

## Best Fit Workloads

- EHR / clinical notes
- adverse event reporting
- prior authorization

## Why This Demo Works

It makes the decision framework visible without requiring a deep explanation of the underlying serving implementation.
