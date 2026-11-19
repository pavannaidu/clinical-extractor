# Demo: Interactive Comparison Cockpit

## Story

Use the synthetic **EHR Note Extraction** fixture to test one concrete distinction:
the chest X-ray is recommended in six weeks, while primary care is next week.
Compare endpoints on that same note, prompt, and schema:

- a Foundation Model API model
- a custom MedGemma endpoint with XGrammar
- optionally base Qwen, after its endpoint has been attached to the app resource bindings

Use fine-tuned Qwen in a separate NCBI disease-mention comparison with its training
schema. The [data guide](../../docs/data.md) describes the two task contracts.

## Execution Layer

- App UI: `src/app`
- Endpoint catalog: `src/app/models.json`

## Suggested Steps

1. Deploy the bundle and open the `clinical-extractor` app.
2. Choose the [EHR workload pack](../workloads/ehr-clinical-notes/README.md), or select **EHR Note Extraction** in the app.
3. Paste the workload sample text and schema into the app.
4. Select one FMAPI model and one custom endpoint.
5. Run extraction and compare:
   - JSON validity
   - agreement with the note's two follow-up timeframes and vitals
   - unsupported or omitted values
   - observed latency and token usage
6. For a scored comparison, load reviewed JSON answers into Unity Catalog and use
   the [evaluation recipe](../../docs/setup.md#run-a-scored-evaluation). Record
   field agreement separately from schema validity.

## Best Fit Workloads

- EHR / clinical notes
- adverse event reporting
- prior authorization

## Why This Demo Works

It makes the decision framework visible without requiring a deep explanation of the underlying serving implementation.
