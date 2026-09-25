# Demo: Batch Inference at Scale

## Story

Show that the same structured extraction approach can be applied across large note volumes, not just interactive requests.

## Execution Layer

- Input generation: `src/batch/01_prepare_batch_input.py`
- Spark / UDF path: `src/batch/02_batch_inference_udf.py`
- SGC path: `src/batch/03_batch_inference_sgc.py`

## Suggested Steps

1. Ensure MedGemma has been registered and the Unity Catalog volume `<catalog>.<schema>.model_weights` exists.
2. Generate or prepare the input note set.
3. Choose either the UDF path or the SGC path.
4. Run distributed extraction.
5. Show the structured output table or sample rows.
6. Explain where the outputs would flow next in a production pipeline.

## Best Fit Workloads

- EHR / clinical notes
- clinical trial data capture
- veterinary diagnostics

## Why This Demo Works

It answers the “How do we operationalize this?” question after the audience is convinced by the structured-output story.
