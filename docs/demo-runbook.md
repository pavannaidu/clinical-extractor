# Demo Runbook

Use this runbook for a customer-facing field demo of structured AI for life sciences on Databricks.

## Goal

Show how a reviewer can turn one clinical note into a schema-checked record,
inspect its facts, and choose a model using measured quality and delivery results.
The concrete opening question is whether the chest X-ray belongs on a six-week
follow-up worklist while the primary-care visit belongs next week.

## Suggested Demo Flow

### 1. Start with the problem

Open the **EHR Note Extraction** preset or the
[EHR workload pack](../showcase/workloads/ehr-clinical-notes/README.md). Point to
“repeat chest x-ray in 6 weeks” and “follow up with primary care next week.” Ask
which field a coordinator needs before adding a record to a follow-up worklist.

This is repository-authored synthetic text. It contains two different timeframes,
so it tests whether the extraction preserves the association between an action
and its timing. Valid JSON alone cannot answer that question. The
[data-source guide](data.md) identifies this fixture and the separate batch and
NCBI inputs.

### 2. Introduce the decision framework

Open [../showcase/decision-framework.md](../showcase/decision-framework.md).

Key message:

- Start with an available managed model or AI Extract for supported field schemas.
- Evaluate custom Qwen/MedGemma when reviewed examples show a reason to own the weights or decoder.
- Test fine-tuning on a fixed task when persistent errors remain, with held-out examples for comparison.

### 3. Pick a workload pack

Choose one folder under [../showcase/workloads](../showcase/workloads):

- `ehr-clinical-notes`
- `adverse-event-reporting`
- `clinical-trial-data-capture`
- `prior-authorization`
- `drug-interaction-screening`

Use the sample note, schema, and prompt directly in the app or endpoint request.
Keep all three fixed across endpoints. The pack's expected-field file is a review
checklist, not a row-level ground-truth answer file.

### 4. Show the app comparison cockpit

Use `src/app` to demonstrate:

- AI Extract for managed extraction
- GPT-5.4 Mini as the general-purpose structured-output baseline
- base Qwen or MedGemma as a custom comparison on the same EHR schema

Check the two follow-up timeframes, heart rate `82`, and oxygen saturation `96`
against the source. Then inspect schema validity, latency, and token usage.
Attach each custom endpoint to the app with `CAN_QUERY` and select it in Model Settings.

For the separate adaptation story, compare base Qwen with fine-tuned Qwen on its
NCBI disease-mention task and `diseases` string-array schema. The broader EHR preset
and the app's richer disease-object preset are different contracts.

Suggested talking points:

- one schema can be passed to different model backends
- custom endpoints still return constrained JSON
- reviewed field comparisons determine whether a model meets the task's acceptance criteria

### 5. Close with scale or specialization

Pick one:

- **Batch**
  Show that the same extraction pattern scales to distributed processing.
- **Fine-tuning**
  Test whether PEFT improves customer-specific terminology handling.

For a concrete recorded result, the November 18, 2026 smoke check produced 20/20
schema-valid rows in each of three batch paths. Those are 60 outputs from the same
20 synthetic inputs. Show the [benchmark scope and limitations](benchmarks.md)
before discussing quality or performance implications.

Close by agreeing on a reviewed dataset and thresholds for field agreement,
unsupported values, omissions, and latency. The
[one-row evaluation recipe](setup.md#run-a-scored-evaluation) is a reproducible
setup check; extend it with independent reviewed examples for a model decision.

## Demo Commands

```bash
databricks bundle validate --strict --target dev
databricks bundle plan --select apps.clinical_extractor,experiments.clinical_extractor_evaluations --target dev
databricks bundle deploy --select apps.clinical_extractor,experiments.clinical_extractor_evaluations --target dev
databricks bundle run clinical_extractor --target dev
```

Optional runnable flows:

Before running registration or the Serverless GPU batch flow, ensure the Unity
Catalog volume `<catalog>.<schema>.model_weights` exists in the namespace configured
by the bundle variables.

```bash
databricks bundle run medgemma_register
databricks bundle run medgemma_deploy
databricks bundle run medgemma_benchmark
```

```bash
databricks bundle run prepare_batch_input
databricks bundle run batch_inference_udf
```

```bash
databricks bundle run inference_vllm_qwen
databricks bundle run finetune_qwen3_disease
```

The [source guide](../src/README.md) lists the optional MedGemma fine-tuning and
serving paths alongside the primary Qwen workflow.

## What to Emphasize

- structured output is enforced at decoding time, not repaired after generation
- one endpoint can serve many schemas
- the solution works for domain models and PEFT variants
- the same approach supports online inference and distributed batch pipelines
