# Demo Runbook

Use this runbook for a customer-facing field demo of structured AI for life sciences on Databricks.

## Goal

Show that Databricks can support a progression from Foundation Model API structured extraction to domain-specific and fine-tuned custom models while keeping JSON outputs reliable enough for downstream workflows.

## Suggested Demo Flow

### 1. Start with the problem

Open [../README.md](../README.md) or [../showcase/overview.md](../showcase/overview.md) and explain:

- FMAPI models support structured output in managed flows
- custom Unity Catalog-served models often return raw text
- life sciences workloads need deterministic JSON for production

### 2. Introduce the decision framework

Open [../showcase/decision-framework.md](../showcase/decision-framework.md).

Key message:

- try FMAPI first
- move to a domain model plus XGrammar if clinical quality matters more
- move to PEFT plus XGrammar if the customer's terminology still requires adaptation

### 3. Pick a workload pack

Choose one folder under [../showcase/workloads](../showcase/workloads):

- `ehr-clinical-notes`
- `adverse-event-reporting`
- `clinical-trial-data-capture`
- `prior-authorization`
- `drug-interaction-screening`

Use the sample note, schema, and prompt directly in the app or endpoint request.

### 4. Show the app comparison cockpit

Use `src/app` to demonstrate:

- AI Extract for managed extraction
- GPT-5.4 Mini as the general-purpose structured-output baseline
- fine-tuned Qwen as the custom model, using its disease-extraction task

Compare base Qwen with fine-tuned Qwen when explaining adaptation. Bring in
MedGemma as an optional medical pretrained model comparison. Attach each custom
endpoint to the app with `CAN_QUERY` and select it in Model Settings.

Suggested talking points:

- one schema can be passed to different model backends
- custom endpoints still return constrained JSON
- the app makes the tradeoff between management simplicity and domain accuracy visible

### 5. Close with scale or specialization

Pick one:

- **Batch**
  Show that the same extraction pattern scales to distributed processing.
- **Fine-tuning**
  Test whether PEFT improves customer-specific terminology handling.

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
