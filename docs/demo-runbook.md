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

- a ready FMAPI endpoint
- a custom MedGemma endpoint
- optionally a fine-tuned or alternate custom endpoint that has been attached to the app with `CAN_QUERY` and selected in Model Settings

Suggested talking points:

- one schema can be passed to different model backends
- custom endpoints still return constrained JSON
- the app makes the tradeoff between management simplicity and domain accuracy visible

### 5. Close with scale or specialization

Pick one:

- **Batch**
  Show that the same extraction pattern scales to distributed processing.
- **Fine-tuning**
  Show that PEFT improves customer-specific terminology handling.

## Demo Commands

```bash
databricks bundle validate --profile FEVM
databricks bundle plan --select apps.clinical_extractor --target dev --profile FEVM
databricks bundle deploy --select apps.clinical_extractor --target dev --profile FEVM
databricks bundle run clinical_extractor --target dev --profile FEVM
```

Optional runnable flows:

Before running registration or the Serverless GPU batch flow, ensure the Unity
Catalog volume `<catalog>.<schema>.model_weights` exists. The default is
`pavan_naidu_catalog.mlops.model_weights`.

```bash
databricks bundle run medgemma_register  --profile FEVM
databricks bundle run medgemma_deploy    --profile FEVM
databricks bundle run medgemma_benchmark --profile FEVM
```

```bash
databricks bundle run prepare_batch_input --profile FEVM
databricks bundle run batch_inference_udf --profile FEVM
```

```bash
databricks bundle run finetune_medgemma --profile FEVM
```

## What to Emphasize

- structured output is enforced at decoding time, not repaired after generation
- one endpoint can serve many schemas
- the solution works for domain models and PEFT variants
- the same approach supports online inference and distributed batch pipelines
