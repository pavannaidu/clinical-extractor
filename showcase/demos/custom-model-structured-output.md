# Demo: Custom Model Structured Output

## Story

Show that a Unity Catalog-served custom model can return schema-valid JSON without post-processing by using XGrammar at decoding time.

## Execution Layer

- Registration: `src/medgemma/01_register.py`
- Deployment: `src/medgemma/02_deploy_endpoint.py`
- Benchmarking: `src/medgemma/03_benchmark_endpoint.py`
- Serving model: `src/medgemma/model.py`

## Suggested Steps

1. Create the Unity Catalog volume `<catalog>.<schema>.model_weights` if it does not already exist (`pavan_naidu_catalog.mlops.model_weights` for the default target).
2. Register MedGemma as an MLflow model in Unity Catalog.
3. Deploy the endpoint.
4. Pick a workload pack from `../workloads`.
5. Invoke the endpoint with the workload schema passed in `custom_inputs.schema`.
6. Show that the response is valid JSON matching the requested structure.

## Best Fit Workloads

- adverse event reporting
- clinical trial data capture
- drug interaction screening

## Why This Demo Works

It directly addresses the core thesis: custom models on Databricks do not have to be limited to raw text output.
