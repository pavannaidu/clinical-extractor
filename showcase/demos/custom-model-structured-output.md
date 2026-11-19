# Demo: Custom Model Structured Output

## Story

Show that a Unity Catalog-served custom model can return schema-valid JSON without post-processing by using XGrammar at decoding time.

## Execution Layer

The primary custom-model path is Qwen:

- Base serving: `src/models/qwen/deploy_vllm.py` (`inference_vllm_qwen`)
- Disease-extraction fine-tuning: `src/models/qwen/finetune_disease.py` (`finetune_qwen3_disease`)

Run those jobs to prepare the endpoints, then compare the base and fine-tuned
models using the disease-extraction task. The app passes schemas to vLLM through
`guided_json` and validates the returned JSON. Use reviewed ground truth to
measure whether fine-tuning improves accuracy.

MedGemma is an optional medical pretrained model comparison. Its XGrammar path is:

- Registration: `src/models/medgemma/xgrammar/01_register.py`
- Deployment: `src/models/medgemma/xgrammar/02_deploy_endpoint.py`
- Benchmarking: `src/models/medgemma/xgrammar/03_benchmark_endpoint.py`
- Serving model: `src/models/medgemma/xgrammar/model.py`

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
