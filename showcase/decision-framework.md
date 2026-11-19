# Decision Framework

Use this framework when deciding how to deliver structured extraction from unstructured clinical text on Databricks.

## Step 1: Start with FMAPI

If a Foundation Model API model gives acceptable extraction quality and structured output, use that first.

Why:

- lowest operational complexity
- fully managed experience
- fastest path to production

## Step 2: Move to a Domain Model + XGrammar

If FMAPI results are not accurate enough on clinical language, move to a domain-specific model such as MedGemma and enforce structured output with XGrammar.

Why:

- better performance on clinical terminology
- caller-supplied schema support
- valid JSON at decoding time

## Step 3: Add PEFT Fine-Tuning + XGrammar

If a base domain model still misses the customer’s specialized terminology, fine-tune with PEFT and keep XGrammar in the serving path.

Why:

- better alignment to customer language and task framing
- lower adaptation cost than full fine-tuning
- preserves schema-valid output

## Decision Tree

```text
Need structured JSON from unstructured clinical text?
|
|-- Try FMAPI first
|   |-- Accurate enough? -> Ship it
|
|-- Need better clinical extraction quality?
|   |-- Use a domain model + XGrammar
|   |-- Accurate enough? -> Ship it
|
|-- Need customer-specific adaptation?
    |-- Fine-tune with PEFT + XGrammar
```

## Mapping to This Repo

- FMAPI comparison story: `src/app`
- Domain model + XGrammar: `src/models/medgemma/xgrammar`
- Primary custom-model demo: `src/models/qwen/deploy_vllm.py`
- Qwen disease-extraction fine-tuning: `src/models/qwen/finetune_disease.py`
- Medical-model vLLM comparison: `src/models/medgemma/deploy_vllm.py`
- MedGemma PEFT + XGrammar: `src/models/medgemma/finetune.py`
- Batch scale-out path: `src/models/medgemma/batch`
