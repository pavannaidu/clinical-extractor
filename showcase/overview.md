# Showcase Overview

## Technical Thesis

Structured extraction from clinical text should not be limited to built-in managed models. When a domain-specific or fine-tuned model is the right accuracy choice, Databricks customers still need deterministic structured outputs for downstream pipelines.

This showcase demonstrates that the gap can be closed today by combining:

- Databricks Model Serving
- Unity Catalog model registration
- XGrammar-constrained decoding
- domain-specific models such as MedGemma
- optional PEFT fine-tuning

## The Problem

Foundation Model API workflows can support structured JSON outputs out of the box, but custom models served from Unity Catalog usually emit unconstrained text. In life sciences, unconstrained text is not enough for:

- regulated workflows
- pipeline-grade data extraction
- schema-driven downstream integrations
- repeatable analytics

## Why Life Sciences Is a Strong Fit

- clinical terminology is specialized
- accuracy matters more than generic language quality
- many workflows already expect structured records
- domain drift and customer-specific language make fine-tuning attractive

## What the Showcase Proves

- a domain model can be registered and served on Databricks
- the caller can pass a schema per request
- XGrammar can constrain decoding to valid JSON at inference time
- the same concept can support app UX, REST invocation, and batch processing
- PEFT fine-tuning can be layered on top for customer-specific terminology

## Workloads Covered

- EHR / clinical notes
- adverse event reporting
- clinical trial data capture
- prior authorization
- drug interaction screening

See [workloads](workloads) for concrete example packs.
