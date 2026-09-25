# Architecture

This repository uses a narrative overlay: the new `showcase/` area is the presentation layer, while the existing `src/` folders remain the execution layer.

The portfolio architecture below shows how the interactive comparison uses one acceptance schema across managed and custom serving, while batch and fine-tuning remain adjacent references rather than app modes. The batch box contains two alternative routes, not one pipeline. The fine-tuning reference uses synthetic training data and represents a reusable implementation pattern rather than a production deployment. This is an asset map, not a diagram of customer production systems.

![Clinical Extractor portfolio architecture](diagrams/clinical-extractor-portfolio.png)

The [Graphviz source](diagrams/clinical-extractor-portfolio.dot) can be regenerated. The [editable XML](diagrams/clinical-extractor-portfolio.xml) was produced for Lucid Chart import; its appearance after import has not yet been checked.

## Capability Map

| Capability | Customer-facing message | Execution layer |
|---|---|---|
| Domain model structured extraction | Test domain-specific quality with XGrammar-constrained JSON; clinical lift still needs reviewed evaluation | `src/medgemma` |
| Alternate custom serving path | Compare vLLM/XGrammar serving with the direct model wrapper | `src/vllm` |
| Batch inference | Reference paths for high-volume structured inference, measured separately from online serving | `src/batch` |
| PEFT fine-tuning | Test whether adaptation improves a repeatable clinical-language error; the reference flow does not establish lift | `src/finetune` |
| Interactive comparison cockpit | Compare AI Extract, FMAPI, and custom endpoints against one input and requested schema | `src/app` |

## Presentation Layer

- `README.md`
  Primary landing page for customers and field teams.
- `showcase/overview.md`
  Business problem, solution thesis, and workloads.
- `showcase/decision-framework.md`
  When to use FMAPI, domain models, or fine-tuning.
- `showcase/workloads/*`
  Concrete demo packs with sample text, schema, system prompt, and expected fields.
- `showcase/demos/*`
  Guided walkthroughs that point back to runnable code paths.
- `showcase/shared/*`
  Reusable prompts and references for demos.

## Execution Layer

### `src/medgemma`

End-to-end MedGemma registration, deployment, and benchmark flow with XGrammar-backed constrained decoding.

### `src/vllm`

Alternate custom model serving paths for MedGemma and Qwen with guided JSON configuration.

### `src/batch`

Delta input generation plus two scalable batch execution styles:

- Spark / UDF-based
- SGC + Ray-based

### `src/finetune`

PEFT fine-tuning flow for customer-specific domain adaptation.

### `src/app`

The comparison cockpit for:

- AI Extract, FMAPI, and custom model selection
- caller-supplied JSON schema extraction
- side-by-side result and schema-status review
- short endpoint load tests, not production capacity certification

`src/app/models.json` seeds the default endpoint selection and is the source of
truth for explicit request-contract mappings and default inference options.
Operators can override the active selection through the app; that selection is
stored in Lakebase when configured and otherwise survives only for the running
process. Every endpoint that the app may invoke must also be attached under
`resources.apps.clinical_extractor.resources` in `databricks.yml` with
`CAN_QUERY`; changing the in-app selection does not grant endpoint permissions.
Rebuild `src/app/static` from the frontend with `npm run build` before an app
deployment.

The MedGemma registration and Serverless GPU batch paths share model weights
through `/Volumes/<catalog>/<schema>/model_weights`. The `model_weights` Unity
Catalog volume must be created before either path runs; the bundle does not
create it.

## Shared Narrative

The portfolio story follows the evidence in this order:

1. Customer use cases expose different needs for schema-governed clinical and business outputs.
2. Early testing made the first hard blocker concrete: an open-weight XGrammar endpoint had the right output shape but insufficient interactive speed in its initial T4 configuration.
3. Clinical Extractor evolved into a comparison of managed extraction, FMAPI, and custom vLLM/XGrammar serving against one requested schema, with independent post-validation.
4. Batch inference and QLoRA-to-Model-Serving are adjacent reference paths, not modes of the interactive app. Reviewed semantic quality and workload cost remain separate gates.
5. Workload and serving proofs of concept are platform evidence, not a claim that Clinical Extractor runs customer products or caused production deployments.
