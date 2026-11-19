# Showcase overview: from a clinical note to a reviewable record

## A specific operational problem

A MedTech clinical-operations team assembling a follow-up worklist needs to know
which action was recommended and when. In the repository's **EHR Note Extraction**
fixture, a pneumonia follow-up note says “repeat chest x-ray in 6 weeks” and “follow
up with primary care next week.” A reviewer must keep those timeframes attached to
the right actions while extracting the same note's diagnoses, medications, and vitals.

A fluent summary does not provide that field-level contract. JSON validity alone
also cannot establish that the X-ray timeframe is correct. The useful handoff is a
schema-checked record whose individual values can be reviewed against the source.

## The named inputs

Start with the [EHR / Clinical Notes pack](workloads/ehr-clinical-notes/README.md):
its synthetic note, JSON schema, system prompt, and expected-field checklist are
all in the repository. The same note appears in the app's `ehr_note_extraction`
preset. No customer records are needed.

For batch execution, the **MedGemma Synthetic Batch Input** notebook defines 20
seed records across seven categories. For the separate disease-extraction
fine-tuning example, the **NCBI Disease Corpus** is loaded from
`ncbi/ncbi_disease` at a pinned revision. The [data-source guide](../docs/data.md)
records the exact files, dataset revisions, transformations, and task boundaries.

## What to show live

1. Open **EHR Note Extraction** or paste the EHR pack's input, prompt, and schema.
2. Run an available managed model. Add base Qwen or MedGemma if its endpoint is
   deployed and bound to the app. Keep all three inputs fixed across models.
3. Check JSON/schema validity, then inspect the two follow-up timeframes, heart
   rate `82`, and oxygen saturation `96`. Treat a schema-valid wrong fact as a
   review failure.
4. Use a Unity Catalog evaluation table with reviewed JSON answers to measure
   field agreement and fully-correct records. The [setup guide](../docs/setup.md#run-a-scored-evaluation)
   includes a complete one-row example.
5. Review latency and token usage only after reviewing the values. Use load tests
   for an interactive delivery question and batch jobs for a scheduled workload.

The comparison supports a concrete decision: whether a managed endpoint meets the
reviewed acceptance criteria, or a custom model deserves further evaluation.
Custom serving adds model packaging and GPU capacity responsibilities. QLoRA is a
separate experiment when reviewed examples show persistent extraction errors.

## The measurable outcome

The November 18, 2026 workflow smoke produced schema-valid JSON for **20 of 20
inputs in each of three batch paths**, with output IDs and schemas checked against
the inputs. The same 20 fixtures yielded 60 output rows across classic driver,
Spark SQL `ai_query`, and Serverless GPU/Ray execution.

That is an observed structural and execution result. A rollout decision also
needs held-out, reviewed field-level results and a delivery target agreed with the
reviewers. The app reports schema validity, field agreement, fully-correct records,
and latency; [the benchmark guide](../docs/benchmarks.md) states what has actually
been measured and the limits of those measurements.

## Close with a reviewable next step

Choose one field that causes rework, a set of reviewed examples, and acceptance
thresholds for omissions, unsupported values, and latency. Compare the simplest
available managed approach with a custom candidate on that fixed task. For the
fine-tuned Qwen example, use its NCBI disease-mention string-array schema rather
than the broader EHR schema.

Use the [presenter runbook](../docs/demo-runbook.md) for the full walkthrough and
the [decision framework](decision-framework.md) to discuss model ownership choices.
