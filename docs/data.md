# Data sources and reproducibility

The opening demo uses repository-authored synthetic clinical text. The optional
Qwen training workflow uses a separate, named public biomedical corpus. These
sources support different tasks and should be evaluated with their own schemas.

## EHR Note Extraction fixture

**Name:** EHR Note Extraction. **Preset ID:** `ehr_note_extraction`.
**Source:** [built-in presets](../src/app/api/presets.py) and the
[EHR sample note](../showcase/workloads/ehr-clinical-notes/sample-note.txt).

This synthetic pneumonia follow-up note explicitly includes:

| Fact | Value stated in the fixture |
|---|---|
| Imaging follow-up | Repeat chest X-ray in 6 weeks |
| Primary-care follow-up | Next week |
| Heart rate | 82 |
| Oxygen saturation | 96 percent on room air |
| Active diagnoses | Improving community-acquired pneumonia; type 2 diabetes mellitus |

The two timeframes make a concrete review test: transferring “next week” to the
X-ray would produce a wrong fact even if the JSON passes its schema. The fixture
also contains medication regimens and vitals for checking extraction fidelity.
It is authored demo text, not an observed patient record or a clinical study.

## Life Sciences Workload Packs

The five named packs are local fixtures; no external dataset download is needed.
Each contains a sample, JSON schema, system prompt, and expected-field checklist.

| Pack | Input | Schema | Review checklist |
|---|---|---|---|
| EHR / Clinical Notes | [note](../showcase/workloads/ehr-clinical-notes/sample-note.txt) | [schema](../showcase/workloads/ehr-clinical-notes/schema.json) | [expected fields](../showcase/workloads/ehr-clinical-notes/expected-fields.md) |
| Adverse Event Reporting | [note](../showcase/workloads/adverse-event-reporting/sample-note.txt) | [schema](../showcase/workloads/adverse-event-reporting/schema.json) | [expected fields](../showcase/workloads/adverse-event-reporting/expected-fields.md) |
| Clinical Trial Data Capture | [note](../showcase/workloads/clinical-trial-data-capture/sample-note.txt) | [schema](../showcase/workloads/clinical-trial-data-capture/schema.json) | [expected fields](../showcase/workloads/clinical-trial-data-capture/expected-fields.md) |
| Prior Authorization | [note](../showcase/workloads/prior-authorization/sample-note.txt) | [schema](../showcase/workloads/prior-authorization/schema.json) | [expected fields](../showcase/workloads/prior-authorization/expected-fields.md) |
| Drug Interaction Screening | [note](../showcase/workloads/drug-interaction-screening/sample-note.txt) | [schema](../showcase/workloads/drug-interaction-screening/schema.json) | [expected fields](../showcase/workloads/drug-interaction-screening/expected-fields.md) |

Use each pack's `system-prompt.txt` with its schema. Keep the input, prompt, and
schema fixed when comparing endpoints. Built-in presets and workload-pack schemas
can differ in required-field and null handling, so record the exact selected
contract with your results.

The expected-field files list what a reviewer should inspect; they are not
machine-readable answer labels. The [one-row evaluation recipe](setup.md#run-a-scored-evaluation)
provides explicit ground-truth JSON and its matching schema.

## MedGemma Synthetic Batch Input

**Delta table name:** `<catalog>.<schema>.medgemma_batch_input`.
**Source:** [batch preparation notebook](../src/models/medgemma/batch/01_prepare_batch_input.py).

The notebook defines 20 fixed seed records and seven schemas:

| Category | Seed records |
|---|---:|
| Medication | 4 |
| Condition | 4 |
| Vitals | 3 |
| Laboratory result | 3 |
| Allergy | 2 |
| Procedure | 2 |
| Discharge | 2 |

Each row contains `id`, `category`, `prompt`, and the per-row JSON `schema`.
The `synthetic_replications` widget controls repeated copies; IDs gain a replication
suffix, such as `med_001_0`. The bundled job sets 50 replications, producing 1,000
rows. One replication produces the 20 rows used in the recorded workflow smoke
check. Repeated copies increase workload size without increasing case diversity.

Preparation overwrites the configured input table. Choose a dedicated demo
namespace before running it. Model jobs persist batch results and registered models
in the configured Unity Catalog namespace; model weights use its volumes.

## NCBI Disease Corpus

**Dataset ID:** `ncbi/ncbi_disease`.
**Source:** [dataset card](https://huggingface.co/datasets/ncbi/ncbi_disease).
**Pinned revision:** [`ddf78b541d0004431f0eee441e439b7881e6213d`](https://huggingface.co/datasets/ncbi/ncbi_disease/tree/ddf78b541d0004431f0eee441e439b7881e6213d).
**Consumer:** [Qwen fine-tuning notebook](../src/models/qwen/finetune_disease.py).

The corpus supplies published biomedical text as tokens with disease-mention
annotations. The notebook downloads the pinned train, validation, and test
Parquet files. It joins token spans using BIO tags, keeps unique mentions in their
first-appearance order, and turns them into targets with this shape:

```json
{"diseases": ["asthma"]}
```

The default job selects at most 2,048 training and 256 validation examples after a
seed-42 shuffle, then filters examples to the 1,024-token context limit. Actual
counts can be lower after filtering. Training uses the train split; validation
uses the validation split. Downloading the test file does not establish a held-out
test score, and this repository publishes no NCBI accuracy improvement claim.
Consult the pinned dataset card for its original notice and citation.

The base model is `Qwen/Qwen3-4B-Instruct-2507`, pinned by the notebook to revision
`cdbee75f17c01a7cc42f958dc650907174af0554`. The fine-tuned model's string-array
contract differs from the app's richer **Disease Mention Extraction** preset,
which requests objects with extra attributes. Use the training contract for a
base-versus-fine-tuned disease-mention comparison.

The recorded execution smoke used tiny synthetic training sets. Those checks
validate training, packaging, and serving; they do not measure performance on the
full NCBI corpus. See [benchmarks](benchmarks.md) for the measured scope.
