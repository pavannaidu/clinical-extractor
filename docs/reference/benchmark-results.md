# MedGemma Benchmark Results

## Model

| Parameter | Value |
|---|---|
| Model | `google/medgemma-1.5-4b-it` |
| Revision | `e9792da5fb8ee651083d345ec4bce07c3c9f1641` |
| Dtype | `bfloat16` · no quantization |
| Constrained generation | XGrammar (JSON schema → LogitsProcessor) |
| Registered model | `pavan_naidu.mlops.medgemma_xgrammar_structured` |
| Endpoint | `medgemma-xgrammar-endpoint` |

## Inference Config by Version

| | v21 — unoptimized | v22+ — optimized |
|---|---|---|
| `max_new_tokens` | 512 | 256 |
| Decoding | Stochastic (`temperature=0.7`) | Greedy (`do_sample=False`) |
| Generation | Serial loop (1 prompt/GPU call) | Batched (all prompts in 1 GPU call) |

> **Note:** The v22+ optimizations exist in `src/model.py` locally but were not re-registered before these benchmarks ran. All results below are for the **v21 unoptimized baseline**.

---

## Run 1 — T4 · v21 unoptimized (GPU_SMALL · Small · 2026-02-28)

| Concurrency | OutTok/s | TotTok/s | AvgOutTok | Rows/s | Wall(s) | p50(s) | p90(s) | p99(s) | Errors |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 12.0 | 18.8 | 67.2 | 0.18 | 1116.3 | 4.98 | 7.25 | 8.66 | 2 |
| 4 | 12.1 | 18.1 | 74.9 | 0.16 | 1240.9 | 20.23 | 34.23 | 57.88 | 5 |

**Why OutTok/s is flat at both concurrency levels:** `GPU_SMALL` = 1 T4 replica. BF16 inference is memory-bandwidth bound (~320 GB/s on T4). One request already saturates the GPU at c=1; at c=4, requests queue on the same hardware — total throughput stays flat while p50 latency scales ~4× (4.98s → 20.23s). To scale throughput, increase replicas (`workload_size=Medium/Large`), not concurrency against a single replica.

---

## Run 2 — T4 · v22 optimized (GPU_SMALL · Small · TBD)

_Pending re-register of `src/model.py` (batched greedy, `max_new_tokens=256`)_
