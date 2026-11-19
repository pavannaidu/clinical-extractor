# Historical MedGemma serving optimization

Recorded February–March 2026. All runs are from the original demo's MLflow
benchmark experiment. Version and configuration descriptions
refer to those experiments; see [setup](../setup.md) for current deployment settings.

## Benchmark History

| Date | Version | GPU | Run Name | c1 tok/s | c4 tok/s | Errors (c1) |
|------|---------|-----|----------|----------|----------|-------------|
| 02-27 | — | A10 | `endpoint_benchmark_20260227_231637` | 11.4 | **30.5** | 2 |
| 02-28 | — | A10 | `endpoint_benchmark_20260228_063725` | 15.8 | 16.7 | 3 |
| 02-28 | — | A10 | `endpoint_benchmark_20260228_162508` | 12.0 | 12.1 | 2 |
| 02-28 | — | A10 | `endpoint_benchmark_20260228_193400` | 12.1 | 12.5 | 0 |
| 03-03 | v26 | T4 | `endpoint_benchmark_20260303_200555` | 12.1 | 12.1 | 1 |
| 03-03 | v26 | A10 | `endpoint_benchmark_20260303_203613` | 16.1 | 16.3 | 2 |
| 03-03 | v27 | A10 | `endpoint_benchmark_20260303_231910` | 11.6 | 11.8 | 7 |
| 03-03 | v27 | T4 | `endpoint_benchmark_20260303_234500` | 8.5 | 8.0 | 11 |
| 03-03 | v28 | A10 | `endpoint_benchmark_20260304_015223` | 11.8 | 11.9 | 9 |
| 03-03 | v28 | T4 | `endpoint_benchmark_20260304_020910` | 8.5 | 8.3 | 9 |
| 03-03 | v29 | A10 | `endpoint_benchmark_20260304_035815` | 14.2 | 14.2 | 7 |
| 03-03 | v29 | T4 | `endpoint_benchmark_20260304_041213` | 10.0 | 10.1 | 4 |
| 03-04 | v31 | A10 | `endpoint_benchmark_20260304_192643` | 8.5 | 8.0 | 7 |
| 03-04 | v31 | T4 | `endpoint_benchmark_20260304_194739` | 6.3 | 4.8 | 3 |
| 03-05 | v32 | A10 | `endpoint_benchmark_20260305_184244` | 13.6 | 14.3 | 3 |
| 03-05 | v32 | T4 | `endpoint_benchmark_20260305_185527` | 10.1 | 10.4 | 6 |
| 03-05 | v26 | A10 | `endpoint_benchmark_20260305_195255` | 14.5 | **27.7** | 3 |
| 03-05 | v26 | T4 | `endpoint_benchmark_20260305_202545` | 10.5 | 10.6 | 5 |

## Model Version Configs

| Version | Run Name | Quantization | Generation | Sampling | max_new_tokens |
|---------|----------|-------------|------------|----------|----------------|
| v20–v26 | `medgemma-xgrammar-bf16` | None (bf16) | Single-row | temp=0.7 | 512 |
| v27–v28 | `medgemma-xgrammar-bf16` | None (bf16) | Batch left-padded | Greedy | 256 |
| v29 | `medgemma-xgrammar-nf4` | NF4 4-bit (bitsandbytes) | Single-row | temp=0.7 | 512 |
| v31 | `medgemma-xgrammar-nf4` | NF4 4-bit (bitsandbytes) + StaticCache + `torch.compile(mode="default")` | Single-row | temp=0.7 | 512 |
| v32 | `medgemma-xgrammar-nf4` | NF4 4-bit (bitsandbytes) + SDPA | Single-row | temp=0.7 | 256 |
| v37 | `medgemma-xgrammar-adaptive` | Adaptive: bf16 on A10+ / NF4 on T4 + flash-attn fallback + grammar cache | Single-row | temp=0.7 | 256 |

Early runs (02-25 to 02-26) used `medgemma-xgrammar-bnb-4bit` and `medgemma-xgrammar-flash-attn` — no benchmark data for those.

## Key Findings

1. **30.5 tok/s peak (02-27) was bf16, not NF4** — achieved at c4 on a warm A10 endpoint. Likely driven by runtime continuous batching, not model config.
2. **Batch+greedy (v27–v28) regressed ~25%** vs bf16 single-row baseline. Batch generation is a no-op at the serving layer — Databricks dispatches one request per replica.
3. **NF4 quantization (v29) improves ~20%** over bf16 batch (v28) on both GPUs. A10: 11.8→14.2, T4: 8.5→10.0.
4. **T4 is viable** — ~70% of A10 throughput, fewer errors with NF4.
5. **Error rates improved with NF4** — A10: 9→7, T4: 9→4 at c1.
6. **StaticCache + torch.compile regressed (v31)** — ~40% slower on A10 (14.2→8.5 tok/s), ~37% on T4 (10.0→6.3). `reduce-overhead` mode caused a CUDAGraphs/StaticCache conflict (RuntimeError); `default` mode avoided the crash but added compilation overhead without benefit. Latencies also degraded significantly (A10 p50: 7s, p99: 59s). **Reverted to v29.**
7. **SDPA + max_new_tokens=256 (v32) — neutral.** A10: 13.6/14.3 tok/s (~same as v29's 14.2). T4: 10.1/10.4 (~same as v29's 10.0). Flash Attention 2 was attempted but the `flash-attn` wheel is incompatible with the Model Serving container. SDPA is the default attention backend anyway. Practical win: max_new_tokens 512→256 bounds worst-case latency.
8. **KV cache is already active** — HuggingFace `model.generate()` uses DynamicCache by default. The bottleneck is NF4 dequantization bandwidth + XGrammar constraint checking, not cache strategy.
9. **bf16 (v26) re-benchmark confirms c4 scaling on A10** — 14.5→27.7 tok/s at c4 (~2x). NF4 (v32) shows no c4 scaling (13.6→14.3) because it's memory-bandwidth-bound from dequantization. T4 shows no c4 scaling for either (VRAM-limited). **For throughput under load, bf16 on A10 is the best config.**

## Recorded `model.py` configuration (v37)

- Adaptive quantization: bf16 on A10+ (SM ≥ 8.0), NF4 4-bit on T4 (SM 7.5)
- Flash Attention 2 with try/except fallback to SDPA
- Grammar cache (avoids recompiling same JSON schema per request)
- Single-row generation loop
- `temperature=0.7`, `do_sample=True`, `max_new_tokens=256`
- XGrammar constrained generation
