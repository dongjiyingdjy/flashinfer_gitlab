# Experiment 6: Epilogue Optimization - RCP Hoisting, Early TMEM Release, Conditional Fence

## Goal
Reduce the cycle count of the epilogue section by optimizing three areas:
1. Hoist `MUFU.RCP` out of the per-element loop in `_epilogue_single`
2. Release TMEM buffers earlier in the epilogue (immediately after TMEM load into registers)
3. Skip unnecessary `fence_view_async_tmem_store()` in `rescale()` when correction is skipped

## Baseline
- ~74,826 avg cycles / ~44.70 us (split_kv kernel, seq_len_k=81920, seq_len_q=4)
- Min: 73,750 cycles
- From commit `3865adec` (split mma_o pipeline into per-iter_n pipelines for parallel epilogue)

## Changes Implemented

### 1. RCP Hoisting in `_epilogue_single` (and `epilogue`)
The `_epilogue_single` function (introduced in commit `3865adec`) did not inherit the RCP hoisting optimization from the earlier `epilogue` function (commit `54ceca07`). The code had:
```python
for i in cutlass.range(...):
    tTR_rAcc[i] = tTR_rAcc[i] * epilogue_params.output_scale * cute.arch.rcp_approx(row_sum)
```
This generates a `MUFU.RCP` instruction per element (~64 elements per warp). Changed to:
```python
epi_scale = epilogue_params.output_scale * cute.arch.rcp_approx(row_sum)
for i in cutlass.range(...):
    tTR_rAcc[i] = tTR_rAcc[i] * epi_scale
```
This reduces to a single `MUFU.RCP` per warp.

### 2. Early TMEM Fence and Release in `_epilogue_single` (and `epilogue`)
Previously, the TMEM fence + pipeline release happened at the very end of the epilogue, after all STG operations and LSE stores:
```
TMEM load -> scale -> type convert -> STG -> LSE store -> fence -> release
```
Changed to release immediately after loading TMEM into registers:
```
TMEM load -> fence -> release -> scale -> type convert -> STG -> LSE store
```
This frees the TMEM buffer for the MMA warp to start accumulating the next tile sooner, improving pipeline overlap.

### 3. Conditional TMEM Store Fence in `rescale()`
When `skip_correction` is true (all warps agree the row_max hasn't changed), the rescale skips the TMEM load/multiply/store but still issued `fence_view_async_tmem_store()`. Moved the fence inside the `if not skip_correction` block, since no TMEM store was performed when correction is skipped.

## Results
- Correctness: **PASS** (verified at seq_len_k=81920, seq_len_q=4, with reference check)
- Performance (2 runs):

  | Run | Minimum (cycles) | Maximum (cycles) | Average (cycles) | Average (us) |
  |-----|------------------|------------------|------------------|--------------|
  | 1   | 73,492           | 78,138           | 74,693           | 44.38        |
  | 2   | 73,447           | 78,129           | 74,522           | 44.32        |

- Improvement vs baseline:
  - Average: ~200-300 cycles (~0.3-0.4%)
  - Minimum: ~300 cycles (73,750 -> 73,450)
  - Latency: ~0.3-0.4 us (44.70 -> 44.32 us)

## Stall Analysis
Profiled stall reasons (per-issue-active ratios):
| Stall Reason | Ratio |
|---|---|
| long_scoreboard (memory latency) | 4.38 |
| wait (pipeline barriers) | 1.64 |
| short_scoreboard (shared memory) | 0.66 |
| not_selected (scheduler) | 0.18 |
| barrier (named barriers) | 0.16 |
| math_pipe_throttle | 0.10 |

The dominant bottleneck is `long_scoreboard` (4.38x) which represents TMA load and TMEM operation latency. The epilogue is not on the critical path for the steady-state loop -- it only runs once at the end of each split_kv partition. At seq_len_k=81920 with 18 splits, each partition processes ~36 tiles, so the epilogue cost is amortized across many iterations.

## Key Insights
1. The `_epilogue_single` function introduced in commit `3865adec` inadvertently regressed the RCP hoisting optimization from commit `54ceca07`
2. Early TMEM release improves pipeline overlap but the effect is limited since epilogue runs once at kernel end
3. The `epilogue()` function (non-split version) is dead code -- it's defined but never called. The split epilogue via `_epilogue_single` is the active code path
4. At seq_len_k=81920, the kernel is memory-latency-bound (long_scoreboard dominant). Further epilogue optimization has diminishing returns; the main loop is the bottleneck
5. The conditional fence in rescale avoids an unnecessary `fence_view_async_tmem_store` in the common skip-correction case, slightly reducing instruction count in the steady-state loop

## Test Configuration
- batch_size=1, seq_len_q=4, seq_len_k=81920, num_heads=128
- latent_dim=512, rope_dim=64, page_size=64
- FP8 input/output, FP32 accumulator
- 2-CTA cluster, split_kv=-1 (auto = 18)
