# Experiment 5: Split-Epilogue Optimization

## Goal
Split the mma_o pipeline into per-iter_n pipelines and parallelize the epilogue across softmax and correction warps. Instead of correction warps processing both iter_n=0 and iter_n=1 sequentially, correction warps handle iter_n=0 while softmax warps handle iter_n=1 in parallel.

## Baseline
- ~73,700-74,900 cycles / ~44 us (split_kv kernel, seq_len_k=81920, seq_len_q=4)
- From commit `54ceca07` (hoist MUFU.RCP out of epilogue inner loop)

## Changes Implemented

### Pipeline Split
1. Added `mma_o_per_n_stage = mma_o_stage // iterations_pv_n` (= 1 stage per iter_n)
2. Replaced single `mma_o_pipeline` with `mma_o_pipelines[]` list (2 pipelines, 1 stage each)
3. Each pipeline uses separate barrier storage regions within the same mbar allocation

### MMA Warp Changes
- Split single `mma_o_producer_state` into `mma_o_ps_0` and `mma_o_ps_1`
- `mma_pv()` now produces to pipelines[0] for acc_stage=0 and pipelines[1] for acc_stage=1
- Unrolled the acc_stage loop into two explicit blocks for compile-time constant pipeline indexing

### Compute (Softmax) Warp Changes
- Added `tiled_mma_pv`, `mma_o_pipelines`, and `H` to compute_common_params
- `compute()` now returns `row_sum` and `row_max` (previously consumed internally)
- After `compute()` returns, softmax warp advances `mma_o_cs_1` by `k_tile_count - 1` (skipping rescale iterations consumed by correction warp) then calls `_epilogue_single()` for iter_n=1

### Correction Warp Changes
- Split `mma_o_consumer_state` into `mma_o_cs_0` and `mma_o_cs_1`
- `rescale()` consumes from both pipelines (per iter_n)
- `correction()` epilogue only handles iter_n=0 via `_epilogue_single()`
- Returns 3 values: `p_cor_consumer_state, mma_o_cs_0, mma_o_cs_1`

### New `_epilogue_single()` Function
- Handles epilogue for a single iter_n value
- Takes separate exchange barrier parameters (pair02 and pair13) for split-barrier optimization
- Performs: row_sum exchange -> TMEM load -> scale/normalize -> type convert -> STG -> LSE store
- Used by both softmax warps (iter_n=1) and correction warps (iter_n=0)

### Barrier Usage
- Softmax warp epilogue (iter_n=1): `softmax_exchange_sync_bar_pair02/13`
- Correction warp epilogue (iter_n=0): `epilogue_exchange_sync_bar_pair02/13`

## Results
- Correctness: **PASS** (verified at seq_len_k=81920, seq_len_q=4)
- Performance:
  | Metric | Minimum | Maximum | Average |
  |--------|---------|---------|---------|
  | gpu__time_duration.sum (us) | 43.74 | 46.46 | 44.57 |
  | sm__cycles_elapsed.avg (cycles) | 73,953 | 78,485 | 74,877 |

## Analysis
- Performance is approximately neutral compared to baseline (~74,877 vs ~73,700-74,900 cycles)
- The minimum of 73,953 cycles is very close to the baseline minimum
- The split-epilogue enables structural parallelism between softmax and correction warps during epilogue
- The small overhead comes from managing two pipeline instances instead of one
- This is a structural refactor that enables further optimization opportunities (e.g., overlapping epilogue STG with other work)

## Test Configuration
- batch_size=1, seq_len_q=4, seq_len_k=81920, num_heads=128
- latent_dim=512, rope_dim=64, page_size=64
- FP8 input/output, FP32 accumulator
- 2-CTA cluster, split_kv=-1 (auto)
