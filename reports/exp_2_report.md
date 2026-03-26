# Experiment 2: Softmax Pipeline Optimization for MLA Decode FP8 Kernel

## Target Kernel
`split_kv_kernel` in `flashinfer/mla/cute_dsl/mla_decode_fp8.py` (M=128 config)

## Baseline
- Committed code at `ad074335` (M=128)
- **75,220 cycles average** (NCU, 11 invocations, re-run for fair comparison)
- **44.65 us average**

## Optimization Summary

### Best Result: Opt4
- **73,720 cycles average** (-1,500 cycles, **-2.0%**)
- **43.98 us average** (-0.67 us, **-1.5%**)
- Min observed: 72,284 cycles

### What was changed

Three source-level optimizations in the softmax pipeline:

1. **Split exchange barriers** (softmax + epilogue): Replaced single 4-warp barriers (`num_threads=128`) with independent pair barriers (`num_threads=64` each). The row_max/row_sum exchange is between warp pairs {warp0,warp2} and {warp1,warp3}. Using one barrier for all 4 warps made each pair wait for the other pair unnecessarily.

2. **Early S release**: Moved `fence_view_async_tmem_load()` + `mma_s_pipeline.consumer_release()` from the end of softmax to immediately after the TMEM load. After loading S to registers, no subsequent operation reads from the S TMEM buffer (row_max reduction, softmax, quantize, P store all operate on registers/SMEM). The correction factor TMEM store is at a separate offset (`correction_factor_offset=384` vs S at `0-128`). This allows the MMA warp to start the next QK computation much earlier.

3. **Pipeline wait/acquire overlaps**:
   - Added `consumer_try_wait` for mma_s before the blocking `p_mma_producer_acquire`, so both waits overlap.
   - Added `producer_try_acquire` for p_cor right after p_mma commit, so the barrier check overlaps with the row_sum computation.

### Why it works

The MLA kernel has a critical pipeline: MMA warp produces S (QK result), softmax warp consumes S and produces P, MMA warp consumes P for PV computation. The MMA warp alternates between mma_qk and mma_pv:

```
MMA:     [mma_qk] → [wait P] → [mma_pv] → [wait S_free] → [mma_qk] → ...
Softmax: [wait S] → [TMEM load] → [softmax] → [P store] → [release S] → ...
```

Before optimization, S release happened after the entire softmax computation, meaning the MMA warp had to wait for softmax to complete before starting the next mma_qk. With early S release, the timeline becomes:

```
MMA:     [mma_qk] → [wait P] → [mma_pv] → [mma_qk] → ...  (no wait for S_free!)
Softmax: [wait S] → [TMEM load → release S] → [softmax] → [P store] → ...
```

The split barriers reduce synchronization overhead for the warp exchange step.

## All Experiments

| Exp | Description | Cycles (avg) | Result |
|-----|------------|-------------|--------|
| Baseline | Committed M=128 code | 75,220 | - |
| Opt1 | Split barriers + early S release | 74,799 | -0.56% |
| **Opt2** | **Opt1 + try_wait/try_acquire overlaps** | **74,235** | **-1.31%** |
| Opt3 | Opt2 + deferred p_mma acquire | 75,674 | **WORSE** (+0.60%) |
| **Opt4** | **Opt2 + row_sum before quantize/P-store** | **73,720** | **-2.0%** |

## Key Learnings

1. **S TMEM area is separate from correction TMEM area** — S at offset 0-128, O at 128-384, correction at 384+. Safe to release S early without affecting correction metadata writes.
2. **4-warp barrier for 2-warp exchange was wasted synchronization** — splitting into pair barriers removes unnecessary cross-pair waiting.
3. **try_wait/try_acquire patterns help overlap independent waits** — issuing non-blocking tries before blocking acquires hides latency.
4. **Source-level pipeline optimizations complement SASS-level tuning** — Exp 1 got 0.83% from SASS; this gets 2.0% from source-level changes. Combined they should stack.
5. **Deferring p_mma acquire hurts** — the p_mma pipeline has 2 stages, so the early blocking acquire overlaps with the mma_s wait. Deferring it to right before P store creates a stall at that critical point instead.
6. **Moving row_sum before quantize/P-store helps** — row_sum FADD reductions overlap with MUFU exp2 pipeline drain, and moving it off the critical path between P-commit and p_cor-acquire reduces the softmax-to-correction latency.
