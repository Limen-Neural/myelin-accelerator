# Routing reductions and adjusted-score selection

Reusable device primitives for a row-major `n_nodes × n_routes` score matrix
plus per-node `state` and `adaptation` values. One operation produces:

- per-node top-k route indices, and optionally their probabilities;
- the sum and maximum of per-node Shannon entropy, in bits;
- the node with the highest `state - adaptation_scale * adaptation`, and its score.

The caller supplies `adaptation_scale` and decides what the inputs and results
mean. This crate has no default dynamics and no consumer policy.
`adjusted_argmax` is the canonical adaptation-adjusted selector. Consumers
with temporal state can reuse it instead of adding a policy-specific selector
here.

Scope and acceptance criteria: [LIM-955](https://linear.app/rpd-34/issue/LIM-955).

| Layer | Location |
|-------|----------|
| Device kernels | `cu/routing_selection.cu` |
| Launch wrappers | `GpuAccelerator` |
| Contract, CPU reference, analytical model | `src/routing.rs` |
| Device differential tests | `tests/routing_gpu.rs` |
| Benchmark cases | `examples/benchmark.rs` (`routing_*`, `adjusted_argmax_*`) |

## Usage

```rust,ignore
use myelin_accelerator::routing::{
    FusedRoutingOutputs, FusedRoutingRequest, RoutingParams, RoutingScoreKind,
};
use myelin_accelerator::{GpuAccelerator, GpuBuffer};

let acc = GpuAccelerator::require_gpu()?;
let params = RoutingParams {
    n_nodes,
    n_routes,
    top_k: 4,
    score_kind: RoutingScoreKind::Logits,
    adaptation_scale,
};
let scores = GpuBuffer::from_slice(&scores)?;         // n_nodes * n_routes
let state = GpuBuffer::from_slice(&state)?;           // n_nodes
let adaptation = GpuBuffer::from_slice(&adaptation)?; // n_nodes

// `true` also writes the probabilities of the selected routes.
let mut outputs = FusedRoutingOutputs::alloc(&params, false)?;
acc.fused_routing_select(FusedRoutingRequest {
    scores: &scores,
    state: &state,
    adaptation: &adaptation,
    outputs: &mut outputs,
    params: &params,
})?;

let summary = outputs.read_summary()?;
let top_k = outputs.top_k_indices.to_vec()?; // n_nodes * top_k
```

`summary.selection.index` is `None` when no node was eligible.

## Choosing a path

| Method | Launches | Use when |
|--------|----------|----------|
| `fused_routing_select` | 2 | You need top-k, entropy, and selection together, without the probability matrix |
| `routing_softmax`, `routing_top_k`, `routing_entropy_reduce`, `adjusted_argmax` | 1, 1, 2, 2 | You need the probability matrix, or only some of the results |
| `adjusted_argmax` | 2 | Selection alone. The default; its latency stayed flat up to 1,048,576 nodes |
| `adjusted_argmax_single_block` | 1 | Selection alone, at most 16,384 nodes, when about 10 µs matters and you want no partial buffers |

Each method has an `_async` form that enqueues on the accelerator stream.
Synchronize before reading or reusing its buffers. An error after the first
launch of a multi-pass method can leave that launch queued.

## Contract

The table in the [`routing` module docs](../src/routing.rs) is normative. In short:

- **Logits:** NaN ranks as `-inf`; `+inf` logits share all of the row's mass;
  a row with nothing above `-inf` is uniform.
- **Probabilities:** non-finite and negative values become 0. Rows are not
  renormalized, and values at or below `1e-8` add no entropy.
- **Order:** higher value first; equal values keep the lower index. Top-k ranks
  input values, so `exp` rounding and underflow cannot reorder routes. Slots
  past `n_routes` are `-1`, with probability 0.
- **Selection:** the score is single-rounded (`fmaf` on the device,
  `f32::mul_add` on the host). NaN and `-inf` scores are ineligible. With no
  eligible node, the index is `NO_SELECTION` (`u32::MAX`) and the score is `-inf`.
- **Empty inputs:** `n_nodes == 0` gives entropy 0 and no selection.
  `n_routes == 0` gives entropy 0 and all-`-1` top-k, and selection still runs.
- **Invalid inputs:** these return `GpuError::InvalidInput` before any launch:
  a dimension above `i32::MAX`, `top_k` outside `1..=8`, a non-finite
  `adaptation_scale`, or a buffer that is too short. `state` and `adaptation`
  must be the same length for the `adjusted_argmax` methods.

**Host and device agreement.** Indices and selected scores match exactly unless
a value is subnormal: the device build flushes subnormals to zero and the
reference keeps them. Probabilities agree within `PROBABILITY_ABS_TOL` and
entropy within `ENTROPY_ABS_TOL` per node, because the device uses fast
`exp`/`log2` and a different summation order.

The fused path computes logit-row entropy from the normalizer without a
per-route cutoff. `routing_entropy_reduce` skips probabilities at or below
`1e-8`. The two results agree within `ENTROPY_ABS_TOL` per node.

## Launch constraints and workspace

- Every launch uses 256 threads per block. Pass-1 launches use
  `grid = ceil(n_nodes / 256)`; pass-2 launches use `<<<1, 256>>>`.
- Pass 2 strides over its partials, so block counts above 32 reduce correctly.
- The accelerator owns partial buffers of `ceil(n_nodes / 256)` elements:
  16 bytes per block across four buffers. They grow on demand and do not shrink.
  Growth synchronizes the stream first.
- Allocate outputs with `FusedRoutingOutputs::alloc`. Longer buffers are
  accepted; only the required prefix is written.
- Omitting `top_k_probabilities` skips those writes. Results are otherwise
  identical.
- `GpuAccelerator` is not `Sync`; use one accelerator per thread.

## Fallback

There is no implicit CPU execution. On a CPU-backend accelerator, or a build
without `cuda`, valid calls return `GpuError::Unavailable` and invalid calls
return `GpuError::InvalidInput`. A caller that wants CPU execution calls
`routing::routing_reference` or `routing::adjusted_argmax` explicitly. They
follow the same contract.

## Analytical estimates

**These figures are not measurements.** `routing::analytical` counts bytes that
kernel code requests from global memory, including repeated reads of a row.
Caches absorb part of this before DRAM. Upload and download are excluded.

The composable path reads a logit matrix five times (softmax three, top-k one,
entropy one) and writes it once. The fused path reads it twice and never writes
it.

| n_nodes | n_routes | k | Composable bytes | Fused bytes | Fewer | Launches |
|---------|----------|---|------------------|-------------|-------|----------|
| 64 | 8 | 2 | 13,360 | 5,168 | 61.3% | 6 → 2 |
| 256 | 64 | 4 | 399,408 | 137,264 | 65.6% | 6 → 2 |
| 2,048 | 16 | 4 | 835,856 | 311,568 | 62.7% | 6 → 2 |
| 1,024 | 128 | 4 | 3,170,448 | 1,073,296 | 66.1% | 6 → 2 |
| 256 | 4,096 | 8 | 25,176,112 | 8,398,896 | 66.6% | 6 → 2 |
| 4,096 | 1,024 | 8 | 100,827,664 | 33,718,800 | 66.6% | 6 → 2 |
| 65,536 | 16 | 4 | 26,746,896 | 9,969,680 | 62.7% | 6 → 2 |

### Occupancy estimate

Register counts below are from `ptxas -arch=sm_120 -v`. The limits are from the
[Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html):
48 warps per SM on compute capability 12.0, which is 1,536 threads or 6 blocks
of 256, and a 64K register file. A kernel at or below 42 registers per thread
can therefore keep all 6 blocks resident, ignoring allocation granularity.

| Kernel | Registers | Stack | Shared |
|--------|-----------|-------|--------|
| `fused_routing_pass1` | 38 | 64 B | 512 B |
| `fused_routing_pass2` | 40 | 0 | 512 B |
| `routing_softmax` | 36 | 0 | 0 |
| `routing_top_k` | 34 | 64 B | 0 |
| `routing_entropy_reduce_pass1` | 36 | 0 | 256 B |
| `latent_reduce_pass2` | 16 | 0 | 256 B |
| `adjusted_argmax_pass1` | 14 | 0 | 256 B |
| `indexed_argmax_reduce_pass2` | 32 | 0 | 256 B |
| `adjusted_argmax_single_block` | 40 | 0 | 256 B |

No kernel is register-limited by this estimate, and register use does not
depend on `n_routes`. Occupancy is then limited by the grid: filling the 84 SMs
of an RTX 5080 (10,752 cores at 128 per SM) takes about 129,024 nodes, or 504
blocks. A 2,048-node launch is 8 blocks, about 1.6% of the device's thread
slots. Each pass-2 launch and the single-block path occupy one block.

The top-k arrays are on the per-thread stack (64 B), not in registers.

## Measured results

Measured on one RTX 5080 at implementation commit
`c4f01862e21b2bfe9b425fb6acbdfdf88079d5ac`, clean tree, under WSL2 on Windows 11:
driver 617.14, CUDA toolkit 13.3 (V13.3.73), rustc 1.99.1, Ryzen 9 9950X.

Each cell is the median of three runs' p50, with the range across runs. Each run
used 50 warmup and 300 measured iterations, via
`cargo run --locked --example benchmark --profile bench --features bench,cuda -- --warmup 50 --iterations 300`.
Time is host wall-clock from the first launch through `synchronize`.

The GPU was also driving the Windows desktop, clocks were not locked, and a bare
launch plus synchronize costs about 30 µs here (`poisson_encode_4096` p50:
33–35 µs). Treat small differences as approximate and re-measure on your host.

### Fused and composable routing

| n_nodes | n_routes | k | Fused p50 µs | Fused + probabilities p50 µs | Composable p50 µs | Composable / fused |
|---------|----------|---|--------------|------------------------------|-------------------|--------------------|
| 64 | 8 | 2 | 48.2 (43.4–56.8) | 48.7 (42.5–64.3) | 78.0 (76.7–80.1) | 1.62 |
| 256 | 64 | 4 | 72.7 (70.9–82.3) | 69.0 (67.0–71.6) | 113.6 (111.0–114.1) | 1.56 |
| 2,048 | 16 | 4 | 47.0 (46.8–49.6) | 47.3 (47.2–50.6) | 70.6 (70.5–72.5) | 1.50 |
| 1,024 | 128 | 4 | 92.1 (87.7–103.1) | 98.2 (86.7–110.1) | 165.9 (163.9–184.6) | 1.80 |
| 256 | 4,096 | 8 | 1,221.8 (1,221.5–1,259.4) | 1,227.8 (1,205.1–1,255.3) | 3,162.8 (3,153.8–3,194.0) | 2.59 |
| 4,096 | 1,024 | 8 | 425.1 (421.0–433.7) | 438.5 (431.3–439.6) | 910.6 (897.3–932.7) | 2.14 |
| 65,536 | 16 | 4 | 89.9 (79.2–113.1) | 86.6 (82.9–88.0) | 124.6 (118.2–131.5) | 1.39 |

### Two-pass and single-block selection

| n | Two-pass p50 µs | Single-block p50 µs |
|---|-----------------|---------------------|
| 64 | 55.8 (53.5–77.4) | 43.2 (41.4–54.9) |
| 256 | 49.5 (39.8–53.6) | 39.1 (34.0–41.5) |
| 512 | 49.6 (47.5–54.6) | 38.8 (34.0–40.2) |
| 1,024 | 49.7 (46.4–50.8) | 35.9 (34.9–38.4) |
| 2,048 | 46.4 (44.7–48.9) | 37.2 (33.9–45.0) |
| 4,096 | 49.8 (45.3–50.9) | 40.4 (37.2–57.4) |
| 8,192 | 52.5 (47.4–56.6) | 41.1 (36.0–48.6) |
| 16,384 | 48.4 (39.7–54.9) | 40.9 (40.8–41.9) |
| 65,536 | 42.4 (36.0–58.6) | 57.2 (56.6–67.7) |
| 262,144 | 52.3 (50.9–57.6) | 135.5 (120.3–147.4) |
| 1,048,576 | 50.5 (45.3–50.7) | 369.4 (366.2–371.3) |

### Compute Sanitizer memcheck

`./scripts/sanitize_lifecycle.sh` ran on the same RTX 5080 under WSL2 with
driver 621.14, CUDA toolkit 13.3 (V13.3.73), and Compute Sanitizer 2026.2.1.0.
Kernel, wrapper, and test sources were unchanged from
`b7408445f4c5bec1ff68736987d9f1f6d5f33ba6`. Each suite ran in its own serial
memcheck process with `--include-ignored`:

| Suite | Passed | Failed | Ignored | Memcheck errors |
|-------|--------|--------|---------|-----------------|
| `gpu_lifecycle` | 8 | 0 | 0 | 0 |
| `snn_fixtures_gpu` | 2 | 0 | 0 | 0 |
| `routing_gpu` | 13 | 0 | 0 | 0 |

Earlier attempts on this host failed to attach (`Failed to initialize WDDM
debugger interface`). Attachment depends on host configuration, so re-run the
script on each qualification host.

### Not measured

Nsight Compute returned `ERR_NVGPUCTRPERM` on this host, including a retry with
driver 621.14, so **occupancy, DRAM traffic, and cache behavior have no hardware
evidence**. The occupancy figures above are estimates only.

## Findings

**When fusion helps.** At every measured shape, fusion was faster: 1.39× to
2.59× by p50. The gain was largest with wide rows, where memory work dominates
(256 × 4,096). At small shapes the absolute gain was about 25–40 µs, which
is consistent with launch cost, though these runs do not separate the two.
Fusion does not help a caller who needs the probability matrix.

**Occupancy.** By the estimate above, registers do not limit any routing size.
Node count decides how much of the device a launch can occupy, and `n_routes`
does not change register use. This has not been confirmed with hardware counters.

**Reductions on the device.** Entropy sum and maximum, top-k, and selection all
stay on the device. Only four scalars and the top-k output need a host copy.
Writing top-k probabilities had no cost distinguishable from noise.

**Single-block or two-pass.** Single-block was faster by roughly 8–14 µs up to
16,384 nodes. Two-pass was faster at 65,536 and above, and its latency stayed
flat through 1,048,576 nodes while single-block grew with `n`. The crossover is
between 16,384 and 65,536; those sizes between were not measured. Both paths
return identical results, so this is a latency choice only. The library keeps
the choice explicit, with two-pass as the default, because one host's small
differences do not justify a built-in threshold.

## Reproducing

```bash
export CUDA_NVCC=/usr/local/cuda/bin/nvcc
cargo test --locked --features cuda --test routing_gpu -- --ignored --test-threads=1
cargo run --locked --example benchmark --profile bench --features bench,cuda
ptxas -arch=sm_120 -v -o /tmp/routing.cubin "$(./scripts/ptx_out_dir.sh)/routing_selection_sm_120.ptx"
```

Where GPU performance counters are available, measure occupancy with:

```bash
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active \
  --kernel-name 'regex:routing|argmax' \
  cargo run --locked --example benchmark --profile bench --features bench,cuda
```

The mandatory `./scripts/sanitize_lifecycle.sh` gate includes `routing_gpu`
alongside the lifecycle and SNN fixture suites. Each suite must execute tests
and report zero memcheck errors.
