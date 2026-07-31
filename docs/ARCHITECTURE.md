# Architecture — backend scope and public API

This document is the ownership map for `myelin-accelerator`. Use it to decide
whether a proposed CUDA / SNN / quantization feature belongs **here** or in a
higher-level repo (`corinth-canal`, experiment harnesses, model code).

Related issues: [GH #8](https://github.com/Limen-Neural/myelin-accelerator/issues/8)
(closed). **Owner:** [Limen-Neural/myelin-accelerator](https://github.com/Limen-Neural/myelin-accelerator)
only — do not track this crate under personal `rmems/*` remotes or deps.

---

## One-line mission

**Low-level, reusable GPU compute for neuromorphic / routing / SAT workloads:**
first-party CUDA sources in `cu/` (compiled to PTX), safe Rust FFI, device memory helpers, and a
local quality/benchmark harness. Not a research orchestrator.

---

## Belongs in this repo

| Area | Examples |
|------|----------|
| CUDA kernels | `cu/*.cu`, shared headers (`cu/common.cuh`), `sm_120`-tuned reductions |
| PTX build path | `build.rs` (`nvcc -ptx`), CMake `cuda_kernels` target, embedded PTX |
| Safe GPU FFI | `GpuContext`, `KernelModule`, `GpuBuffer`, launch wrappers |
| Feature gates | `cuda` (cust + nvtx), `bench` (serde JSON/CSV harness) |
| CPU-safe stub | `src/gpu_stub.rs` when `cuda` is off (CI / sandboxes) |
| Host packing utilities | Binary / ternary bitpacking (`src/bitpacking.rs`) — host-side layout helpers that match future device kernels |
| Launch / stream helpers | Default streams, aux buffers for multi-pass reductions |
| Local QA | CTest matrix, GPU `#[ignore]` tests, `examples/benchmark.rs` |
| Kernel-level telemetry primitives | On-device reduce passes (entropy, membrane stats) that stay generic |

**Rule of thumb:** if another project would copy a `.cu` file or re-implement
`cust` load/launch to get the same primitive, it should live here instead.

---

## Does **not** belong here

| Area | Put it in… |
|------|------------|
| Model-specific prompts, chat loops, agent policies | Application / product repos |
| Dataset download, eval harnesses, leaderboards | Experiment / MLOps repos |
| High-level SAAQ experiment orchestration, recipes, manifests | e.g. `corinth-canal`, `magere-brug` |
| Training loops, optimizers, full SNN training frameworks | Spikenaut / plasticity / training crates |
| Cloud infra, multi-tenant runners (except this repo’s CI workflow) | Infra docs / org runners |
| UI, dashboards, visualization of results | Viz / notebook projects |
| Copying kernels into consumer trees “for convenience” | Depend on this crate instead |

**Rule of thumb:** if the change is “run experiment X on dataset Y and log
business metrics,” it is **not** a myelin feature.

---

## Expected consumers

| Consumer | Typical dependency |
|----------|-------------------|
| `corinth-canal` | Routing / SAAQ-related GPU primitives; stay out of experiment orchestration |
| SNN libraries / Spikenaut-class stacks | Poisson encode, LIF step, rate stats, STDP-style updates when exposed |
| Telemetry / profiling experiments | Kernel launches + nvtx ranges; host only aggregates results |
| Quantization prototypes (ternary / SAAQ) | Packed layouts + future GEMV/GEMM kernels; host packing already here |
| SAT / search research | `satsolver_*` path and aux reduce |
| Future Metis / neuromorphic prototypes | Thin dependency on public Rust API + features |

Consumers should depend on **published crate surface** (`lib.rs` re-exports and
documented modules), not on private `src/gpu/*` internals or raw `OUT_DIR` PTX
paths.

---

## Module map

```text
myelin-accelerator/
├── cu/                          # First-party CUDA device sources
│   ├── common.cuh
│   ├── spiking_network.cu       # Poisson, LIF, STDP, reduce passes
│   ├── vector_similarity.cu     # Cosine batched + top-k routing
│   ├── satsolver.cu             # Parallel SAT walkers + reduces
│   └── ternary_gemm.cu          # Group-scaled ternary GEMV / GEMM
├── src/
│   ├── lib.rs                   # Crate root; public re-exports
│   ├── bitpacking.rs            # Host binary/ternary pack/unpack + scales/ref
│   ├── gpu_stub.rs              # CPU-safe stand-ins (no cuda feature)
│   └── gpu/                     # Real CUDA path (feature = "cuda")
│       ├── mod.rs               # Internal module tree + re-exports
│       ├── context.rs           # Device / primary context
│       ├── kernel.rs            # PTX embed + Module/Function map
│       ├── memory.rs            # GpuBuffer
│       ├── error.rs             # GpuError / GpuResult
│       └── accelerator.rs       # High-level launch wrappers
├── examples/benchmark.rs        # Optional bench harness (feature = "bench")
├── build.rs                     # nvcc → PTX into OUT_DIR
├── CMakeLists.txt               # CLion/CTest quality gate (nvcc -ptx)
└── docs/
    ├── ARCHITECTURE.md          # This file
    └── TERNARY.md               # Ternary encoding, scales, GOZ1, kernels
```

### Cargo features

| Feature | Effect |
|---------|--------|
| *(default empty)* | Stub GPU API; no `nvcc` required |
| `cuda` | Real `src/gpu/*`, `cust`, optional `nvtx` profiling ranges |
| `bench` | Serde deps for `examples/benchmark` (pair with `cuda` for GPU) |

---

## Public API vs internal modules

### Stable public surface (crate root)

Re-exported from `src/lib.rs` (names available with or without `cuda` via stub):

| Symbol | Role |
|--------|------|
| `GpuAccelerator` | Primary entry: construct, readiness, kernel launches |
| `GpuContext` | Context init / presence |
| `GpuBuffer` | Device buffer helper |
| `KernelModule` | Loaded PTX modules + `get_function` |
| `GpuError` | Error type re-exported at the crate root |
| `bitpacking` module | Host packing APIs (`pack_ternary`, `pack_binary`, …) |

`GpuResult<T>` (`type` alias for `Result<T, GpuError>`) is **not** re-exported
from the crate root today. Use `Result<_, myelin_accelerator::GpuError>` at the
boundary, or `myelin_accelerator::gpu::GpuResult` if you want the alias (via the
`gpu` / stub module path). Prefer
`use myelin_accelerator::{GpuAccelerator, GpuError, …}` over deep paths into
internal files.

### High-level launches today (`GpuAccelerator`)

These are the **ergonomic** wrappers currently implemented:

- Lifecycle: `new`, `is_ready`, `kernels`, `synchronize`
- SAT: `satsolver_extract` / `_async`, `satsolver_aux_reduce_best` / `_async`
- Spiking: `poisson_encode` / `_async`
- Ternary quant matmul: `ternary_gemv` / `_async`, `ternary_gemm` / `_async` (see [TERNARY.md](TERNARY.md))

Additional kernels may be **loaded** in `KernelModule` and still lack a
dedicated `GpuAccelerator` method. Advanced callers can use
`accelerator.kernels()?.get_function("…")` and launch with `cust` while we
grow the wrapper surface. Growing a stable method is preferred for anything
consumers share.

### Device kernels currently registered (`KernelModule::load`)

| PTX module | Symbols |
|------------|---------|
| `spiking_network` | `poisson_encode`, `lif_step`, `lif_step_weighted`, `spike_rate`, `reset_membrane`, `stdp_update`, `neuro_bias_logits`, `membrane_dv_dt_reduce_pass1`, `routing_entropy_reduce_pass1`, `latent_reduce_pass2` |
| `vector_similarity` | `cosine_similarity_batched`, `cosine_similarity_top_k` |
| `satsolver` | `satsolver_init`, `satsolver_step`, `satsolver_aux_update`, `satsolver_check_solution`, `satsolver_extract`, `satsolver_best_reduce_pass1`, `satsolver_best_reduce_pass2` |
| `ternary_gemm` | `ternary_gemv`, `ternary_gemm` |

### Internal (not a stability promise)

- `src/gpu/*` private details, aux buffer caching, launch grid heuristics
- `build.rs` env vars and PTX version flooring (Blackwell ≥ 9.2)
- Generated `OUT_DIR/*_sm_120.ptx` and CMake build trees
- Agent/IDE/tooling files (`.opencode/`, `.swarm/`, worktrees)

---

## Build and quality boundaries

| Path | Purpose |
|------|---------|
| CPU CI | `cargo test --locked`, `cargo build --no-default-features` |
| GPU local / self-hosted | `cargo test --features cuda -- --ignored`, benchmark example |
| CLion | CMake CXX-only + `nvcc -ptx` custom target — **not** CMake native `CUDA` language |

See `CLAUDE.md`, `AGENTS.md`, and `REVIEW.md` for command recipes and hard
“do not” rules (CMake generator, PTX rewrite, nvtx feature placement).

---

## Decision checklist (paste into PRs / issues)

When proposing a feature, answer:

1. Is this a **reusable kernel**, packing layout, or launch primitive? → **yes = myelin**
2. Does it require **model weights, datasets, or experiment config**? → **higher-level repo**
3. Would **two consumers** copy this code if we refuse it? → **yes = myelin**
4. Is it **fused orchestration** of many kernels with product policy? → split: keep fusion **kernel** here only if generic; policy stays out
5. Does it change the **public API**? → document in this file + rustdoc

### Examples

| Proposal | Decision |
|----------|----------|
| Packed ternary GEMV/GEMM on `sm_120` | **Here** (GH #9) — after encoding + correctness tests |
| Fuse routing + SAAQ selection reduce | **Here only as reusable kernel** (GH #14); experiment loops elsewhere |
| “Run SAAQ recipe on GOZ1 artifacts” | **Not here** |
| Add `cosine_similarity_top_k` wrapper method | **Here** — public API growth |
| Train LIF network with backprop | **Not here** |

---

## Future work that stays in scope

Tracked elsewhere but **in-boundary** if they stay low-level:

- Packed ternary device kernels `ternary_gemv` / `ternary_gemm` (GH #9 / [LIM-890](https://linear.app/rpd-34/issue/LIM-890)) — host + device live in `src/bitpacking.rs`, `cu/ternary_gemm.cu`, `docs/TERNARY.md`
- Fused routing / SAAQ kernels (GH #14 / [LIM-891](https://linear.app/rpd-34/issue/LIM-891)) — only generic device code + benches
- More `GpuAccelerator` wrappers for already-loaded symbols
- Wider public surface for bitpacking + device kernel parity docs

---

## Maintenance

Update this file when:

- A new public re-export is added or removed
- A new `cu/*.cu` module is introduced
- Ownership of a feature is disputed between myelin and a consumer

Do not use OpenCode `.swarm/` runtime directories for architecture prose — keep
durable docs under `docs/` and issue/PR comments.
