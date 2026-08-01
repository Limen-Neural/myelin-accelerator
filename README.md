# Myelin-Accelerator

Blackwell-first CUDA kernels for neuromorphic inference, SAT search, and routing-heavy GPU workloads on RTX 5080-class hardware.

This repo is the **low-level compute layer** behind the stack: CUDA PTX modules, Rust bindings, and the launch paths that keep the GPU busy instead of serializing work through one thread at a time.

**Backend scope and public API:** see **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — what belongs here vs higher-level experiment/orchestration repos.

## Current State

- `sm_120` target: the kernels are built and tuned for RTX 5080 / Blackwell.
- `16 GB VRAM` discipline: kernel footprints are kept static and bounded; the big memory costs are in user-managed tensors, not temporary scratch.
- `SAT path`: `atomicMin` is gone from the hot reduction path.
- `Routing path`: `cosine_similarity_top_k` now uses warp-participating top-k reduction instead of a single-thread selection tail.
- `Rust FFI`: kernel symbols are loaded through `src/gpu/kernel.rs`; the codebase stays ABI-consistent with the CUDA side.
- Host **binary/ternary bitpacking** + group scales + CPU ref matmul: `src/bitpacking.rs`.
- Device **ternary GEMV/GEMM** (group-scaled, optional zero-skip): `cu/ternary_gemm.cu` — see [docs/TERNARY.md](docs/TERNARY.md).

## Module map

| Path | Role | Public? |
|------|------|---------|
| `src/lib.rs` | Crate root re-exports | yes |
| `src/bitpacking.rs` | Host binary/ternary pack/unpack, scales, ref GEMV/GEMM | yes (`bitpacking`) |
| `src/gpu/` | CUDA context, PTX load, buffers, launches | via re-exports when `cuda` |
| `src/gpu_stub.rs` | CPU-safe stand-ins without toolkit | used when `cuda` off |
| `cu/*.cu` | Device kernels (spiking, similarity, SAT, ternary) | via PTX + wrappers |
| `examples/benchmark.rs` | Latency / GPU info harness | feature `bench` (+ `cuda` for GPU) |
| `build.rs` / `CMakeLists.txt` | `nvcc -ptx` quality path | build-only |
| `docs/ARCHITECTURE.md` | Ownership + API boundary | docs |
| `docs/TERNARY.md` | Ternary encoding, scales, GOZ1 interop, kernels | docs |

### Features

| Feature | Meaning |
|---------|---------|
| *(default)* | Stub GPU API; no `nvcc` |
| `cuda` | Real GPU path (`cust`, `nvtx`) |
| `bench` | Benchmark example serde deps |

### Public symbols (crate root)

`GpuAccelerator`, `GpuContext`, `GpuBuffer`, `KernelModule`, `GpuError` — plus the `bitpacking` module. Prefer these over deep `gpu::…` paths. Full list of loaded device symbols and what stays out of this repo is in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## What Changed

- `satsolver.cu` now reduces `(score, walker)` with warp shuffles and shared memory, then writes one result per block before a final reduction pass.
- `vector_similarity.cu` now keeps top-k selection parallel across the warp, with register-resident candidates and block-local merge stages.
- `.gitignore` now excludes generated PTX, build outputs, CMake artifacts, IDE files, and other local-only clutter.

## Why It Matters

- Better SM occupancy on Blackwell.
- No global atomic serialization in the SAT best-score path.
- No single-thread MoE routing bottleneck.
- Clear boundary so consumers (SNN stacks, quant prototypes) depend on one accelerator crate instead of copying kernels.


## Usage

```toml
[dependencies]
myelin-accelerator = "0.2.0"
# Optional GPU:
# myelin-accelerator = { version = "0.2.0", features = ["cuda"] }
```

```rust
use myelin_accelerator::{GpuAccelerator, bitpacking};

let gpu = GpuAccelerator::new();
if gpu.is_ready() {
    // launch wrappers when a device is present
}
let packed = bitpacking::pack_ternary(&[-1, 0, 1, 1]);
let _ = packed;
```

CPU-safe checks:

```bash
cargo test --locked
cargo build --locked --no-default-features
```

## Citation

```bibtex
@software{myelin_accelerator,
  title  = {Myelin-Accelerator},
  author = {Raul Montoya Cardenas},
  year   = {2026},
  url    = {https://github.com/Limen-Neural/myelin-accelerator}
}
```
No formal citation required — use freely under the Apache 2.0 or MIT license. A link back to this repo is appreciated but not mandatory.
## License

Licensed under either of:

- [Apache License, Version 2.0](LICENSE-APACHE)
- [MIT License](LICENSE-MIT)

at your option.
