# Myelin-Accelerator

[![CI](https://github.com/Limen-Neural/myelin-accelerator/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Limen-Neural/myelin-accelerator/actions/workflows/ci.yml)
[![GPU CI](https://img.shields.io/github/check-runs/Limen-Neural/myelin-accelerator/main?nameFilter=CUDA%20build%20%5Bself-hosted%5D%20%28sm_120%29&label=GPU%20CI)](https://github.com/Limen-Neural/myelin-accelerator/actions/workflows/ci.yml)
[![CodeRabbit Reviews](https://img.shields.io/coderabbit/prs/github/Limen-Neural/myelin-accelerator?utm_source=oss&utm_medium=github&utm_campaign=Limen-Neural%2Fmyelin-accelerator&labelColor=171717&color=FF570A&label=CodeRabbit+Reviews)](https://coderabbit.ai/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Limen-Neural/myelin-accelerator)
[![Maintainability](https://qlty.sh/gh/Limen-Neural/projects/myelin-accelerator/maintainability.svg)](https://qlty.sh/gh/Limen-Neural/projects/myelin-accelerator)
[![CodeScene Average Code Health](https://codescene.io/projects/85266/status-badges/average-code-health)](https://codescene.io/projects/85266)
[![CodeScene Hotspot Code Health](https://codescene.io/projects/85266/status-badges/hotspot-code-health)](https://codescene.io/projects/85266)
[![CodeScene System Mastery](https://codescene.io/projects/85266/status-badges/system-mastery)](https://codescene.io/projects/85266)
[![Analyzed by CodeScene](https://codescene.io/images/analyzed-by-codescene-badge.svg)](https://codescene.io/projects/85266)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](#license)
[![Release: v0.2.0 candidate](https://img.shields.io/badge/release-v0.2.0%20candidate-orange)](https://github.com/Limen-Neural/myelin-accelerator/issues/37)

Reusable Rust/CUDA acceleration primitives for SNN / neuromorphic workloads: spiking networks, routing, SAT search, and packed ternary matrix operations. The CUDA path targets `sm_120` (Blackwell); the default build works without a CUDA toolkit or device.

**Release status:** `Cargo.toml` is at `0.2.0` as a crates.io candidate (not yet published). Qualification is tracked in [release issue #37](https://github.com/Limen-Neural/myelin-accelerator/issues/37); publication after qualification is owned by LIM-1460. Cargo/crates.io is the supported distribution path; Docker/container images are not a supported consumer surface. The CI and GPU CI badges reflect `main`, not this checkout or an open pull request.

## Get started

After v0.2.0 publication, add the crate with:

```bash
cargo add myelin-accelerator
# For GPU launches: cargo add myelin-accelerator --features cuda
```

Prior to publication, use a source checkout at a reviewed commit (Git dependency or Cargo `path` dependency). For a local checkout:

```toml
[dependencies]
myelin-accelerator = { path = "../myelin-accelerator" }
# For GPU launches, add: features = ["cuda"]
```

Host packing works with the default feature set:

```rust
use myelin_accelerator::bitpacking::{pack_ternary, unpack_ternary};

let values = [-1, 0, 1, 1];
let packed = pack_ternary(&values);
assert_eq!(unpack_ternary(&packed, Some(values.len())), values);
```

`GpuAccelerator::new()` may select a CPU backend and records the fallback reason. That backend does **not** run CPU versions of GPU launch methods; those methods return `GpuError::Unavailable`. For a GPU-required application, use `GpuAccelerator::require_gpu()` and handle the error. See [architecture and capability policy](docs/ARCHITECTURE.md#capability-probe-and-fallback-policy).

## What is available

| Area | Current interface |
| --- | --- |
| Host utilities | Binary and ternary packing, group scales, reference matmul in [`bitpacking`](src/bitpacking.rs); scalar test oracles in [`oracle`](src/oracle.rs). |
| Wrapped CUDA launches | Poisson encoding, STDP, SAT result extraction and auxiliary best reduction, ternary GEMV and GEMM through `GpuAccelerator`. |
| Loaded CUDA kernels | Additional LIF, spike statistics, routing, SAT walker, and reduction symbols are available through `KernelModule`, but do not all have high-level launch wrappers. |
| Benchmark support | Optional `bench` example and manifest utilities; see [benchmark instructions](docs/BENCHMARKS.md). |

The [architecture guide](docs/ARCHITECTURE.md#high-level-launches-today-gpuaccelerator) lists exact wrapper and device symbol coverage. [Ternary layout and kernel contracts](docs/TERNARY.md) describe the packed data format. The STDP wrapper launches its weight and trace kernels in stream order so each trace index has one writer.

## Build and verify

CPU-safe checks need only the Rust toolchain:

```bash
cargo test --locked
cargo build --locked --no-default-features
```

For the CUDA path, use a CUDA 13.2+ toolkit with `nvcc` and an `sm_120` capable GPU for device execution:

```bash
CUDA_NVCC=/usr/local/cuda/bin/nvcc cargo build --locked --features cuda
CUDA_NVCC=/usr/local/cuda/bin/nvcc cargo test --locked --features cuda -- --ignored --nocapture
```

The ignored tests require a working CUDA device. CI also compiles and assembles PTX without a GPU, then runs device tests on a self-hosted GPU runner. See [REVIEW.md](REVIEW.md) §6–§7 for the full local quality gate.

## Citation

```bibtex
@software{myelin_accelerator,
  title  = {Myelin-Accelerator},
  author = {Raul Montoya Cardenas},
  year   = {2026},
  url    = {https://github.com/Limen-Neural/myelin-accelerator}
}
```

A formal citation is optional.

## License

Choose either [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT).
