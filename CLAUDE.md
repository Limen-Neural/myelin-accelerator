# CLAUDE.md

Guidance for Claude (and other AI coding agents) working in this repository.

## Boundaries (what not to do)

- Do **not** enable CMake's native `CUDA` language (`project(... CUDA)`) on
  this host — use the existing CXX-only + `nvcc -ptx` custom targets.
- Do **not** use CLion "New Target" / `add_executable(... .cu)`; use the
  `cuda_kernels` CMake target for PTX quality.
- Do **not** re-add `CMakeLists.txt` to `.gitignore`.
- Do **not** create a new CMake build directory; reuse `cmake-build-debug`
  with the **Ninja** generator (not Unix Makefiles).
- Do **not** use `cargo bench` as the GPU quality gate (no `[[bench]]` here).
- Do **not** force-rewrite PTX `.version` to an exact value in CI; let
  `build.rs` floor Blackwell to ≥ 9.2 and leave nvcc's header otherwise.
- Do **not** declare `nvtx` as a non-optional dependency; it belongs only on
  the `cuda` feature.
- Prefer not to weaken or `#[ignore]` a failing test solely to green CI.
  GPU-only tests use `#[ignore] // requires GPU + driver ≥ 570`.

## Project

`myelin-accelerator` is a Rust crate exposing safe FFI wrappers around a set
of first-party CUDA kernels (spiking-network simulation, vector similarity
search, and a SAT solver), targeting Blackwell / RTX 5080-class GPUs
(`sm_120`). It builds cleanly with or without a CUDA toolkit present via the
`cuda` Cargo feature:

- **Default (`cuda` feature off):** `src/gpu_stub.rs` is used, `build.rs`
  writes stub PTX files, no GPU/nvcc required. This is the CPU-safe path and
  is intended to work in CI and sandboxed environments without a GPU.
- **`--features cuda`:** `src/gpu/` is compiled, `build.rs` invokes `nvcc` to
  compile `cu/*.cu` → PTX, embedded at compile time via `include_str!`.
  Prefer **CUDA toolkit 13.2+**; ShipOfTheseus local default is **13.3.1**
  (`/usr/local/cuda` → `cuda-13.3`). Override with `CUDA_NVCC` /
  `CUDA_HOME` only if you intentionally use an older tree. Runtime needs
  an `sm_120`-capable driver (≥ 570; UMD **13.x** on current Blackwell hosts).

## Build and test

Run CPU-safe checks first — these should pass without a GPU:

```bash
cargo test --locked
cargo build --locked --no-default-features
```

If a CUDA toolkit is available, also verify the GPU path (set `CUDA_NVCC` if
`nvcc` isn't on `PATH`):

```bash
CUDA_NVCC=/usr/local/cuda/bin/nvcc cargo build --locked --features cuda
```

### CMake / CLion integration

`CMakeLists.txt` wires `nvcc` through custom commands (not CMake's native
CUDA language — see the CUDA/CMake write-up in the repository root) so CLion
can drive both the CUDA PTX compilation and the Cargo test suite via CTest:

```bash
cmake --build cmake-build-debug --target cuda_kernels
ctest --test-dir cmake-build-debug --output-on-failure
```

CTest covers: `cargo_tests`, `cargo_build_no_default_features`,
`cargo_fmt_check`, `cargo_clippy_no_default`, `cargo_build_bench_example`,
`cuda_kernel_build`.

GPU kernel benchmarks need **both** features (not just `bench`):

```bash
CUDA_NVCC=/usr/local/cuda/bin/nvcc \
  cargo run --example benchmark --profile bench --features bench,cuda
```

### Local CUDA GPU quality gate (preferred over cloud)

Full GPU proof is **local / self-hosted** (Blackwell + driver). Cloud runners
rarely have `sm_120`. Quick local set:

```bash
export CUDA_NVCC=/usr/local/cuda/bin/nvcc
cargo build --locked --features cuda
# Offline assemble needs a .ptx file (not bare -arch):
#   ptxas -arch=sm_120 -o /tmp/sn.cubin cmake-build-debug/spiking_network.ptx
cargo test --locked --features cuda
cargo test --locked --features cuda -- --ignored --nocapture
cargo run --locked --example benchmark --profile bench --features bench,cuda
```

## Conventions

- `CMakeLists.txt` is tracked in git; generated CMake artifacts stay ignored.
- Keep `nvcc` flags in `CMakeLists.txt` and `build.rs` in sync
  (`-std=c++17`, `-D__STRICT_ANSI__`, `--allow-unsupported-compiler`,
  `--expt-relaxed-constexpr`, `-Xcompiler -fno-builtin`).
- `nvtx` (`range_push!`/`range_pop!`) is optional under
  `cuda = ["dep:cust", "dep:nvtx"]` so the CPU-safe path does not link NVTX.

## See also

- Repository root CUDA/CMake quality-gate notes (companion agent primer).
- `AGENTS.md` — brief agent build/test primer.
