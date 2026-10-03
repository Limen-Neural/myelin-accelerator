# AGENTS.md

## Project and current scope

`myelin-accelerator` is a Rust crate for Blackwell / RTX 5080-class (`sm_120`) CUDA primitives. Default features are empty: the CPU/stub build needs no CUDA toolkit or GPU. `--features cuda` compiles first-party `cu/*.cu` kernels to PTX and embeds them in the Rust library. CUDA toolkit 13.2+ is supported; the local default is 13.3 through `/usr/local/cuda`.

This is a compute layer, not a full SNN framework. The pinned Spikenaut and Synfire fixtures are **workload-only** tests; they do not establish full-model compatibility. See `docs/SNN_COMPATIBILITY.md` and the deferred interoperability issue GH #43 / LIM-1461 before making a model-support claim.

## Build and test in order

Run CPU-safe checks first:

```bash
cargo test --locked
cargo build --locked --no-default-features
cargo fmt --check
cargo clippy --locked --no-default-features -- -D warnings
```

The cloud CI job covers CPU tests, formatting, lint, and PTX compilation/assembly. CPU coverage on Qlty covers the stub build and benchmark harness; it is not GPU coverage. A green cloud-only check does not prove device execution.

On an `sm_120` device with a matching driver and Compute Sanitizer, run the CUDA path:

```bash
export CUDA_NVCC=/usr/local/cuda/bin/nvcc
cargo build --locked --features cuda
cargo test --locked --features cuda
cargo test --locked --features cuda -- --ignored --nocapture --test-threads=1
./scripts/sanitize_lifecycle.sh
```

The sanitizer script runs `gpu_lifecycle` and `snn_fixtures_gpu` serially under unsuppressed memcheck. Require executed tests and a zero-error summary for **each** suite. See `REVIEW.md` §6–§7 for the complete local gate, including PTX, CMake/CTest, benchmark, lint, rustdoc, and the full device sanitizer matrix. Keep CMake on its CXX custom `nvcc -ptx` path; do not enable CMake's native CUDA language for this host. `CLAUDE.md` has the host/tooling details.

## Package and release boundary

Cargo/crates.io is the supported distribution path. The v0.2.0 package has an explicit `Cargo.toml` include inventory; verify the actual `.crate`, including CUDA headers, tests, fixtures, and scripts. For a clean exact commit, `docs/RELEASING.md` describes `scripts/prepare_crate.py`, extracted-crate tests, separate consumers, and the publication dry-run. The script requires Python 3.11.4+, CUDA, and Compute Sanitizer. It never uploads or tags.

v0.2.0 is published from `6cfb49c60fb7f95818e87d0ea0e1ce76c2360fb5`. Later `main` commits do not change that registry artifact; future publication requires a new version and its own exact-commit qualification.

Record the exact SHA and artifact hash. Merging a PR creates a new commit; earlier qualification cannot be assigned to it. Re-run the required qualification on the final candidate before changing LIM-1460's publication pin. Do not publish to crates.io, move the existing `v0.2.0` tag, or create a GitHub release as part of a preparation or reporting PR. The `linear-release.yml` workflow reports a verified stable GitHub release to Linear **after** the crate is live, using `LINEAR_ACCESS_KEY`; it is not a publishing workflow.

Keep PR review, CI, and tracker status tied to the current head. When a change is reviewed or merged, distinguish passing checks from unresolved review threads and from actual registry publication.
