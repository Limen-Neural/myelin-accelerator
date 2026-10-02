# Rust test coverage

The `Rust coverage` GitHub Actions workflow generates an LCOV report with
`cargo-llvm-cov` and uploads it to the existing Qlty project using OIDC.
It runs on pull requests, pushes to `main`, and manual dispatches. PR runs
measure the PR head commit so the uploaded report corresponds to that SHA.

## Scope

The initial `cpu-rust` report uses `--no-default-features --features bench
--all-targets`. It measures CPU/stub Rust code and the benchmark harness.
Test fixtures and test source files, plus `build.rs`, are excluded from the
coverage denominator. Inline unit tests inside reported source files can still
contribute to their coverage. Dependencies and generated build output are not
project coverage.

The report does **not** measure the CUDA-feature Rust wrappers in `src/gpu/`
or device code in `cu/`. It is not an overall Rust/CUDA coverage percentage.
CUDA numerical tests, PTX validation, and unsuppressed sanitizer gates remain
independent requirements. Model-shaped fixtures remain workload-only evidence.

## Local reproduction

Use a Rust toolchain with its matching `llvm-tools-preview` component and
`cargo-llvm-cov` 0.9.1:

```bash
mkdir -p target/coverage
cargo llvm-cov clean --workspace
cargo llvm-cov --locked --no-default-features --features bench \
  --all-targets --no-report --remap-path-prefix
cargo llvm-cov report --lcov \
  --ignore-filename-regex '(^|/)(tests/|build\.rs$)' \
  --output-path target/coverage/lcov.info
grep -Fxq 'SF:src/gpu_stub.rs' target/coverage/lcov.info
grep -Fxq 'SF:examples/benchmark.rs' target/coverage/lcov.info
```

Tests and export are separate because the combined command's default filter
excludes remapped `examples/` paths. Export uses the already-remapped profiles,
and CI checks that both the CPU stub and benchmark harness appear in LCOV.
The explicit clean prevents profiles from earlier runs contaminating the report.

CI retains `target/coverage/lcov.info` in the `cpu-rust-coverage` artifact
for 14 days. Report generation and upload failures fail the workflow rather
than silently presenting a missing report as successful coverage.

## Qlty and authentication

The upload job grants only `contents: read` and `id-token: write`; no long-lived
coverage token is stored. Fork and Dependabot pull requests generate the report
artifact but skip the OIDC upload. The workflow uses `pull_request`, not
`pull_request_target`.

Coverage is initially informational: this change sets no percentage threshold,
does not enable Qlty coverage/diff-coverage gates, and does not add a required
branch-protection check. Verify an uploaded branch report against its commit
and establish the `main` baseline before deciding on regression thresholds.
Qlty's main overview needs a coverage upload from `main`, so PR validation
alone does not establish the default-branch baseline.

A later CUDA-enabled Rust report can extend host-wrapper coverage on the GPU
runner. Use the Qlty CLI for that upload: the Qlty coverage action is not
supported on self-hosted runners. That still would not instrument `.cu` kernels.

References: [Qlty coverage setup](https://docs.qlty.sh/coverage/quickstart),
[cargo-llvm-cov](https://github.com/taiki-e/cargo-llvm-cov).
