# Benchmark manifests and regression budgets

Every `examples/benchmark` run writes a versioned, machine-readable manifest
beside the existing JSON/CSV output. Optional device fields are always present
and may be `null`. Local usernames, home paths, and secret-shaped tokens are
redacted before the file is written.

Hardware performance budgets are **opt-in**. Default CPU CI runs schema and
statistics tests only; it never sets the enforcement flag, so a run on
different hardware cannot fail the suite.

## Record a run

CPU-only (CI-safe smoke; no GPU):

```bash
cargo run --locked --example benchmark --features bench -- \
  --warmup 2 --iterations 10 --output /tmp/myelin-bench
```

GPU kernels (local / self-hosted, toolkit 13.2+):

```bash
CUDA_NVCC=/usr/local/cuda/bin/nvcc \
  cargo run --locked --example benchmark --profile bench --features bench,cuda -- \
  --output /tmp/myelin-bench
```

Artifacts (never committed from the working directory prefix
`benchmark_results`):

| File | Role |
|------|------|
| `{prefix}.json` / `{prefix}.csv` | Legacy latency table (unchanged public shape) |
| `{prefix}.manifest.json` | Versioned provenance + samples |
| `{prefix}.comparison.json` | Written only when `--baseline` is set |

The manifest includes git commit/dirty state, enabled Cargo features, kernel
variant and input dimensions per case, seed, warmup/sample counts, device
identity, compute capability, driver/runtime/nvcc versions, and `nvidia-smi`
power/clock fields when that binary is available.

## Compare (informational)

```bash
cargo run --locked --example benchmark --features bench -- \
  --baseline tests/fixtures/bench/manifest.sanitized.json \
  --output /tmp/myelin-bench-cmp
```

`--baseline` is read-only. The harness refuses to use an `--output` prefix that
would overwrite the baseline file.

Classification uses **median** latency and **relative MAD** dispersion:

1. `insufficient_samples` — either side has fewer than `--min-samples`
2. `noisy` — relative MAD exceeds `--noisy-dispersion`
3. `fail` — current median is slower **and** both `--budget-relative` **and**
   `--budget-abs-us` are exceeded
4. `pass` — otherwise (including speedups)

Without `--enforce-budget`, `fail` is printed and recorded but the process
still exits 0.

## Enforce a budget (opt-in)

```bash
MYELIN_BENCH_ENFORCE_BUDGET=1 \
  cargo run --locked --example benchmark --features bench -- \
  --baseline path/to/committed.manifest.json \
  --output /tmp/myelin-bench-cmp \
  --enforce-budget
```

Either the environment variable (`1` / `true` / `yes` / `on`) or
`--enforce-budget` turns on process failure, and **only** `class=fail` fails
the process. Do not set this in ordinary GitHub-hosted CPU CI.

Defaults: 10% relative, 2 µs absolute, 8 samples, 0.25 relative MAD.

## Refresh a baseline

Refreshing a baseline is a **reviewable file change**, not an implicit
overwrite:

1. Record a new run to a throwaway prefix (`--output /tmp/myelin-bench-new`).
2. Inspect `{prefix}.manifest.json` (redaction, git dirty flag, device fields).
3. Copy it into the tree, for example
   `tests/fixtures/bench/baselines/<host>.manifest.json`.
4. Commit that copy in a PR.

Do not point `--output` at the committed baseline path. The harness will
refuse to clobber a `--baseline` file.

Checked-in classification fixtures live at:

- `tests/fixtures/bench/manifest.sanitized.json`
- `tests/fixtures/bench/comparison.fixture.json`

`cargo test --locked` exercises schema round-trips and deterministic
pass/fail/noisy/insufficient-sample classification from those fixtures.
