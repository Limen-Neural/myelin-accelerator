#!/usr/bin/env bash
# Copyright 2026 Raul Cardenas Montoya
# SPDX-License-Identifier: MIT OR Apache-2.0

set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
checker="$root_dir/scripts/check-package-boundary.sh"

printf '%s\n' \
  '.cargo_vcs_info.json' \
  'Cargo.lock' \
  'Cargo.toml' \
  'Cargo.toml.orig' \
  'LICENSE-APACHE' \
  'LICENSE-MIT' \
  'README.md' \
  'REVIEW.md' \
  'build.rs' \
  'src/lib.rs' \
  'src/gif.rs' \
  'cu/common.cuh' \
  'cu/myelin_shim.cu' \
  'cu/myelin_shim.cuh' \
  'examples/benchmark.rs' \
  'tests/api_contract.rs' \
  'tests/capability_probe.rs' \
  'tests/cosine_gpu.rs' \
  'tests/gif_gpu.rs' \
  'tests/gpu_buffer.rs' \
  'tests/gpu_lifecycle.rs' \
  'tests/kernel_hazards_gpu.rs' \
  'tests/oracle.rs' \
  'tests/oracle_gpu.rs' \
  'tests/ptx_entry_manifest.rs' \
  'tests/snn_fixtures.rs' \
  'tests/snn_fixtures_gpu.rs' \
  'tests/snn_support/mod.rs' \
  'tests/ternary_gpu.rs' \
  'tests/test_release_prep.py' \
  'tests/fixtures/snn/README.md' \
  'tests/fixtures/snn/spikenaut/fixture.json' \
  'tests/fixtures/snn/spikenaut/parameters_weights.mem' \
  'docs/ARCHITECTURE.md' \
  'scripts/check-package-boundary.sh' \
  'scripts/snn_fixtures/verify_fixtures.py' \
  'scripts/ptx_entries.txt' \
  | "$checker"

for forbidden_path in \
  'experiments/research/recipe.toml' \
  'research/experiment/results.json' \
  'examples/reference_weights.bin' \
  'examples/experimental.rs' \
  'tests/results.jsonl' \
  'tests/fixtures/results.tsv' \
  'tests/fixtures/snn/unapproved/fixture.json' \
  'tests/fixtures/snn/spikenaut/weights.py' \
  'tests/run_experiment.py' \
  'tests/run_experiment.rs' \
  'tests/bench_repro.rs' \
  'tests/fixtures/bench/manifest.sanitized.json' \
  'docs/benchmark_manifest.json' \
  'scripts/experimental_runner.py' \
  'LICENSE-results.json' \
  'mystery/dataset.json'; do
  if printf '%s\n' "$forbidden_path" | "$checker" >/dev/null 2>&1; then
    echo "package boundary accepted forbidden path: $forbidden_path" >&2
    exit 1
  fi
done
