#!/usr/bin/env bash
# Copyright 2026 Raul Cardenas Montoya
# SPDX-License-Identifier: MIT OR Apache-2.0

set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
checker="$root_dir/scripts/check-package-boundary.sh"

printf '%s\n' \
  'src/lib.rs' \
  'src/saaq.rs' \
  'cu/saaq.cu' \
  'cu/fused_routing_saaq.cu' \
  'cu/common.cuh' \
  'examples/benchmark.rs' \
  'tests/api_contract.rs' \
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
  'experiments/saaq/recipe.toml' \
  'research/saaq/results.json' \
  'examples/saaq/manifest.json' \
  'tests/fixtures/saaq/dataset.json' \
  'docs/saaq/experiment.md' \
  'src/saaq/recipe.toml' \
  'src/saaq/results.json' \
  'src/saaq/results.jsonl' \
  'src/saaq/weights.bin' \
  'cu/saaq/dataset.json' \
  'cu/saaq/results.tsv' \
  'src/quantization/saaq_manifest.json' \
  'examples/saaq_manifest.json' \
  'examples/reference_weights.bin' \
  'examples/experimental.rs' \
  'tests/results.jsonl' \
  'tests/fixtures/results.tsv' \
  'tests/fixtures/snn/unapproved/fixture.json' \
  'tests/fixtures/snn/spikenaut/weights.py' \
  'tests/saaq_experiment.py' \
  'tests/run_experiment.py' \
  'tests/bench_repro.rs' \
  'tests/fixtures/bench/manifest.sanitized.json' \
  'docs/benchmark_manifest.json' \
  'docs/saaq_results.md' \
  'docs/SAAQ-experiment.md' \
  'scripts/saaq/results.jsonl' \
  'scripts/saaq/recipe.py' \
  'scripts/saaq/run_experiment.py' \
  'scripts/experimental_runner.py'; do
  if printf '%s\n' "$forbidden_path" | "$checker" >/dev/null 2>&1; then
    echo "package boundary accepted forbidden path: $forbidden_path" >&2
    exit 1
  fi
done
