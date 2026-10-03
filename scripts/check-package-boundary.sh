#!/usr/bin/env bash
# Copyright 2026 Raul Cardenas Montoya
# SPDX-License-Identifier: MIT OR Apache-2.0

set -euo pipefail

tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT
cat >"$tmp"

# Ban research/experiment trees, SAAQ research layouts, and repository-only
# benchmark regression inputs.
# These SAAQ prefixes match both subtrees and prefixed payloads.
if grep -Ein '(^|/)(experiments|research)(/|$)|^(docs|examples|scripts)/saaq|^tests/fixtures/(saaq|bench)/|^tests/bench_repro\.rs$' "$tmp"; then
  echo 'experimental research must not be shipped in the crate archive' >&2
  exit 1
fi

check_allowed_paths() {
  local directory=$1
  local allowed=$2
  local paths

  paths=$(grep -E "^${directory}/" "$tmp" || true)
  if [[ -n "$paths" ]] && printf '%s\n' "$paths" | grep -Ev "$allowed"; then
    echo 'experimental research must not be shipped in the crate archive' >&2
    exit 1
  fi
}

# Broad Cargo include globs are narrowed to reusable source and documentation.
# The named SNN workload fixtures are intentional package-test inputs.
check_allowed_paths 'src' '^src/.*\.rs$'
check_allowed_paths 'cu' '^cu/.*\.(cu|cuh)$'
check_allowed_paths 'examples' '^examples/benchmark\.rs$'
check_allowed_paths 'tests' '^tests/[^/]+\.rs$|^tests/test_release_prep\.py$|^tests/snn_support/.*\.rs$|^tests/fixtures/snn/README\.md$|^tests/fixtures/snn/(spikenaut|synfire_lifneuron)/fixture\.json$|^tests/fixtures/snn/spikenaut/parameters_weights\.mem$'
check_allowed_paths 'docs' '^docs/(ARCHITECTURE|BENCHMARKS|COVERAGE|RELEASING|SNN_COMPATIBILITY|TERNARY)\.md$'
check_allowed_paths 'scripts' '^scripts/(check-package-boundary|check_ptx_entries|gen_compile_commands_cuda|ptx_out_dir|sanitize_lifecycle|test-package-boundary)\.sh$|^scripts/prepare_crate\.py$|^scripts/ptx_entries\.txt$|^scripts/snn_fixtures/verify_fixtures\.py$'
