#!/usr/bin/env bash
# Copyright 2026 Raul Cardenas Montoya
# SPDX-License-Identifier: MIT OR Apache-2.0

set -euo pipefail

tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT
cat >"$tmp"

unapproved_paths=$(grep -Ev '^(src|cu|examples|tests|scripts|docs)/|^(\.cargo_vcs_info\.json|Cargo\.lock|Cargo\.toml|Cargo\.toml\.orig|LICENSE-APACHE|LICENSE-MIT|README\.md|REVIEW\.md|build\.rs)$' "$tmp" || true)
if [[ -n "$unapproved_paths" ]]; then
  printf '%s\n' "$unapproved_paths"
  echo 'unapproved package path must not be shipped in the crate archive' >&2
  exit 1
fi

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
check_allowed_paths 'tests' '^tests/(api_contract|capability_probe|cosine_gpu|gpu_buffer|gpu_lifecycle|kernel_hazards_gpu|oracle|oracle_gpu|ptx_entry_manifest|snn_fixtures|snn_fixtures_gpu|ternary_gpu)\.rs$|^tests/test_release_prep\.py$|^tests/snn_support/mod\.rs$|^tests/fixtures/snn/README\.md$|^tests/fixtures/snn/(spikenaut|synfire_lifneuron)/fixture\.json$|^tests/fixtures/snn/spikenaut/parameters_weights\.mem$'
check_allowed_paths 'docs' '^docs/(ARCHITECTURE|BENCHMARKS|COVERAGE|RELEASING|SNN_COMPATIBILITY|TERNARY)\.md$'
check_allowed_paths 'scripts' '^scripts/(check-package-boundary|check_ptx_entries|gen_compile_commands_cuda|ptx_out_dir|sanitize_lifecycle|test-package-boundary)\.sh$|^scripts/prepare_crate\.py$|^scripts/ptx_entries\.txt$|^scripts/snn_fixtures/verify_fixtures\.py$'
