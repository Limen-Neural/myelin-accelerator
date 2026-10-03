#!/usr/bin/env bash
# Copyright 2026 Raul Cardenas Montoya
# SPDX-License-Identifier: MIT OR Apache-2.0

set -euo pipefail

tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT
cat >"$tmp"

# Ban research/experiment trees, SAAQ research layouts, and repository-only
# benchmark regression inputs.
# ^examples/saaq matches examples/saaq/... and examples/saaq_manifest.json.
if grep -Ein '(^|/)(experiments|research)(/|$)|^examples/saaq|^tests/fixtures/(saaq|bench)/|^tests/bench_repro\.rs$|^docs/saaq/' "$tmp"; then
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
check_allowed_paths 'examples' '^examples/.*\.rs$'
check_allowed_paths 'tests' '^tests/.*\.(rs|py)$|^tests/fixtures/snn/README\.md$|^tests/fixtures/snn/[^/]+/fixture\.json$|^tests/fixtures/snn/spikenaut/parameters_weights\.mem$'
check_allowed_paths 'docs' '^docs/.*\.md$'
check_allowed_paths 'scripts' '^scripts/.*\.(sh|py)$|^scripts/ptx_entries\.txt$'
