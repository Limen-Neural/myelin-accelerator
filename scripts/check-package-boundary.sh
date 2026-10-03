#!/usr/bin/env bash
# Copyright 2026 Raul Cardenas Montoya
# SPDX-License-Identifier: MIT OR Apache-2.0

set -euo pipefail

tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT
cat >"$tmp"

# Ban research/experiment trees and SAAQ research layouts.
# ^examples/saaq matches examples/saaq/... and examples/saaq_manifest.json.
if grep -Ein '(^|/)(experiments|research)(/|$)|^examples/saaq|^tests/fixtures/saaq/|^docs/saaq/' "$tmp"; then
  echo 'experimental research must not be shipped in the crate archive' >&2
  exit 1
fi

# Packaged source trees: allow only reusable .rs / .cu / .cuh sources.
src_cu=$(grep -Ein '^(src|cu)/' "$tmp" || true)
if [[ -n "$src_cu" ]] && printf '%s\n' "$src_cu" | grep -Eiv '\.(rs|cu|cuh)$'; then
  echo 'experimental research must not be shipped in the crate archive' >&2
  exit 1
fi