#!/usr/bin/env bash
# Copyright 2026 Raul Montoya Cardenas
# SPDX-License-Identifier: MIT OR Apache-2.0

set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
checker="$root_dir/scripts/check-package-boundary.sh"

printf '%s\n' \
  'src/lib.rs' \
  'src/saaq.rs' \
  'cu/saaq.cu' \
  'cu/fused_routing_saaq.cu' \
  'docs/ARCHITECTURE.md' \
  | "$checker"

for forbidden_path in \
  'experiments/saaq/recipe.toml' \
  'research/saaq/results.json' \
  'examples/saaq/manifest.json' \
  'tests/fixtures/saaq/dataset.json' \
  'docs/saaq/experiment.md' \
  'src/saaq/recipe.toml' \
  'src/saaq/results.json' \
  'cu/saaq/dataset.json' \
  'src/quantization/saaq_manifest.json'; do
  if printf '%s\n' "$forbidden_path" | "$checker" >/dev/null 2>&1; then
    echo "package boundary accepted forbidden path: $forbidden_path" >&2
    exit 1
  fi
done
