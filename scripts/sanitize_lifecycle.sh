#!/usr/bin/env bash
# Copyright 2026 Raul Montoya Cardenas
# SPDX-License-Identifier: MIT OR Apache-2.0
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Match the build's selected toolkit. An explicit sanitizer override is useful
# when the toolkit's sanitizer is installed separately from nvcc.
if [[ -n ${CUDA_NVCC:-} ]]; then
    nvcc="$(command -v "$CUDA_NVCC")"
elif [[ -n ${CUDA_HOME:-} ]]; then
    nvcc="$CUDA_HOME/bin/nvcc"
else
    nvcc="$(command -v nvcc || true)"
    nvcc="${nvcc:-/usr/local/cuda/bin/nvcc}"
fi
export CUDA_NVCC="$nvcc"
if [[ -n ${COMPUTE_SANITIZER:-} ]]; then
    sanitizer="$COMPUTE_SANITIZER"
else
    sanitizer="$(dirname "$(readlink -f "$nvcc")")/compute-sanitizer"
    if [[ ! -x "$sanitizer" ]]; then
        sanitizer="$(command -v compute-sanitizer || true)"
    fi
fi
if [[ -z "$sanitizer" ]] || ! command -v "$sanitizer" >/dev/null 2>&1; then
    echo 'ERROR: Compute Sanitizer is required; set COMPUTE_SANITIZER or select its CUDA toolkit.' >&2
    exit 1
fi
command -v python3 >/dev/null || { echo 'ERROR: python3 is required to parse Cargo JSON.' >&2; exit 1; }
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
cargo test --locked --features cuda --test gpu_lifecycle --no-run \
    --message-format=json-render-diagnostics > "$work/cargo.json"
binary="$(python3 - "$work/cargo.json" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as messages:
    artifacts = [json.loads(line) for line in messages]
executables = {
    item["executable"] for item in artifacts
    if item.get("reason") == "compiler-artifact"
    and item.get("target", {}).get("name") == "gpu_lifecycle"
    and item.get("profile", {}).get("test")
    and item.get("executable")
}
if len(executables) != 1:
    sys.exit(f"ERROR: expected one gpu_lifecycle executable, found {len(executables)}")
print(executables.pop())
PY
)"
printf 'Running: %q --tool memcheck --error-exitcode 99 %q --ignored --test-threads=1 --nocapture\n' "$sanitizer" "$binary"
"$sanitizer" --tool memcheck --error-exitcode 99 "$binary" \
    --ignored --test-threads=1 --nocapture 2>&1 | tee "$work/sanitizer.log"
grep -Eq '^========= ERROR SUMMARY: 0 errors$' "$work/sanitizer.log" || {
    echo 'ERROR: Compute Sanitizer did not confirm zero errors.' >&2
    exit 1
}
