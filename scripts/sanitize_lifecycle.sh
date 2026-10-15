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
cargo test --locked --features cuda --test gpu_lifecycle --test snn_fixtures_gpu --no-run \
    --message-format=json-render-diagnostics > "$work/cargo.json"
python3 - "$work/cargo.json" > "$work/binaries.tsv" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as messages:
    artifacts = [json.loads(line) for line in messages]
for suite in ("gpu_lifecycle", "snn_fixtures_gpu"):
    executables = {
        item["executable"] for item in artifacts
        if item.get("reason") == "compiler-artifact"
        and item.get("target", {}).get("name") == suite
        and item.get("profile", {}).get("test")
        and item.get("executable")
    }
    if len(executables) != 1:
        sys.exit(f"ERROR: expected one {suite} executable, found {len(executables)}")
    print(f"{suite}\t{executables.pop()}")
PY

# Separate processes and logs keep each owner's context lifetime observable.
while IFS=$'\t' read -r suite binary; do
    log="$work/$suite.log"
    printf 'Running %s: %q --tool memcheck --error-exitcode 99 %q --include-ignored --test-threads=1 --nocapture\n' "$suite" "$sanitizer" "$binary"
    "$sanitizer" --tool memcheck --error-exitcode 99 "$binary" \
        --include-ignored --test-threads=1 --nocapture 2>&1 | tee "$log"
    grep -Eq '^test result: ok\. [1-9][0-9]* passed; 0 failed; 0 ignored;' "$log" || {
        echo "ERROR: $suite did not confirm executed, passing device tests." >&2
        exit 1
    }
    grep -Eq '^========= ERROR SUMMARY: 0 errors$' "$log" || {
        echo "ERROR: $suite did not confirm zero Compute Sanitizer errors." >&2
        exit 1
    }
done < "$work/binaries.tsv"
