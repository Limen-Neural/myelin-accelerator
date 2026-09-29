#!/usr/bin/env bash
# Copyright 2026 Raul Montoya Cardenas
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Print the myelin-accelerator build-script OUT_DIR deterministically.
# Derives the path from Cargo JSON (`build-script-executed`) instead of
# selecting a build directory by modification time.
#
# Usage: ptx_out_dir.sh [--release]
set -euo pipefail

release=0
for arg in "$@"; do
    case "$arg" in
        --release) release=1 ;;
        *) echo "usage: ptx_out_dir.sh [--release]" >&2; exit 2 ;;
    esac
done

command -v cargo >/dev/null || { echo 'ERROR: cargo is required.' >&2; exit 1; }
command -v python3 >/dev/null || { echo 'ERROR: python3 is required to parse Cargo JSON.' >&2; exit 1; }

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
# stderr stays visible: with json-render-diagnostics, human-readable
# diagnostics go to stderr while JSON goes to stdout.
if [ "$release" -eq 1 ]; then
    cargo build --locked --features cuda --release --message-format=json-render-diagnostics > "$work/cargo.json"
else
    cargo build --locked --features cuda --message-format=json-render-diagnostics > "$work/cargo.json"
fi

out_dir="$(python3 - "$work/cargo.json" <<'PY'
import json
import sys
out_dirs = set()
with open(sys.argv[1], encoding="utf-8") as messages:
    for line in messages:
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if item.get("reason") != "build-script-executed":
            continue
        out = item.get("out_dir", "")
        # Match on the Cargo unit directory, which embeds the package name
        # (`build/myelin-accelerator-<hash>/out`). The `package_id` for a
        # path dependency is a file URI of the checkout directory and does
        # not reliably contain the package name.
        if "/build/myelin-accelerator-" not in f"/{out}/":
            continue
        if out:
            out_dirs.add(out)
if len(out_dirs) != 1:
    sys.exit(f"ERROR: expected one myelin-accelerator OUT_DIR, found {len(out_dirs)}")
print(out_dirs.pop())
PY
)"
printf '%s' "$out_dir"
