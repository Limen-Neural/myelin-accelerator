#!/usr/bin/env bash
# Verify every kernel entry registered by KernelModule::load exists as a
# `.entry` symbol in the corresponding generated PTX file.
# Usage: check_ptx_entries.sh <ptx_dir> [label]
set -euo pipefail

ptx_dir="${1:?usage: check_ptx_entries.sh <ptx_dir> [label]}"
label="${2:-}"
manifest="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ptx_entries.txt"

status=0
while IFS= read -r line; do
    case "$line" in
        ''|'#'*) continue ;;
    esac
    file="${line%%:*}"
    kernels="${line#*:}"
    if [ ! -f "$ptx_dir/$file" ]; then
        echo "::error::Missing PTX file${label:+ ($label)}: $file"
        status=1
        continue
    fi
    for kernel in $kernels; do
        if grep -Eq "\\.entry[[:space:]]+${kernel}[[:space:]]*\\(" "$ptx_dir/$file"; then
            echo "✓ ${label:+$label }$file: $kernel"
        else
            echo "::error::Missing kernel entry: $kernel in${label:+ $label} $file"
            status=1
        fi
    done
done < "$manifest"

exit "$status"
