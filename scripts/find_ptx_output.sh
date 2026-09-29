#!/usr/bin/env bash
# Locate the one Cargo build-script output directory containing every PTX file.
# Cargo hashes build-script directories, so do not select one by modification time.
set -euo pipefail

target_dir="${1:?usage: find_ptx_output.sh <cargo-target-dir> <debug|release>}"
profile="${2:?usage: find_ptx_output.sh <cargo-target-dir> <debug|release>}"
build_dir="$target_dir/$profile/build"
required="spiking_network_sm_120.ptx vector_similarity_sm_120.ptx satsolver_sm_120.ptx ternary_gemm_sm_120.ptx"

if [[ ! -d "$build_dir" ]]; then
  echo "PTX build directory does not exist: $build_dir" >&2
  exit 1
fi

matches=()
while IFS= read -r -d '' candidate; do
  valid=true
  for file in $required; do
    [[ -f "$candidate/$file" ]] || valid=false
  done
  "$valid" && matches+=("$candidate")
done < <(find "$build_dir" -mindepth 2 -maxdepth 2 -type d -name out -print0 | sort -z)

if (( ${#matches[@]} != 1 )); then
  echo "Expected exactly one complete PTX output directory under $build_dir; found ${#matches[@]}" >&2
  printf '  %s\n' "${matches[@]}" >&2
  exit 1
fi

printf '%s\n' "${matches[0]}"
