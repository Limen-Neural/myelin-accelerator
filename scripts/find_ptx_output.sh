#!/usr/bin/env bash
# Locate the newest real-CUDA Cargo build-script PTX output directory.
# Cargo retains hashed output directories for prior feature sets and revisions.
set -euo pipefail

target_dir="${1:?usage: find_ptx_output.sh <cargo-target-dir> <debug|release>}"
profile="${2:?usage: find_ptx_output.sh <cargo-target-dir> <debug|release>}"
build_dir="$target_dir/$profile/build"
required="spiking_network_sm_120.ptx vector_similarity_sm_120.ptx satsolver_sm_120.ptx ternary_gemm_sm_120.ptx"

if [[ ! -d "$build_dir" ]]; then
  echo "PTX build directory does not exist: $build_dir" >&2
  exit 1
fi

latest_candidate=""
latest_mtime=""
while IFS= read -r -d '' candidate; do
  valid=true
  for file in $required; do
    [[ -f "$candidate/$file" ]] || valid=false
    grep -q '^\.target sm_120' "$candidate/$file" || valid=false
  done
  if "$valid"; then
    mtime="$(stat -c '%y' "$candidate")"
    if [[ -z "$latest_mtime" || "$mtime" > "$latest_mtime" ]]; then
      latest_candidate="$candidate"
      latest_mtime="$mtime"
    fi
  fi
done < <(find "$build_dir" -mindepth 2 -maxdepth 2 -type d -name out -print0 | sort -z)

if [[ -z "$latest_candidate" ]]; then
  echo "No complete sm_120 PTX output directory found under $build_dir" >&2
  exit 1
fi

printf '%s\n' "$latest_candidate"
