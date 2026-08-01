#!/usr/bin/env bash
# Copyright 2026 Raul Montoya Cardenas
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Emit compile_commands.json for cu/*.cu so C++ tooling (Qodana/clangd) can
# parse kernel sources. Flags mirror build.rs / CMakeLists.txt host path
# (C++17 + STRICT_ANSI). This is NOT a device compile — only host-side analysis.
#
# Toolkit resolution (aligned with build.rs find_nvcc):
#   1. CUDA_NVCC (explicit nvcc path) → parent/../ as home when possible
#   2. CUDA_HOME if it is a real directory with include/
#   3. CUDA_PATH if it is a real directory with include/
#   4. /usr/local/cuda when present
#   5. else /usr/local/cuda (may be missing; frontend falls back to host C++)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ROOT

resolve_cuda_home() {
  if [[ -n "${CUDA_NVCC:-}" && -x "${CUDA_NVCC}" ]]; then
    # CUDA_NVCC=/path/to/bin/nvcc → /path/to
    local bin_dir home
    bin_dir="$(cd "$(dirname "${CUDA_NVCC}")" && pwd)"
    home="$(cd "${bin_dir}/.." && pwd)"
    if [[ -d "${home}/include" ]]; then
      echo "${home}"
      return
    fi
  fi
  if [[ -n "${CUDA_HOME:-}" && -d "${CUDA_HOME}/include" ]]; then
    echo "${CUDA_HOME}"
    return
  fi
  if [[ -n "${CUDA_PATH:-}" && -d "${CUDA_PATH}/include" ]]; then
    echo "${CUDA_PATH}"
    return
  fi
  if [[ -x /usr/local/cuda/bin/nvcc || -d /usr/local/cuda/include ]]; then
    readlink -f /usr/local/cuda 2>/dev/null || echo /usr/local/cuda
    return
  fi
  echo /usr/local/cuda
}

export CUDA_HOME
CUDA_HOME="$(resolve_cuda_home)"

python3 <<'PY'
import json
import os
import shutil
from pathlib import Path

root = Path(os.environ["ROOT"])
cu_dir = root / "cu"
cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
include_cuda = cuda_home / "include"
toolkit_ok = cuda_home.is_dir() and include_cuda.is_dir()

# Prefer toolkit-local nvcc over PATH so CUDA_HOME/CUDA_NVCC overrides win.
nvcc_home = cuda_home / "bin" / "nvcc"
nvcc = None
if os.environ.get("CUDA_NVCC") and Path(os.environ["CUDA_NVCC"]).is_file():
    nvcc = os.environ["CUDA_NVCC"]
elif nvcc_home.is_file():
    nvcc = str(nvcc_home)
else:
    nvcc = shutil.which("nvcc")

# Prefer a CUDA-capable frontend only when a real toolkit is present so
# clang++ -x cuda --cuda-path=... can resolve cuda_runtime.h.
clangxx = shutil.which("clang++") if toolkit_ok else None


def base_args(rel: str) -> list[str]:
    if clangxx:
        return [
            clangxx,
            "-std=c++17",
            "-x",
            "cuda",
            f"--cuda-path={cuda_home}",
            f"-I{cu_dir}",
            f"-I{include_cuda}",
            # Host-side parse only; do not require a device binary.
            "--cuda-host-only",
            "-c",
            rel,
        ]
    if nvcc:
        return [
            nvcc,
            "-std=c++17",
            "-D__STRICT_ANSI__",
            f"-I{cu_dir}",
            "-c",
            rel,
        ]
    # Last resort: plain C++ (limited CUDA parse fidelity).
    args = [
        os.environ.get("CXX", "c++"),
        "-std=c++17",
        "-D__STRICT_ANSI__",
        "-D__CUDACC__",
        "-x",
        "c++",
        f"-I{cu_dir}",
    ]
    if include_cuda.is_dir():
        args.append(f"-I{include_cuda}")
    args.extend(["-c", rel])
    return args


entries = []
for src in sorted(cu_dir.glob("*.cu")):
    rel = f"cu/{src.name}"
    entries.append(
        {
            "directory": str(root),
            "file": rel,
            "arguments": base_args(rel),
        }
    )

out = root / "compile_commands.json"
out.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
frontend = "clang++ -x cuda" if clangxx else ("nvcc" if nvcc else "c++ fallback")
print(f"Wrote {out} ({len(entries)} TUs; CUDA_HOME={cuda_home}; frontend={frontend})")
PY
