#!/usr/bin/env bash
# Copyright 2026 Raul Mc
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Emit compile_commands.json for cu/*.cu so C++ tooling (Qodana/clangd) can
# parse kernel sources. Flags mirror build.rs / CMakeLists.txt host path
# (C++17 + STRICT_ANSI). This is NOT a device compile — only host-side analysis.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ROOT

if [[ -x /usr/local/cuda/bin/nvcc ]]; then
  export CUDA_HOME
  CUDA_HOME="$(readlink -f /usr/local/cuda 2>/dev/null || echo /usr/local/cuda)"
elif [[ -n "${CUDA_HOME:-}" ]]; then
  export CUDA_HOME
else
  export CUDA_HOME="/usr/local/cuda"
fi

python3 <<'PY'
import json
import os
import shutil
from pathlib import Path

root = Path(os.environ["ROOT"])
cu_dir = root / "cu"
cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
include_cuda = cuda_home / "include"
# Prefer a CUDA-capable frontend so Qodana/clangd understand __global__/blockIdx.
# Fall back to host C++ only if neither clang++ nor nvcc is available.
clangxx = shutil.which("clang++")
nvcc = shutil.which("nvcc") or (
    str(cuda_home / "bin" / "nvcc") if (cuda_home / "bin" / "nvcc").is_file() else None
)


def base_args(rel: str) -> list[str]:
    if clangxx:
        args = [
            clangxx,
            "-std=c++17",
            "-x",
            "cuda",
            f"--cuda-path={cuda_home}",
            f"-I{cu_dir}",
        ]
        if include_cuda.is_dir():
            args.append(f"-I{include_cuda}")
        # Host-side parse only; do not require a device binary.
        args.extend(["--cuda-host-only", "-c", rel])
        return args
    if nvcc:
        args = [
            nvcc,
            "-std=c++17",
            "-D__STRICT_ANSI__",
            f"-I{cu_dir}",
            "-c",
            rel,
        ]
        return args
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
