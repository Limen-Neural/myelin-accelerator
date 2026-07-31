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
from pathlib import Path

root = Path(os.environ["ROOT"])
cu_dir = root / "cu"
cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
include_cuda = cuda_home / "include"
cxx = os.environ.get("CXX", "c++")

entries = []
for src in sorted(cu_dir.glob("*.cu")):
    rel = f"cu/{src.name}"
    args = [
        cxx,
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
    entries.append(
        {
            "directory": str(root),
            "file": rel,
            "arguments": args,
        }
    )

out = root / "compile_commands.json"
out.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
print(f"Wrote {out} ({len(entries)} translation units; CUDA_HOME={cuda_home})")
PY
