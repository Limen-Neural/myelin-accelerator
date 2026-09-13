#!/usr/bin/env bash
# Copyright 2026 Raul Montoya Cardenas
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Emit compile_commands.json for cu/*.cu so C++ tooling (clangd) can
# parse kernel sources. Flags mirror build.rs / CMakeLists.txt host path
# (C++17 + STRICT_ANSI). This is NOT a device compile — only host-side analysis.
#
# Requires Python 3.10+ (union syntax); prefer latest stable (3.14.x).
#
# Toolkit resolution (aligned with build.rs find_nvcc):
#   1. CUDA_NVCC (executable) → parent/../ as home when cuda_runtime.h exists
#   2. CUDA_HOME if include/cuda_runtime.h exists
#   3. CUDA_PATH if include/cuda_runtime.h exists
#   4. /usr/local/cuda when present
#   5. else /usr/local/cuda (may be missing; frontend falls back to host C++)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ROOT

is_cuda_home() {
  local home="$1"
  [[ -d "${home}/include" && -f "${home}/include/cuda_runtime.h" ]]
}

# Echo absolute path of a directory that exists (for compile_commands portability).
abs_dir() {
  local d="$1"
  (cd "${d}" && pwd -P) 2>/dev/null || (cd "${d}" && pwd)
}

resolve_cuda_home() {
  local nvcc_path=""
  if [[ -n "${CUDA_NVCC:-}" ]]; then
    if [[ -x "${CUDA_NVCC}" ]]; then
      nvcc_path="${CUDA_NVCC}"
    else
      # Bare PATH name (e.g. ccache-nvcc) — resolve like which(1).
      nvcc_path="$(command -v "${CUDA_NVCC}" 2>/dev/null || true)"
    fi
  fi
  if [[ -n "${nvcc_path}" && -x "${nvcc_path}" ]]; then
    local bin_dir home
    bin_dir="$(cd "$(dirname "${nvcc_path}")" && pwd)"
    home="$(cd "${bin_dir}/.." && pwd)"
    if is_cuda_home "${home}"; then
      abs_dir "${home}"
      return
    fi
  fi
  if [[ -n "${CUDA_HOME:-}" ]] && is_cuda_home "${CUDA_HOME}"; then
    abs_dir "${CUDA_HOME}"
    return
  fi
  if [[ -n "${CUDA_PATH:-}" ]] && is_cuda_home "${CUDA_PATH}"; then
    abs_dir "${CUDA_PATH}"
    return
  fi
  if is_cuda_home /usr/local/cuda; then
    abs_dir /usr/local/cuda
    return
  fi
  echo /usr/local/cuda
}

export CUDA_HOME
CUDA_HOME="$(resolve_cuda_home)"

python3 <<'PY'
"""Generate compile_commands.json for cu/*.cu (host analysis only).

Requires Python 3.10+ (PEP 604 unions). Prefer latest stable (3.14.x).
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

if sys.version_info < (3, 10):
    raise SystemExit(
        f"gen_compile_commands_cuda.sh needs Python >= 3.10 (got {sys.version_info.major}.{sys.version_info.minor}); "
        "use latest stable Python (3.14.x preferred)."
    )

root = Path(os.environ["ROOT"])
cu_dir = root / "cu"
cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
include_cuda = cuda_home / "include"
runtime_h = include_cuda / "cuda_runtime.h"
toolkit_ok = cuda_home.is_dir() and runtime_h.is_file()


def executable_path(p: Path | str | None) -> str | None:
    """Absolute path if p exists and is executable.

    Bare names (no directory separator) are resolved via PATH (shutil.which)
    so CUDA_NVCC=ccache-nvcc works. Uses absolute() not resolve() so
    basename-dispatched wrappers keep their entry-point name.
    """
    if not p:
        return None
    raw = str(p)
    # Single-component PATH lookup (e.g. "nvcc", "ccache-nvcc").
    if os.sep not in raw and (os.altsep is None or os.altsep not in raw):
        found = shutil.which(raw)
        if not found:
            return None
        path = Path(found)
    else:
        path = Path(raw)
    if not path.is_file():
        return None
    if not os.access(path, os.X_OK):
        return None
    return str(path.absolute())


# Prefer toolkit-local nvcc over PATH so CUDA_HOME/CUDA_NVCC overrides win.
nvcc: str | None = None
if env_nvcc := os.environ.get("CUDA_NVCC"):
    nvcc = executable_path(env_nvcc)
if nvcc is None:
    nvcc = executable_path(cuda_home / "bin" / "nvcc")
if nvcc is None:
    nvcc = executable_path("nvcc")

# Prefer a CUDA-capable frontend only when a real toolkit is present so
# clang++ -x cuda --cuda-path=... can resolve cuda_runtime.h.
clang_which = shutil.which("clang++") if toolkit_ok else None
clangxx = executable_path(clang_which) if clang_which else None


def resolve_cxx() -> str:
    """Host C++ frontend: validated CXX override, else PATH c++, else 'c++'."""
    cxx_env = os.environ.get("CXX", "").strip()
    if cxx_env:
        resolved = executable_path(cxx_env) or executable_path(shutil.which(cxx_env))
        if resolved:
            return resolved
    which_cxx = shutil.which("c++")
    if which_cxx:
        resolved = executable_path(which_cxx)
        if resolved:
            return resolved
    return "c++"


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
        resolve_cxx(),
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
print(
    f"Wrote {out} ({len(entries)} TUs; CUDA_HOME={cuda_home}; "
    f"frontend={frontend}; python={sys.version_info.major}.{sys.version_info.minor})"
)
PY
