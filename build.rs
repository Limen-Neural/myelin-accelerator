// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::env;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

// Stub PTX is only embedded when the `cuda` feature is off. It is never
// JIT-loaded on a real GPU, so an older ISA + sm_80 target is fine.
const PTX_STUB: &str = ".version 8.5\n.target sm_80\n.address_size 64\n";

// sm_120 (Blackwell) requires PTX ISA ≥ 9.0. nvcc 13.x emits .version 9.2.
// Never default to the stub's 8.5 — that yields InvalidPtx at JIT time.
const DEFAULT_REAL_PTX_VERSION: &str = "9.2";

const KERNELS: &[(&str, &str)] = &[
    ("spiking_network.cu", "spiking_network_sm_120.ptx"),
    ("vector_similarity.cu", "vector_similarity_sm_120.ptx"),
    ("satsolver.cu", "satsolver_sm_120.ptx"),
    ("ternary_gemm.cu", "ternary_gemm_sm_120.ptx"),
];

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let cu_dir = manifest_dir.join("cu");
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));

    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_NVCC");
    println!("cargo:rerun-if-env-changed=MYELIN_CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=MYELIN_PTX_VERSION");
    println!("cargo:rerun-if-env-changed=MYELIN_NVCC_THREADS");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=cu/common.cuh");
    for &(cu_name, _) in KERNELS {
        println!("cargo:rerun-if-changed=cu/{cu_name}");
    }

    let cuda_feature_enabled = env::var("CARGO_FEATURE_CUDA").is_ok();
    if !cuda_feature_enabled {
        emit_stub_ptx(&out_dir);
        println!("cargo:warning=cuda feature not enabled; wrote stub PTX files");
        return;
    }

    let nvcc = find_nvcc();
    let arch = env::var("MYELIN_CUDA_ARCH").unwrap_or_else(|_| "sm_120".to_string());
    // Treat empty MYELIN_PTX_VERSION as unset (Some("") would write ".version ").
    let ptx_version_override = env::var("MYELIN_PTX_VERSION")
        .ok()
        .filter(|v| !v.trim().is_empty());
    let is_blackwell = arch.starts_with("sm_12") || arch.starts_with("compute_12");

    let Some(nvcc_path) = nvcc else {
        panic!(
            "cuda feature is enabled but nvcc was not found. Install CUDA toolkit or set CUDA_NVCC."
        );
    };

    match nvcc_version(&nvcc_path) {
        Some(v) => println!("cargo:warning=using nvcc: {v}"),
        None => println!("cargo:warning=using nvcc at {}", nvcc_path.display()),
    }

    for &(cu_name, ptx_name) in KERNELS {
        let source = cu_dir.join(cu_name);
        let output = out_dir.join(ptx_name);
        compile_to_ptx(&nvcc_path, &cu_dir, &source, &output, &arch);
        // sm_120 + PTX < 9.2 is invalid. Explicit overrides are applied for
        // non-Blackwell arches; on Blackwell, clamp any override below the floor.
        if let Some(ref ver) = ptx_version_override {
            let effective = if is_blackwell && ptx_version_less(ver, DEFAULT_REAL_PTX_VERSION) {
                println!(
                    "cargo:warning=MYELIN_PTX_VERSION={ver} is below {DEFAULT_REAL_PTX_VERSION} for {arch}; clamping"
                );
                DEFAULT_REAL_PTX_VERSION
            } else {
                ver.as_str()
            };
            patch_ptx_version_any(&output, effective);
            println!(
                "cargo:warning=compiled {cu_name} -> {ptx_name} (arch={arch}, ptx={effective} [override])"
            );
        } else if is_blackwell {
            // Leave nvcc's header when already high enough; raise only if lower.
            ensure_min_ptx_version(&output, DEFAULT_REAL_PTX_VERSION);
            println!(
                "cargo:warning=compiled {cu_name} -> {ptx_name} (arch={arch}, ptx>={DEFAULT_REAL_PTX_VERSION})"
            );
        } else {
            println!(
                "cargo:warning=compiled {cu_name} -> {ptx_name} (arch={arch}, ptx=nvcc-default)"
            );
        }
    }
}

fn find_nvcc() -> Option<PathBuf> {
    if let Ok(path) = env::var("CUDA_NVCC") {
        let p = PathBuf::from(path);
        if is_nvcc_binary(&p) {
            return Some(p);
        }
    }

    for root_var in ["CUDA_HOME", "CUDA_PATH"] {
        if let Ok(root) = env::var(root_var) {
            let p = PathBuf::from(root).join("bin").join(exe_name("nvcc"));
            if is_nvcc_binary(&p) {
                return Some(p);
            }
        }
    }

    let candidate = PathBuf::from(exe_name("nvcc"));
    if is_nvcc_binary(&candidate) {
        Some(candidate)
    } else {
        None
    }
}

fn is_nvcc_binary(path: &Path) -> bool {
    let Ok(out) = Command::new(path).arg("--version").output() else {
        return false;
    };
    if !out.status.success() {
        return false;
    }
    let mut blob = String::new();
    if let Ok(s) = String::from_utf8(out.stdout) {
        blob.push_str(&s);
    }
    if let Ok(s) = String::from_utf8(out.stderr) {
        blob.push_str(&s);
    }
    blob.to_ascii_lowercase().contains("nvcc")
}

fn exe_name(base: &str) -> OsString {
    if cfg!(windows) {
        format!("{base}.exe").into()
    } else {
        base.into()
    }
}

fn nvcc_version(nvcc: &Path) -> Option<String> {
    let out = Command::new(nvcc).arg("--version").output().ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8(out.stdout)
        .ok()
        .and_then(|s| s.lines().last().map(|l| l.trim().to_string()))
}

fn compile_to_ptx(nvcc: &Path, cu_dir: &Path, source: &Path, output: &Path, arch: &str) {
    let threads_raw = env::var("MYELIN_NVCC_THREADS").unwrap_or_else(|_| "0".to_string());
    let threads = threads_raw.parse::<usize>().unwrap_or_else(|_| {
        panic!("MYELIN_NVCC_THREADS must be a non-negative integer, got \"{threads_raw}\"")
    });

    // Flags:
    //   -std=c++17                      keep the Edison host front-end in C++17
    //                                    mode so libstdc++ 16's C++23 paths aren't
    //                                    parsed (avoids the type_traits errors
    //                                    seen on GCC 16 / nvcc 13.2 hosts).
    //   --expt-relaxed-constexpr        allow device-side relaxed constexpr.
    //   -Xcompiler -fno-builtin         pass -fno-builtin to the host compiler
    //                                    (GCC/Clang) so nvcc doesn't misread
    //                                    host libm builtins as device intrinsics.
    //                                    On MSVC the equivalent is /Oi-, but the
    //                                    CI only targets Linux + the
    //                                    self-hosted runner is Linux, so this
    //                                    is fine. Add an MSVC branch here if
    //                                    Windows CUDA builds become supported.
    let mut cmd = Command::new(nvcc);
    cmd.arg("-ptx")
        .arg(format!("-arch={arch}"))
        .arg("-O3")
        .arg("--use_fast_math")
        .arg("--restrict")
        .arg("--threads")
        .arg(threads.to_string())
        .arg("-std=c++17")
        .arg("-D__STRICT_ANSI__")
        .arg("--allow-unsupported-compiler")
        .arg("--expt-relaxed-constexpr")
        .arg("-I")
        .arg(cu_dir)
        .arg("-o")
        .arg(output)
        .arg(source);
    if cfg!(unix) {
        cmd.arg("-Xcompiler").arg("-fno-builtin");
    }

    let status = cmd
        .status()
        .unwrap_or_else(|e| panic!("Failed to invoke nvcc for {}: {e}", source.display()));

    if !status.success() {
        panic!("nvcc failed to compile {}", source.display());
    }

    assert!(
        output.exists(),
        "nvcc completed but did not emit {}",
        output.display()
    );
}

fn emit_stub_ptx(out_dir: &Path) {
    for &(_, ptx_name) in KERNELS {
        fs::write(out_dir.join(ptx_name), PTX_STUB)
            .unwrap_or_else(|e| panic!("Failed to write stub PTX {ptx_name}: {e}"));
    }
}

fn patch_ptx_version_any(path: &Path, new_ver: &str) {
    let Ok(text) = fs::read_to_string(path) else {
        return;
    };
    let mut replaced = false;
    let patched = text
        .lines()
        .map(|line| {
            if line.trim_start().starts_with(".version ") {
                replaced = true;
                format!(".version {new_ver}")
            } else {
                line.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n");

    if replaced {
        let had_trailing_newline = text.ends_with('\n');
        let output = if had_trailing_newline {
            format!("{patched}\n")
        } else {
            patched
        };
        fs::write(path, output)
            .unwrap_or_else(|e| panic!("Failed to patch PTX version in {}: {e}", path.display()));
    }
}

/// Raise `.version` only when the current value is lower than `min_ver`.
/// Leaves equal/higher versions (e.g. nvcc's native 9.2) unchanged.
fn ensure_min_ptx_version(path: &Path, min_ver: &str) {
    let Ok(text) = fs::read_to_string(path) else {
        return;
    };
    let Some(current) = text.lines().find_map(|line| {
        let t = line.trim_start();
        t.strip_prefix(".version ").map(str::trim)
    }) else {
        return;
    };
    if ptx_version_less(current, min_ver) {
        patch_ptx_version_any(path, min_ver);
    }
}

fn ptx_version_less(a: &str, b: &str) -> bool {
    let parse = |s: &str| -> (u32, u32) {
        let mut parts = s.split('.');
        let major = parts.next().and_then(|p| p.parse().ok()).unwrap_or(0);
        let minor = parts.next().and_then(|p| p.parse().ok()).unwrap_or(0);
        (major, minor)
    };
    parse(a) < parse(b)
}
