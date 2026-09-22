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

const FATBINS: &[(&str, &str)] = &[
    ("spiking_network.cu", "spiking_network_sm_120.fatbin"),
    ("vector_similarity.cu", "vector_similarity_sm_120.fatbin"),
    ("satsolver.cu", "satsolver_sm_120.fatbin"),
    ("ternary_gemm.cu", "ternary_gemm_sm_120.fatbin"),
];

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let cu_dir = manifest_dir.join("cu");
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let cuda_feature_enabled = env::var("CARGO_FEATURE_CUDA").is_ok();
    let arch = if cuda_feature_enabled {
        env::var("MYELIN_CUDA_ARCH").unwrap_or_else(|_| "sm_120".to_string())
    } else {
        "sm_120".to_string()
    };
    let (arch_major, arch_minor) = compute_capability_from_arch(&arch).unwrap_or_else(|| {
        panic!("MYELIN_CUDA_ARCH must look like sm_120 or compute_120, got \"{arch}\"")
    });
    fs::write(
        out_dir.join("compiled_cuda_capability.rs"),
        format!("ComputeCapability {{ major: {arch_major}, minor: {arch_minor} }}\n"),
    )
    .expect("write compiled CUDA capability");
    println!("cargo:rustc-env=MYELIN_COMPILED_CUDA_ARCH={arch}");

    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_NVCC");
    println!("cargo:rerun-if-env-changed=RUSTC");
    println!("cargo:rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");
    println!("cargo:rerun-if-env-changed=CARGO_CFG_TARGET_FEATURE");
    println!("cargo:rerun-if-env-changed=MYELIN_CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=MYELIN_PTX_VERSION");
    println!("cargo:rerun-if-env-changed=MYELIN_NVCC_THREADS");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=cu/common.cuh");
    println!("cargo:rerun-if-changed=cu/myelin_shim.cu");
    println!("cargo:rerun-if-changed=cu/myelin_shim.h");
    for &(cu_name, _) in KERNELS {
        println!("cargo:rerun-if-changed=cu/{cu_name}");
    }

    let saaq_feature_enabled = env::var("CARGO_FEATURE_SAAQ").is_ok();
    let opt_level = env::var("OPT_LEVEL").unwrap_or_else(|_| "unknown".to_string());
    println!("cargo:rustc-env=OPT_LEVEL={opt_level}");
    if let Some(version) = rustc_version() {
        println!("cargo:rustc-env=MYELIN_BUILD_RUSTC_VERSION={version}");
    }
    let rustflags = env::var("CARGO_ENCODED_RUSTFLAGS").unwrap_or_default();
    println!("cargo:rustc-env=MYELIN_BUILD_RUSTFLAGS={rustflags}");
    let target_features = env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();
    println!("cargo:rustc-env=MYELIN_BUILD_TARGET_FEATURES={target_features}");
    if let Ok(target) = env::var("TARGET") {
        println!("cargo:rustc-env=MYELIN_BUILD_TARGET={target}");
    }
    emit_cargo_profile_provenance();
    emit_git_provenance(&manifest_dir);
    if !cuda_feature_enabled {
        emit_stub_ptx(&out_dir);
        println!("cargo:warning=cuda feature not enabled; wrote stub PTX files");
        return;
    }

    let nvcc = find_nvcc();
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
        Some(v) => {
            println!("cargo:warning=using nvcc: {v}");
            println!("cargo:rustc-env=MYELIN_BUILD_NVCC_VERSION={v}");
        }
        None => println!("cargo:warning=using nvcc at {}", nvcc_path.display()),
    }

    let mut emitted_ptx_version = None;
    for &(cu_name, ptx_name) in KERNELS {
        let source = cu_dir.join(cu_name);
        let output = out_dir.join(ptx_name);
        compile_to_ptx(
            &nvcc_path,
            &cu_dir,
            &source,
            &output,
            &arch,
            &nvcc_feature_defines(cu_name, saaq_feature_enabled),
        );
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
        if let Some(version) = read_ptx_version(&output) {
            if let Some(previous) = emitted_ptx_version.as_deref() {
                assert_eq!(
                    previous, version,
                    "compiled kernels use different PTX versions"
                );
            } else {
                emitted_ptx_version = Some(version.to_string());
            }
        }
    }

    for &(cu_name, fatbin_name) in FATBINS {
        let source = cu_dir.join(cu_name);
        let output = out_dir.join(fatbin_name);
        compile_to_fatbin(
            &nvcc_path,
            &cu_dir,
            &source,
            &output,
            &arch,
            &nvcc_feature_defines(cu_name, saaq_feature_enabled),
        );
        println!("cargo:warning=compiled {cu_name} -> {fatbin_name} ({arch} SASS + PTX fallback)");
    }

    if saaq_feature_enabled {
        build_myelin_shim(&nvcc_path, &cu_dir, &out_dir, &arch);
        emit_cuda_runtime_linking(&nvcc_path);
    } else {
        println!("cargo:warning=saaq feature off; skipped myelin_shim.cu and cudart link");
    }
    println!("cargo:rustc-env=MYELIN_BUILD_CUDA_ARCH={arch}");
    if let Some(version) = emitted_ptx_version {
        println!("cargo:rustc-env=MYELIN_BUILD_PTX_VERSION={version}");
    }
}

fn emit_cargo_profile_provenance() {
    let profile_class = env::var("PROFILE").unwrap_or_else(|_| "unknown".to_string());
    let panic_strategy = env::var("CARGO_CFG_PANIC").unwrap_or_else(|_| "unwind".to_string());

    // Cargo exposes the effective profile *class* (debug/release), not the
    // selected custom profile name. Do not infer a name from the output
    // directory: built-in `bench`, for example, also writes to `release/`.
    // Effective profile settings are recorded at runtime from Cargo's exact
    // unit fingerprint instead of guessing LTO/codegen-unit defaults here.
    println!("cargo:rustc-env=MYELIN_BUILD_CARGO_PROFILE={profile_class}");
    println!("cargo:rustc-env=MYELIN_BUILD_PANIC_STRATEGY={panic_strategy}");
}

fn emit_git_provenance(manifest_dir: &Path) {
    if let Some(commit) = git_stdout(manifest_dir, &["rev-parse", "HEAD"]) {
        println!("cargo:rustc-env=MYELIN_BUILD_GIT_COMMIT={commit}");
    }
    if let Some(status) = git_stdout(
        manifest_dir,
        &["status", "--porcelain", "--untracked-files=no"],
    ) {
        println!(
            "cargo:rustc-env=MYELIN_BUILD_GIT_DIRTY={}",
            !status.is_empty()
        );
    }

    if let Some(files) = git_stdout_bytes(manifest_dir, &["ls-files", "-z"]) {
        for file in files
            .split(|byte| *byte == 0)
            .filter(|file| !file.is_empty())
        {
            let path = manifest_dir.join(String::from_utf8_lossy(file).as_ref());
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }

    for git_path in ["HEAD", "index", "packed-refs"] {
        emit_git_rerun_path(manifest_dir, git_path);
    }
    if let Some(symbolic_ref) = git_stdout(manifest_dir, &["symbolic-ref", "-q", "HEAD"]) {
        emit_git_rerun_path(manifest_dir, &symbolic_ref);
    }
}

fn emit_git_rerun_path(manifest_dir: &Path, git_path: &str) {
    let Some(path) = git_stdout(manifest_dir, &["rev-parse", "--git-path", git_path]) else {
        return;
    };
    let path = PathBuf::from(path);
    let path = if path.is_absolute() {
        path
    } else {
        manifest_dir.join(path)
    };
    println!("cargo:rerun-if-changed={}", path.display());
}

fn git_stdout(dir: &Path, args: &[&str]) -> Option<String> {
    let bytes = git_stdout_bytes(dir, args)?;
    String::from_utf8(bytes)
        .ok()
        .map(|output| output.trim().to_string())
}

fn git_stdout_bytes(dir: &Path, args: &[&str]) -> Option<Vec<u8>> {
    let output = Command::new("git")
        .args(args)
        .current_dir(dir)
        .output()
        .ok()?;
    output.status.success().then_some(output.stdout)
}

fn rustc_version() -> Option<String> {
    let rustc = env::var_os("RUSTC")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(exe_name("rustc")));
    let out = Command::new(rustc).arg("--version").output().ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8(out.stdout)
        .ok()
        .and_then(|s| s.lines().next().map(|line| line.trim().to_string()))
        .filter(|line| !line.is_empty())
}

fn compute_capability_from_arch(arch: &str) -> Option<(u32, u32)> {
    let suffix = arch
        .strip_prefix("sm_")
        .or_else(|| arch.strip_prefix("compute_"))?;
    let digits: String = suffix.chars().take_while(char::is_ascii_digit).collect();
    if digits.len() < 2 {
        return None;
    }
    let value = digits.parse::<u32>().ok()?;
    Some((value / 10, value % 10))
}

fn find_nvcc() -> Option<PathBuf> {
    if let Ok(path) = env::var("CUDA_NVCC") {
        let p = PathBuf::from(path);
        if is_nvcc_binary(&p) {
            return Some(resolve_nvcc_absolute(&p));
        }
    }

    for root_var in ["CUDA_HOME", "CUDA_PATH"] {
        if let Ok(root) = env::var(root_var) {
            let p = PathBuf::from(root).join("bin").join(exe_name("nvcc"));
            if is_nvcc_binary(&p) {
                return Some(resolve_nvcc_absolute(&p));
            }
        }
    }

    let candidate = PathBuf::from(exe_name("nvcc"));
    if is_nvcc_binary(&candidate) {
        Some(resolve_nvcc_absolute(&candidate))
    } else {
        None
    }
}

/// Prefer an absolute `nvcc` path so `parent()/parent()` can yield `lib64`.
fn resolve_nvcc_absolute(path: &Path) -> PathBuf {
    if path.is_absolute() {
        return path.to_path_buf();
    }
    if let Some(found) = which_on_path(path.as_os_str()) {
        return found;
    }
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

fn which_on_path(bin: &std::ffi::OsStr) -> Option<PathBuf> {
    let path_var = env::var_os("PATH")?;
    env::split_paths(&path_var).find_map(|dir| {
        let candidate = dir.join(bin);
        candidate.is_file().then_some(candidate)
    })
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
    String::from_utf8(out.stdout).ok().and_then(|s| {
        s.lines()
            .map(str::trim)
            .find(|line| line.contains("Cuda compilation tools, release "))
            .map(str::to_string)
    })
}

fn nvcc_feature_defines(cu_name: &str, saaq_feature_enabled: bool) -> Vec<String> {
    let mut defines = Vec::new();
    if saaq_feature_enabled && cu_name == "spiking_network.cu" {
        defines.push("-DMYELIN_SAAQ".to_string());
    }
    defines
}

fn compile_to_ptx(
    nvcc: &Path,
    cu_dir: &Path,
    source: &Path,
    output: &Path,
    arch: &str,
    extra_args: &[String],
) {
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
    for arg in extra_args {
        cmd.arg(arg);
    }
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

fn arch_gencode_parts(arch: &str) -> (String, String) {
    let digits = arch
        .strip_prefix("sm_")
        .or_else(|| arch.strip_prefix("compute_"))
        .unwrap_or("120");
    (format!("compute_{digits}"), format!("sm_{digits}"))
}

fn compile_to_fatbin(
    nvcc: &Path,
    cu_dir: &Path,
    source: &Path,
    output: &Path,
    arch: &str,
    extra_args: &[String],
) {
    let threads_raw = env::var("MYELIN_NVCC_THREADS").unwrap_or_else(|_| "0".to_string());
    let threads = threads_raw.parse::<usize>().unwrap_or_else(|_| {
        panic!("MYELIN_NVCC_THREADS must be a non-negative integer, got \"{threads_raw}\"")
    });
    let (compute, sm) = arch_gencode_parts(arch);

    let mut cmd = Command::new(nvcc);
    cmd.arg("-fatbin")
        .arg(format!("-gencode=arch={compute},code={sm}"))
        .arg(format!("-gencode=arch={compute},code={compute}"))
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
    for arg in extra_args {
        cmd.arg(arg);
    }
    if cfg!(unix) {
        cmd.arg("-Xcompiler").arg("-fno-builtin");
    }

    let status = cmd
        .status()
        .unwrap_or_else(|e| panic!("Failed to invoke nvcc fatbin for {}: {e}", source.display()));
    if !status.success() {
        panic!("nvcc failed to fatbin {}", source.display());
    }
    assert!(
        output.exists(),
        "nvcc completed but did not emit {}",
        output.display()
    );
}

fn nvcc_common_host_flags(cmd: &mut Command) {
    cmd.arg("-std=c++17")
        .arg("-D__STRICT_ANSI__")
        .arg("--allow-unsupported-compiler")
        .arg("--expt-relaxed-constexpr");
    if cfg!(unix) {
        cmd.arg("-Xcompiler").arg("-fno-builtin");
    }
}

fn build_myelin_shim(nvcc: &Path, cu_dir: &Path, out_dir: &Path, arch: &str) {
    let source = cu_dir.join("myelin_shim.cu");
    let object = out_dir.join("myelin_shim.o");
    let threads_raw = env::var("MYELIN_NVCC_THREADS").unwrap_or_else(|_| "0".to_string());
    let threads = threads_raw.parse::<usize>().unwrap_or_else(|_| {
        panic!("MYELIN_NVCC_THREADS must be a non-negative integer, got \"{threads_raw}\"")
    });

    let mut cmd = Command::new(nvcc);
    cmd.arg("-c")
        .arg(format!("-arch={arch}"))
        .arg("-O3")
        .arg("--use_fast_math")
        .arg("--restrict")
        .arg("--threads")
        .arg(threads.to_string());
    nvcc_common_host_flags(&mut cmd);
    cmd.arg("-DMYELIN_SAAQ")
        .arg("-I")
        .arg(cu_dir)
        .arg("-Xcompiler")
        .arg("-fPIC")
        .arg("-o")
        .arg(&object)
        .arg(&source);

    let status = cmd
        .status()
        .unwrap_or_else(|e| panic!("Failed to invoke nvcc for myelin_shim.cu: {e}"));
    if !status.success() {
        panic!("nvcc failed to compile myelin_shim.cu");
    }

    cc::Build::new()
        .cpp(true)
        .object(&object)
        .compile("myelin_shim");
    println!("cargo:warning=compiled myelin_shim.cu → libmyelin_shim.a");
}

fn emit_cuda_runtime_linking(nvcc: &Path) {
    for search_dir in cuda_library_search_paths(nvcc) {
        println!("cargo:rustc-link-search=native={}", search_dir.display());
    }
    println!("cargo:rustc-link-lib=dylib=cudart");
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
}

fn cuda_library_search_paths(nvcc: &Path) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(dir) = libcudart_search_dir(nvcc) {
        candidates.push(dir);
    }
    for env_var in ["CUDA_HOME", "CUDA_PATH"] {
        if let Ok(root) = env::var(env_var) {
            let root = PathBuf::from(root);
            candidates.push(root.join("lib64"));
            candidates.push(root.join("lib"));
            candidates.push(root.join("targets").join("x86_64-linux").join("lib"));
        }
    }
    if let Some(root) = nvcc.parent().and_then(|bin| bin.parent()) {
        candidates.push(root.join("lib64"));
        candidates.push(root.join("lib"));
        candidates.push(root.join("targets").join("x86_64-linux").join("lib"));
    }
    candidates.push(PathBuf::from("/usr/local/cuda/lib64"));
    candidates.push(PathBuf::from("/usr/local/cuda/lib"));
    candidates.push(PathBuf::from("/usr/lib/x86_64-linux-gnu"));

    let mut deduped = Vec::new();
    for candidate in candidates {
        if candidate.exists() && !deduped.iter().any(|existing| existing == &candidate) {
            deduped.push(candidate);
        }
    }
    deduped
}

fn libcudart_search_dir(nvcc: &Path) -> Option<PathBuf> {
    let names: &[&str] = if cfg!(windows) {
        &["cudart.lib", "libcudart.lib"]
    } else {
        &["libcudart.so", "libcudart.so.13", "libcudart.so.12"]
    };
    for name in names {
        let out = match Command::new(nvcc)
            .arg(format!("--print-file-name={name}"))
            .output()
        {
            Ok(out) => out,
            Err(_) => continue,
        };
        if !out.status.success() {
            continue;
        }
        let printed = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if printed.is_empty() || printed == *name {
            continue;
        }
        let path = PathBuf::from(printed);
        if path.is_file() {
            return path.parent().map(Path::to_path_buf);
        }
        if path.is_dir() {
            return Some(path);
        }
    }
    None
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

fn read_ptx_version(path: &Path) -> Option<String> {
    let text = fs::read_to_string(path).ok()?;
    text.lines().find_map(|line| {
        line.trim_start()
            .strip_prefix(".version ")
            .map(str::trim)
            .filter(|version| !version.is_empty())
            .map(str::to_string)
    })
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
