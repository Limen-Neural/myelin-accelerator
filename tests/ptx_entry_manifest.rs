// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Drift guard: `scripts/ptx_entries.txt` (read by `check_ptx_entries.sh` in
//! both CI jobs) must stay in sync with the kernel name lists registered by
//! `KernelModule::load` in `src/gpu/kernel.rs`. Runs on the CPU-safe path.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

fn manifest_dir() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

/// Extract the `"..."` entries of `const <name>: &[&str] = &[ ... ];`.
fn kernel_list<'a>(source: &'a str, const_name: &str) -> BTreeSet<&'a str> {
    let marker = format!("const {const_name}");
    let decl = source
        .find(&marker)
        .unwrap_or_else(|| panic!("{const_name} not found in kernel.rs"));
    // rustfmt may wrap before the array literal; find `[` after the `=`.
    let eq = source[decl..].find('=').expect("malformed const") + decl;
    let body_start = source[eq..].find('[').expect("missing array") + eq + 1;
    let end = source[body_start..]
        .find(']')
        .expect("unterminated kernel const");
    source[body_start..body_start + end]
        .split('"')
        .skip(1)
        .step_by(2)
        .collect()
}

#[test]
fn ptx_entry_manifest_matches_kernel_registration() {
    let kernel_rs = manifest_dir().join("src/gpu/kernel.rs");
    let source = fs::read_to_string(&kernel_rs).expect("read kernel.rs");

    let manifest = fs::read_to_string(manifest_dir().join("scripts/ptx_entries.txt"))
        .expect("read scripts/ptx_entries.txt");

    let expected: &[(&str, &str)] = &[
        ("spiking_network_sm_120.ptx", "SPIKING_KERNELS"),
        ("vector_similarity_sm_120.ptx", "VECTOR_SIMILARITY_KERNELS"),
        ("satsolver_sm_120.ptx", "SATSOLVER_KERNELS"),
        ("ternary_gemm_sm_120.ptx", "TERNARY_GEMM_KERNELS"),
    ];

    let mut seen_files = BTreeSet::new();
    for line in manifest.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (file, kernels) = line
            .split_once(':')
            .unwrap_or_else(|| panic!("malformed manifest line: {line}"));
        seen_files.insert(file.to_string());
        let listed: BTreeSet<&str> = kernels.split_whitespace().collect();
        let (ptx_file, const_name) = expected
            .iter()
            .find(|(f, _)| *f == file)
            .unwrap_or_else(|| panic!("manifest lists unknown PTX file {file}"));
        assert_eq!(file, *ptx_file);
        let registered = kernel_list(&source, const_name);
        let listed: BTreeSet<String> = listed.iter().map(|s| s.to_string()).collect();
        let registered: BTreeSet<String> = registered.iter().map(|s| s.to_string()).collect();
        assert_eq!(
            listed, registered,
            "ptx_entries.txt {file} differs from {const_name} in kernel.rs"
        );
    }
    assert_eq!(
        seen_files.len(),
        expected.len(),
        "manifest must cover every registered PTX module"
    );
}
