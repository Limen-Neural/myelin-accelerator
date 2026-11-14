// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Drift guard: `scripts/ptx_entries.txt` (read by `check_ptx_entries.sh` in
//! both CI jobs) must stay in sync with the kernel names registered by
//! `KernelModule::load` in `src/gpu/kernel.rs`. Runs on the CPU-safe path.
//!
//! The expected module set is derived from kernel.rs itself — every
//! `MODULE_SPECS` entry (`ptx`, `name`, `funcs`) — so registering a new module
//! without adding it to the manifest fails this test.

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

fn manifest_dir() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn identifiers(source: &str) -> impl Iterator<Item = &str> {
    source
        .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .filter(|tok| !tok.is_empty() && tok.chars().next().unwrap().is_ascii_alphabetic())
}

/// Map each `static <NAME>_PTX` to its embedded PTX filename, e.g.
/// `static SPIKING_NETWORK_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/spiking_network_sm_120.ptx"));`
fn ptx_statics(source: &str) -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();
    let mut rest = source;
    while let Some(idx) = rest.find("static ") {
        rest = &rest[idx + "static ".len()..];
        let name_end = rest
            .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .unwrap_or(0);
        let name = &rest[..name_end];
        if !name.ends_with("_PTX") {
            continue;
        }
        let window = &rest[..rest.len().min(400)];
        // Filename is the `"/<file>.ptx"` literal after `env!("OUT_DIR")`.
        let file = window
            .find("OUT_DIR")
            .and_then(|o| window[o..].find("\"/").map(|q| o + q + 2))
            .and_then(|q| window[q..].find('"').map(|e| &window[q..q + e]));
        if let Some(file) = file {
            map.insert(name.to_string(), file.trim_start_matches('/').to_string());
        }
    }
    map
}

/// Read `MODULE_SPECS` entries and return (ptx file, kernels const) pairs.
fn registered_modules(source: &str, ptx_files: &BTreeMap<String, String>) -> Vec<(String, String)> {
    let marker = "const MODULE_SPECS";
    let decl = source
        .find(marker)
        .unwrap_or_else(|| panic!("{marker} not found in kernel.rs"));
    let eq = source[decl..].find('=').expect("malformed MODULE_SPECS") + decl;
    let body_start = source[eq..].find('[').expect("missing MODULE_SPECS array") + eq + 1;
    let body_end = source[body_start..]
        .find(']')
        .expect("unterminated MODULE_SPECS")
        + body_start;
    let body = &source[body_start..body_end];

    let mut out = Vec::new();
    let mut ptx = None;
    for tok in identifiers(body) {
        if ptx.is_none() && ptx_files.contains_key(tok) {
            ptx = Some(tok.to_string());
        } else if tok.ends_with("_KERNELS") {
            let ptx = ptx.take().expect("MODULE_SPECS entry without PTX static");
            out.push((ptx_files[&ptx].clone(), tok.to_string()));
        }
    }
    out
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
    let ptx_files = ptx_statics(&source);
    let registered = registered_modules(&source, &ptx_files);
    assert!(!registered.is_empty(), "no load_and_map calls found");

    let manifest = fs::read_to_string(manifest_dir().join("scripts/ptx_entries.txt"))
        .expect("read scripts/ptx_entries.txt");
    let mut manifest_files = BTreeMap::new();
    for line in manifest.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (file, kernels) = line
            .split_once(':')
            .unwrap_or_else(|| panic!("malformed manifest line: {line}"));
        manifest_files.insert(
            file.trim().to_string(),
            kernels
                .split_whitespace()
                .map(str::to_string)
                .collect::<BTreeSet<_>>(),
        );
    }

    let registered_files: BTreeSet<&String> = registered.iter().map(|(f, _)| f).collect();
    let manifest_file_set: BTreeSet<&String> = manifest_files.keys().collect();
    assert_eq!(
        manifest_file_set, registered_files,
        "manifest PTX files differ from modules registered in kernel.rs"
    );

    for (file, const_name) in &registered {
        let listed = &manifest_files[file];
        let registered_names: BTreeSet<String> = kernel_list(&source, const_name)
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(
            *listed, registered_names,
            "ptx_entries.txt {file} differs from {const_name} in kernel.rs"
        );
    }
}
