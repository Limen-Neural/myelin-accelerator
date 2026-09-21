// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Default / crates.io surface must not expose fused routing / SAAQ APIs.

#![cfg(not(feature = "saaq"))]

#[test]
fn default_surface_does_not_compile_fused_module() {
    // This file intentionally does not `use myelin_accelerator::fused`.
    // Host fused APIs exist only with `--features saaq`.
    let _acc = myelin_accelerator::GpuAccelerator::new();
}
