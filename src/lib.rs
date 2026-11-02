// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Rust/CUDA acceleration primitives for spiking neural networks (SNN) and
//! neuromorphic workloads: safe Rust wrappers around first-party CUDA kernels
//! covering spiking-network dynamics, routing, SAT search, and packed ternary
//! GEMV/GEMM. The `cuda` feature targets `sm_120` (NVIDIA Blackwell); the
//! default build is CPU-only and needs no CUDA toolkit or device.
//!
//! This is a low-level compute layer, not a complete SNN framework or a
//! general model-compatibility layer.
pub mod bench;
pub mod bitpacking;
pub mod capability;
mod error;
pub mod oracle;

#[cfg(not(feature = "cuda"))]
pub mod gpu_stub;
#[cfg(not(feature = "cuda"))]
pub use gpu_stub as gpu;
#[cfg(feature = "cuda")]
pub mod gpu;

// Re-export the main public API at the crate root for ergonomic use.
pub use capability::{
    Backend, CapabilityFacts, CapabilityReport, ComputeCapability, ExecutionPolicy, FallbackReason,
    FallbackRecord, KernelAvailability, evaluate_capabilities, probe_capabilities,
    sanitize_diagnostic,
};
#[cfg(feature = "cuda")]
pub use gpu::{GpuAccelerator, GpuBuffer, GpuContext, GpuError, KernelModule};
#[cfg(not(feature = "cuda"))]
pub use gpu_stub::{GpuAccelerator, GpuBuffer, GpuContext, GpuError, KernelModule};

#[cfg(feature = "cuda")]
pub(crate) fn host_facts() -> capability::CapabilityFacts {
    gpu::context::host_facts()
}
