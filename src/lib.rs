// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

// myelin-accelerator: safe Rust FFI wrappers around CUDA spiking-network kernels.
pub mod bitpacking;
pub mod capability;
mod error;

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

pub(crate) fn host_facts() -> capability::CapabilityFacts {
    #[cfg(feature = "cuda")]
    {
        gpu::context::host_facts()
    }
    #[cfg(not(feature = "cuda"))]
    {
        gpu_stub::host_facts()
    }
}
