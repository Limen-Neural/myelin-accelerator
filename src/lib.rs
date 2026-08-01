// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

// myelin-accelerator: safe Rust FFI wrappers around CUDA spiking-network kernels.
pub mod bitpacking;

#[cfg(not(feature = "cuda"))]
pub mod gpu_stub;
#[cfg(not(feature = "cuda"))]
pub use gpu_stub as gpu;
#[cfg(feature = "cuda")]
pub mod gpu;

// Re-export the main public API at the crate root for ergonomic use.
#[cfg(feature = "cuda")]
pub use gpu::{GpuAccelerator, GpuBuffer, GpuContext, GpuError, KernelModule};
#[cfg(not(feature = "cuda"))]
pub use gpu_stub::{GpuAccelerator, GpuBuffer, GpuContext, GpuError, KernelModule};
