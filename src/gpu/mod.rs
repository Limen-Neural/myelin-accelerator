// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

// ════════════════════════════════════════════════════════════════════
//  gpu/mod.rs — GPU sub-module declarations and public re-exports
// ════════════════════════════════════════════════════════════════════

pub mod accelerator;
pub mod context;
pub mod error;
pub mod kernel;
pub mod memory;

pub use crate::capability::{
    Backend, CapabilityFacts, CapabilityReport, ComputeCapability, ExecutionPolicy, FallbackReason,
    FallbackRecord, KernelAvailability,
};
pub use accelerator::GpuAccelerator;
pub use context::GpuContext;
pub use error::{GpuError, GpuResult};
pub use kernel::KernelModule;
pub use memory::GpuBuffer;
