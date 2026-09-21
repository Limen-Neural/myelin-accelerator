// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

//! GPU error types shared by the CUDA backend and the CPU stub.

use crate::capability::{FallbackReason, sanitize_diagnostic};
use std::fmt;

pub type GpuResult<T> = Result<T, GpuError>;

#[derive(Debug)]
pub enum GpuError {
    /// CUDA context / device initialisation failed.
    InitFailed(String),
    /// PTX module failed to load or JIT-compile.
    ModuleLoadFailed(String),
    /// A kernel function was not found in the loaded module.
    KernelNotFound(String),
    /// Memory allocation or copy failed.
    MemoryError(String),
    /// Kernel launch failed.
    LaunchFailed(String),
    /// Generic CUDA error forwarded from `cust`.
    CudaError(String),
    /// GPU is not available on this system (legacy catch-all).
    NoGpu,
    /// GPU was required or a launch ran without a usable device.
    Unavailable {
        reason: FallbackReason,
        /// Sanitized, path-free diagnostic.
        detail: String,
    },
    /// Caller-supplied arguments are invalid.
    InvalidInput(String),
}

impl GpuError {
    /// GPU unavailable with a stable reason and sanitized detail.
    pub fn unavailable(reason: FallbackReason, detail: impl Into<String>) -> Self {
        Self::Unavailable {
            reason,
            detail: sanitize_diagnostic(&detail.into()),
        }
    }

    /// Invalid launch arguments with a sanitized detail string.
    pub fn invalid_input(detail: impl Into<String>) -> Self {
        Self::InvalidInput(sanitize_diagnostic(&detail.into()))
    }

    /// Map this error to a fallback/reason code when one applies.
    pub fn fallback_reason(&self) -> Option<FallbackReason> {
        match self {
            Self::Unavailable { reason, .. } => Some(*reason),
            Self::InvalidInput(_) => Some(FallbackReason::InvalidInput),
            Self::NoGpu => {
                #[cfg(feature = "cuda")]
                {
                    None
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Some(FallbackReason::CudaFeatureNotBuilt)
                }
            }
            Self::InitFailed(_) => Some(FallbackReason::DriverRuntimeFailure),
            Self::ModuleLoadFailed(_) | Self::KernelNotFound(_) => {
                Some(FallbackReason::KernelSpecializationUnavailable)
            }
            Self::MemoryError(_) | Self::LaunchFailed(_) | Self::CudaError(_) => None,
        }
    }
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuError::InitFailed(s) => write!(f, "GPU init failed: {s}"),
            GpuError::ModuleLoadFailed(s) => write!(f, "PTX module load failed: {s}"),
            GpuError::KernelNotFound(s) => write!(f, "Kernel not found: {s}"),
            GpuError::MemoryError(s) => write!(f, "GPU memory error: {s}"),
            GpuError::LaunchFailed(s) => write!(f, "Kernel launch failed: {s}"),
            GpuError::CudaError(s) => write!(f, "CUDA error: {s}"),
            GpuError::NoGpu => {
                #[cfg(feature = "cuda")]
                {
                    write!(f, "No GPU available (or built without `cuda` feature)")
                }
                #[cfg(not(feature = "cuda"))]
                {
                    write!(f, "No GPU available (built without `cuda` feature)")
                }
            }
            GpuError::Unavailable { reason, detail } => {
                write!(
                    f,
                    "GPU unavailable ({}): {}",
                    reason.code(),
                    sanitize_diagnostic(detail)
                )
            }
            GpuError::InvalidInput(s) => {
                write!(f, "invalid input: {}", sanitize_diagnostic(s))
            }
        }
    }
}

impl std::error::Error for GpuError {}

#[cfg(feature = "cuda")]
impl From<cust::error::CudaError> for GpuError {
    fn from(e: cust::error::CudaError) -> Self {
        GpuError::CudaError(sanitize_diagnostic(&format!("{e:?}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unavailable_display_uses_stable_reason_code() {
        let err = GpuError::unavailable(
            FallbackReason::CudaFeatureNotBuilt,
            "crate built without the cuda feature",
        );
        assert_eq!(
            err.to_string(),
            "GPU unavailable (cuda_feature_not_built): crate built without the cuda feature"
        );
        assert_eq!(
            err.fallback_reason(),
            Some(FallbackReason::CudaFeatureNotBuilt)
        );
    }

    #[test]
    fn invalid_input_maps_to_reason_code() {
        let err = GpuError::invalid_input("n_vars must be >= 0, got -1");
        assert_eq!(
            err.to_string(),
            "invalid input: n_vars must be >= 0, got -1"
        );
        assert_eq!(err.fallback_reason(), Some(FallbackReason::InvalidInput));
    }

    #[test]
    fn unavailable_sanitizes_paths() {
        let err = GpuError::unavailable(
            FallbackReason::DriverRuntimeFailure,
            "failed /home/alice/cuda/lib",
        );
        match err {
            GpuError::Unavailable { detail, .. } => {
                assert!(!detail.contains("/home/alice"));
                assert!(detail.contains("<path>"));
            }
            other => panic!("expected Unavailable, got {other}"),
        }
    }

    #[test]
    fn display_sanitizes_public_variant_payloads() {
        let unavailable = GpuError::Unavailable {
            reason: FallbackReason::DriverRuntimeFailure,
            detail: "module=/home/alice/private.ptx token=secret".to_string(),
        };
        let invalid = GpuError::InvalidInput(
            "file=\"/Users/bob/My Models/input.bin\" password=hunter2".to_string(),
        );

        assert_eq!(
            unavailable.to_string(),
            "GPU unavailable (driver_runtime_failure): module=<path> token=<redacted>"
        );
        assert_eq!(
            invalid.to_string(),
            "invalid input: file=<path> password=<redacted>"
        );
    }
}
