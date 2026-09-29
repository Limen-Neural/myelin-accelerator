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
                    Some(FallbackReason::DeviceUnavailable)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Some(FallbackReason::CudaFeatureNotBuilt)
                }
            }
            Self::InitFailed(_) | Self::CudaError(_) => Some(FallbackReason::DriverRuntimeFailure),
            Self::ModuleLoadFailed(_) | Self::KernelNotFound(_) => {
                Some(FallbackReason::KernelSpecializationUnavailable)
            }
            Self::MemoryError(_) | Self::LaunchFailed(_) => None,
        }
    }
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuError::InitFailed(s) => {
                write!(f, "GPU init failed: {}", sanitize_diagnostic(s))
            }
            GpuError::ModuleLoadFailed(s) => {
                write!(f, "PTX module load failed: {}", sanitize_diagnostic(s))
            }
            GpuError::KernelNotFound(s) => {
                write!(f, "Kernel not found: {}", sanitize_diagnostic(s))
            }
            GpuError::MemoryError(s) => {
                write!(f, "GPU memory error: {}", sanitize_diagnostic(s))
            }
            GpuError::LaunchFailed(s) => {
                write!(f, "Kernel launch failed: {}", sanitize_diagnostic(s))
            }
            GpuError::CudaError(s) => {
                write!(f, "CUDA error: {}", sanitize_diagnostic(s))
            }
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

/// Checked host-side byte size for `GpuBuffer::<T>::alloc(len)`, shared by the
/// CPU stub and the CUDA backend so overflow handling cannot drift (GH #48).
///
/// Returns the total byte footprint when `size_of::<T>() * len` is defined and
/// within the `isize::MAX` single-allocation ceiling; otherwise returns a
/// categorised [`GpuError::MemoryError`] instead of panicking. Callers still
/// perform their own fallible reservation after this check.
pub(crate) fn checked_alloc_bytes<T>(len: usize) -> GpuResult<usize> {
    let elem_size = std::mem::size_of::<T>();
    match elem_size.checked_mul(len) {
        Some(bytes) if bytes <= isize::MAX as usize => Ok(bytes),
        _ => Err(GpuError::MemoryError(format!(
            "alloc({len}): size overflow, {len} elements of {elem_size} bytes exceeds isize::MAX"
        ))),
    }
}

#[cfg(feature = "cuda")]
impl From<cust::error::CudaError> for GpuError {
    fn from(e: cust::error::CudaError) -> Self {
        use cust::error::CudaError;

        let detail = sanitize_diagnostic(&format!("{e:?}"));
        match e {
            CudaError::NoDevice | CudaError::InvalidDevice => {
                GpuError::unavailable(FallbackReason::DeviceUnavailable, detail)
            }
            CudaError::InvalidImage
            | CudaError::NoBinaryForGpu
            | CudaError::InvalidPtx
            | CudaError::InvalidSource => GpuError::ModuleLoadFailed(detail),
            _ => GpuError::CudaError(detail),
        }
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
    fn legacy_no_gpu_and_cuda_errors_have_stable_reasons() {
        let expected_no_gpu = if cfg!(feature = "cuda") {
            FallbackReason::DeviceUnavailable
        } else {
            FallbackReason::CudaFeatureNotBuilt
        };
        assert_eq!(GpuError::NoGpu.fallback_reason(), Some(expected_no_gpu));
        assert_eq!(
            GpuError::CudaError("InvalidContext".into()).fallback_reason(),
            Some(FallbackReason::DriverRuntimeFailure)
        );
        assert_eq!(
            GpuError::ModuleLoadFailed("InvalidPtx".into()).fallback_reason(),
            Some(FallbackReason::KernelSpecializationUnavailable)
        );
        assert_eq!(
            GpuError::KernelNotFound("lif_step".into()).fallback_reason(),
            Some(FallbackReason::KernelSpecializationUnavailable)
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cust_errors_preserve_device_specialization_and_runtime_categories() {
        use cust::error::CudaError;

        assert_eq!(
            GpuError::from(CudaError::NoDevice).fallback_reason(),
            Some(FallbackReason::DeviceUnavailable)
        );
        assert_eq!(
            GpuError::from(CudaError::InvalidPtx).fallback_reason(),
            Some(FallbackReason::KernelSpecializationUnavailable)
        );
        assert_eq!(
            GpuError::from(CudaError::InvalidContext).fallback_reason(),
            Some(FallbackReason::DriverRuntimeFailure)
        );
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

        let payload = "token=supersecret /home/alice/key".to_string();
        let cases = [
            (
                GpuError::InitFailed(payload.clone()),
                "GPU init failed: token=<redacted> <path>",
            ),
            (
                GpuError::ModuleLoadFailed(payload.clone()),
                "PTX module load failed: token=<redacted> <path>",
            ),
            (
                GpuError::KernelNotFound(payload.clone()),
                "Kernel not found: token=<redacted> <path>",
            ),
            (
                GpuError::MemoryError(payload.clone()),
                "GPU memory error: token=<redacted> <path>",
            ),
            (
                GpuError::LaunchFailed(payload.clone()),
                "Kernel launch failed: token=<redacted> <path>",
            ),
            (
                GpuError::CudaError(payload),
                "CUDA error: token=<redacted> <path>",
            ),
        ];
        for (err, expected) in cases {
            assert_eq!(err.to_string(), expected);
        }
    }

    #[test]
    fn checked_alloc_bytes_accepts_zero_ordinary_and_zst() {
        assert_eq!(checked_alloc_bytes::<u64>(0).unwrap(), 0);
        assert_eq!(checked_alloc_bytes::<u64>(4).unwrap(), 32);
        assert_eq!(checked_alloc_bytes::<u8>(16).unwrap(), 16);
        // Zero-sized types occupy no bytes at any length.
        assert_eq!(checked_alloc_bytes::<()>(0).unwrap(), 0);
        assert_eq!(checked_alloc_bytes::<()>(usize::MAX).unwrap(), 0);
    }

    #[test]
    fn checked_alloc_bytes_rejects_overflow_and_over_ceiling() {
        for len in [usize::MAX, isize::MAX as usize, isize::MAX as usize / 8 + 1] {
            assert_checked_alloc_overflow::<u64>(len);
        }
        for len in [usize::MAX, isize::MAX as usize + 1] {
            assert_checked_alloc_overflow::<u8>(len);
        }
        // Exactly at the ceiling is still accepted; one byte past is not.
        assert_eq!(
            checked_alloc_bytes::<u8>(isize::MAX as usize).unwrap(),
            isize::MAX as usize
        );
    }

    /// Assert `checked_alloc_bytes::<T>(len)` fails with the overflow `MemoryError`.
    fn assert_checked_alloc_overflow<T>(len: usize) {
        match checked_alloc_bytes::<T>(len).unwrap_err() {
            GpuError::MemoryError(msg) => assert!(msg.contains("size overflow"), "msg: {msg}"),
            other => panic!("expected MemoryError, got {other}"),
        }
    }
}
