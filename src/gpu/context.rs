// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

// ════════════════════════════════════════════════════════════════════
//  gpu/context.rs — CUDA device context initialisation
// ════════════════════════════════════════════════════════════════════

use crate::capability::{
    CapabilityFacts, ComputeCapability, FallbackReason, KernelAvailability, sanitize_diagnostic,
};
use crate::gpu::error::{GpuError, GpuResult};
use cust::context::Context;
use cust::device::{Device, DeviceAttribute};

/// Owns a CUDA primary context for device 0.
pub struct GpuContext {
    pub(crate) _ctx: Context,
    pub(crate) compute_capability: Option<ComputeCapability>,
}

impl GpuContext {
    /// Initialise CUDA and create a context on the first available device.
    pub fn init() -> GpuResult<Self> {
        cust::init(cust::CudaFlags::empty()).map_err(|e| {
            GpuError::InitFailed(sanitize_diagnostic(&format!("cust::init: {e:?}")))
        })?;

        let device = Device::get_device(0).map_err(device_lookup_failure)?;

        let compute_capability = query_compute_capability(&device);

        let ctx = Context::new(device).map_err(|e| {
            GpuError::InitFailed(sanitize_diagnostic(&format!("Context::new: {e:?}")))
        })?;

        Ok(Self {
            _ctx: ctx,
            compute_capability,
        })
    }

    /// Returns `true` when a CUDA device is accessible.
    pub fn is_available() -> bool {
        cust::init(cust::CudaFlags::empty()).is_ok() && Device::get_device(0).is_ok()
    }

    /// Device 0 compute capability, if the driver reported it.
    pub fn compute_capability(&self) -> Option<ComputeCapability> {
        self.compute_capability
    }
}

pub(crate) fn host_facts() -> CapabilityFacts {
    let kernels = KernelAvailability::compiled_unverified();

    if cust::init(cust::CudaFlags::empty()).is_err() {
        return CapabilityFacts {
            cuda_built: true,
            runtime_available: false,
            device_available: false,
            compute_capability: None,
            kernels,
        };
    }

    let Ok(device) = Device::get_device(0) else {
        return CapabilityFacts {
            cuda_built: true,
            runtime_available: true,
            device_available: false,
            compute_capability: None,
            kernels,
        };
    };

    CapabilityFacts {
        cuda_built: true,
        runtime_available: true,
        device_available: true,
        compute_capability: query_compute_capability(&device),
        kernels,
    }
}

fn query_compute_capability(device: &Device) -> Option<ComputeCapability> {
    let major = device
        .get_attribute(DeviceAttribute::ComputeCapabilityMajor)
        .ok()?;
    let minor = device
        .get_attribute(DeviceAttribute::ComputeCapabilityMinor)
        .ok()?;
    Some(ComputeCapability {
        major: u32::try_from(major).ok()?,
        minor: u32::try_from(minor).ok()?,
    })
}

fn device_lookup_failure(error: cust::error::CudaError) -> GpuError {
    use cust::error::CudaError;

    let detail = format!("get_device(0): {error:?}");
    match error {
        CudaError::NoDevice | CudaError::InvalidDevice => {
            GpuError::unavailable(FallbackReason::DeviceUnavailable, detail)
        }
        _ => GpuError::InitFailed(sanitize_diagnostic(&detail)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_device_maps_to_device_unavailable() {
        for cuda_error in [
            cust::error::CudaError::NoDevice,
            cust::error::CudaError::InvalidDevice,
        ] {
            let error = device_lookup_failure(cuda_error);
            assert_eq!(
                error.fallback_reason(),
                Some(FallbackReason::DeviceUnavailable),
                "{cuda_error:?}"
            );
        }
    }

    #[test]
    fn non_absence_device_lookup_maps_to_driver_runtime_failure() {
        let error = device_lookup_failure(cust::error::CudaError::InvalidContext);
        assert_eq!(
            error.fallback_reason(),
            Some(FallbackReason::DriverRuntimeFailure)
        );
        assert!(matches!(error, GpuError::InitFailed(_)));
    }
}
