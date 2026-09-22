// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

pub use crate::capability::{
    Backend, CapabilityFacts, CapabilityReport, ComputeCapability, ExecutionPolicy, FallbackReason,
    FallbackRecord, KernelAvailability,
};
use crate::capability::{apply_failure_to_facts, evaluate_capabilities};
pub use crate::error::{GpuError, GpuResult};
#[cfg(feature = "saaq")]
use crate::gif::SnapshotChannels;

pub(crate) fn host_facts() -> CapabilityFacts {
    CapabilityFacts::not_built()
}

#[derive(Debug)]
pub struct GpuContext;
impl GpuContext {
    pub fn init() -> GpuResult<Self> {
        Err(GpuError::NoGpu)
    }
    pub fn is_available() -> bool {
        false
    }

    /// Device 0 compute capability; unavailable in a CPU-only build.
    pub fn compute_capability(&self) -> Option<ComputeCapability> {
        None
    }
}

#[derive(Debug)]
pub struct KernelModule;
#[derive(Debug)]
pub struct Function;
impl KernelModule {
    pub fn load() -> GpuResult<Self> {
        Err(GpuError::NoGpu)
    }
    pub fn load_satsolver() -> GpuResult<Self> {
        Err(GpuError::NoGpu)
    }
    pub fn get_function(&self, _: &str) -> GpuResult<Function> {
        Err(GpuError::NoGpu)
    }
}

pub struct GpuBuffer<T> {
    data: Vec<T>,
}
impl<T: Default + Clone> GpuBuffer<T> {
    pub fn alloc(len: usize) -> GpuResult<Self> {
        Ok(Self {
            data: vec![T::default(); len],
        })
    }
    pub fn from_slice(data: &[T]) -> GpuResult<Self> {
        Ok(Self {
            data: data.to_vec(),
        })
    }
    pub fn to_vec(&self) -> GpuResult<Vec<T>> {
        Ok(self.data.clone())
    }
    pub fn upload(&mut self, data: &[T]) -> GpuResult<()> {
        if data.len() != self.data.len() {
            return Err(GpuError::MemoryError(format!(
                "upload: length mismatch, buffer has {} elements but input has {}",
                self.data.len(),
                data.len()
            )));
        }
        self.data.clone_from_slice(data);
        Ok(())
    }
    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    pub fn as_device_ptr(&self) -> *const T {
        self.data.as_ptr()
    }
}

impl GpuBuffer<f32> {
    /// Zero the first `count` elements; leaves any tail untouched (host stub).
    ///
    /// Matches the CUDA `GpuBuffer<f32>::zero_prefix` surface so callers do not
    /// depend on feature-specific method resolution.
    pub fn zero_prefix(&mut self, count: usize) -> GpuResult<()> {
        if count > self.data.len() {
            return Err(GpuError::MemoryError(format!(
                "zero_prefix: count {count} > buffer len {}",
                self.data.len()
            )));
        }
        for slot in &mut self.data[..count] {
            *slot = 0.0;
        }
        Ok(())
    }
}

pub struct GpuAccelerator {
    capabilities: CapabilityReport,
}

impl GpuAccelerator {
    /// Construct with [`ExecutionPolicy::PreferGpu`] (caller-approved CPU fallback).
    pub fn new() -> Self {
        match Self::with_policy(ExecutionPolicy::PreferGpu) {
            Ok(acc) => acc,
            Err(_) => unreachable!("PreferGpu construction is infallible"),
        }
    }

    /// Fail closed: never return a CPU-backend accelerator.
    pub fn require_gpu() -> GpuResult<Self> {
        Self::with_policy(ExecutionPolicy::RequireGpu)
    }

    /// Construct under an explicit execution policy.
    pub fn with_policy(policy: ExecutionPolicy) -> GpuResult<Self> {
        let mut facts = host_facts();
        apply_failure_to_facts(&mut facts, FallbackReason::CudaFeatureNotBuilt);
        let capabilities = evaluate_capabilities(&facts);
        match policy {
            ExecutionPolicy::PreferGpu => Ok(Self { capabilities }),
            ExecutionPolicy::RequireGpu => {
                let fb = capabilities.fallback.clone().unwrap_or_else(|| {
                    FallbackRecord::cpu(
                        FallbackReason::CudaFeatureNotBuilt,
                        "GPU was required but the cuda feature is not enabled",
                    )
                });
                Err(GpuError::unavailable(fb.reason, fb.detail))
            }
        }
    }

    pub fn is_ready(&self) -> bool {
        false
    }

    /// Always `false` on the CPU stub (no context, stream, or modules).
    pub fn kernels_ready(&self) -> bool {
        false
    }

    pub fn capabilities(&self) -> &CapabilityReport {
        &self.capabilities
    }

    pub fn selected_backend(&self) -> Backend {
        self.capabilities.selected_backend
    }

    pub fn fallback(&self) -> Option<&FallbackRecord> {
        self.capabilities.fallback.as_ref()
    }

    fn unavailable_error(&self) -> GpuError {
        match &self.capabilities.fallback {
            Some(fb) => GpuError::unavailable(fb.reason, fb.detail.clone()),
            None => GpuError::NoGpu,
        }
    }

    pub fn kernels(&self) -> GpuResult<&KernelModule> {
        Err(self.unavailable_error())
    }
    pub fn satsolver_extract(
        &self,
        _: &GpuBuffer<u8>,
        _: &GpuBuffer<i32>,
        _: &mut GpuBuffer<u8>,
        _: i32,
        _: i32,
    ) -> GpuResult<()> {
        Err(self.unavailable_error())
    }
    #[allow(clippy::too_many_arguments)]
    pub fn satsolver_aux_reduce_best(
        &self,
        _: &GpuBuffer<u8>,
        _: &mut GpuBuffer<u8>,
        _: &GpuBuffer<i32>,
        _: &mut GpuBuffer<i32>,
        _: &mut GpuBuffer<i32>,
        _: &GpuBuffer<i32>,
        _: i32,
        _: i32,
        _: i32,
        _: i32,
    ) -> GpuResult<()> {
        Err(self.unavailable_error())
    }
    pub fn poisson_encode(
        &self,
        _: &GpuBuffer<f32>,
        _: &mut GpuBuffer<u32>,
        _: u32,
    ) -> GpuResult<()> {
        Err(self.unavailable_error())
    }

    pub fn satsolver_extract_async(
        &self,
        _: &GpuBuffer<u8>,
        _: &GpuBuffer<i32>,
        _: &mut GpuBuffer<u8>,
        _: i32,
        _: i32,
    ) -> GpuResult<()> {
        Err(self.unavailable_error())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn satsolver_aux_reduce_best_async(
        &self,
        _: &GpuBuffer<u8>,
        _: &mut GpuBuffer<u8>,
        _: &GpuBuffer<i32>,
        _: &mut GpuBuffer<i32>,
        _: &mut GpuBuffer<i32>,
        _: &GpuBuffer<i32>,
        _: i32,
        _: i32,
        _: i32,
        _: i32,
    ) -> GpuResult<()> {
        Err(self.unavailable_error())
    }

    pub fn poisson_encode_async(
        &self,
        _: &GpuBuffer<f32>,
        _: &mut GpuBuffer<u32>,
        _: u32,
    ) -> GpuResult<()> {
        Err(self.unavailable_error())
    }

    #[cfg(feature = "saaq")]
    pub fn ensure_temporal_state(&mut self, _: usize) -> GpuResult<()> {
        Err(GpuError::NoGpu)
    }

    #[cfg(feature = "saaq")]
    pub fn project_snapshot_current(&mut self, _: SnapshotChannels, _: usize) -> GpuResult<()> {
        Err(GpuError::NoGpu)
    }

    #[cfg(feature = "saaq")]
    pub fn gif_step_weighted_tick(&mut self, _: usize) -> GpuResult<u32> {
        Err(GpuError::NoGpu)
    }

    #[cfg(feature = "saaq")]
    pub fn reset_temporal_state(&mut self) -> GpuResult<()> {
        Err(GpuError::NoGpu)
    }

    #[cfg(feature = "saaq")]
    pub fn load_synapse_weights(&mut self, _: &[f32]) -> GpuResult<()> {
        Err(GpuError::NoGpu)
    }

    #[cfg(feature = "saaq")]
    pub fn load_synapse_weights_named(&mut self, _: &str, _: &[f32]) -> GpuResult<()> {
        Err(GpuError::NoGpu)
    }

    #[cfg(feature = "saaq")]
    pub fn load_synapse_weights_f16_registered(&mut self, _: &str, _: &[u16]) -> GpuResult<()> {
        Err(GpuError::NoGpu)
    }

    #[cfg(feature = "saaq")]
    pub fn synapse_signature(&self) -> Option<&str> {
        None
    }

    #[cfg(feature = "saaq")]
    pub fn temporal_spikes_to_vec(&self, _: usize) -> GpuResult<Vec<u32>> {
        Err(GpuError::NoGpu)
    }

    #[cfg(feature = "saaq")]
    pub fn temporal_membrane_to_vec(&self, _: usize) -> GpuResult<Vec<f32>> {
        Err(GpuError::NoGpu)
    }

    #[cfg(feature = "saaq")]
    pub fn temporal_adaptation_to_vec(&self, _: usize) -> GpuResult<Vec<f32>> {
        Err(GpuError::NoGpu)
    }

    #[cfg(feature = "saaq")]
    pub fn upload_temporal_input_spikes(&mut self, _: &[f32]) -> GpuResult<()> {
        Err(GpuError::NoGpu)
    }

    #[cfg(feature = "saaq")]
    pub fn saaq_find_best_walker(&mut self, _: usize) -> GpuResult<u32> {
        Err(GpuError::NoGpu)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn ternary_gemv(
        &self,
        _: &GpuBuffer<u32>,
        _: &GpuBuffer<f32>,
        _: &GpuBuffer<f32>,
        _: &mut GpuBuffer<f32>,
        _: i32,
        _: i32,
        _: i32,
        _: bool,
    ) -> GpuResult<()> {
        Err(self.unavailable_error())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn ternary_gemv_async(
        &self,
        _: &GpuBuffer<u32>,
        _: &GpuBuffer<f32>,
        _: &GpuBuffer<f32>,
        _: &mut GpuBuffer<f32>,
        _: i32,
        _: i32,
        _: i32,
        _: bool,
    ) -> GpuResult<()> {
        Err(self.unavailable_error())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn ternary_gemm(
        &self,
        _: &GpuBuffer<u32>,
        _: &GpuBuffer<f32>,
        _: &GpuBuffer<f32>,
        _: &mut GpuBuffer<f32>,
        _: i32,
        _: i32,
        _: i32,
        _: i32,
        _: bool,
    ) -> GpuResult<()> {
        Err(self.unavailable_error())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn ternary_gemm_async(
        &self,
        _: &GpuBuffer<u32>,
        _: &GpuBuffer<f32>,
        _: &GpuBuffer<f32>,
        _: &mut GpuBuffer<f32>,
        _: i32,
        _: i32,
        _: i32,
        _: i32,
        _: bool,
    ) -> GpuResult<()> {
        Err(self.unavailable_error())
    }

    pub fn synchronize(&self) -> GpuResult<()> {
        Err(self.unavailable_error())
    }
}
impl Default for GpuAccelerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "saaq")]
    use crate::gif::SnapshotChannels;

    #[test]
    fn context_api_reports_no_compute_capability() {
        let ctx = GpuContext;
        assert_eq!(ctx.compute_capability(), None);
    }

    // ── GpuError Display ────────────────────────────────────────────────────

    #[test]
    fn error_display_nogpu() {
        let err = GpuError::NoGpu;
        assert_eq!(
            err.to_string(),
            "No GPU available (built without `cuda` feature)"
        );
    }

    #[test]
    fn error_display_init_failed() {
        let err = GpuError::InitFailed("device busy".into());
        assert_eq!(err.to_string(), "GPU init failed: device busy");
    }

    #[test]
    fn error_display_module_load_failed() {
        let err = GpuError::ModuleLoadFailed("bad ptx".into());
        assert_eq!(err.to_string(), "PTX module load failed: bad ptx");
    }

    #[test]
    fn error_display_kernel_not_found() {
        let err = GpuError::KernelNotFound("foo_kernel".into());
        assert_eq!(err.to_string(), "Kernel not found: foo_kernel");
    }

    #[test]
    fn error_display_memory_error() {
        let err = GpuError::MemoryError("OOM".into());
        assert_eq!(err.to_string(), "GPU memory error: OOM");
    }

    #[test]
    fn error_display_launch_failed() {
        let err = GpuError::LaunchFailed("grid too large".into());
        assert_eq!(err.to_string(), "Kernel launch failed: grid too large");
    }

    #[test]
    fn error_display_cuda_error() {
        let err = GpuError::CudaError("illegal memory".into());
        assert_eq!(err.to_string(), "CUDA error: illegal memory");
    }

    #[test]
    fn error_display_unavailable() {
        let err = GpuError::unavailable(
            FallbackReason::CudaFeatureNotBuilt,
            "crate built without the cuda feature",
        );
        assert_eq!(
            err.to_string(),
            "GPU unavailable (cuda_feature_not_built): crate built without the cuda feature"
        );
    }

    #[test]
    fn error_display_invalid_input() {
        let err = GpuError::invalid_input("n_vars must be >= 0, got -1");
        assert_eq!(
            err.to_string(),
            "invalid input: n_vars must be >= 0, got -1"
        );
    }

    #[test]
    fn error_is_std_error() {
        let err: &dyn std::error::Error = &GpuError::NoGpu;
        assert!(!err.to_string().is_empty());
    }

    // ── GpuContext ──────────────────────────────────────────────────────────

    #[test]
    fn context_init_returns_no_gpu() {
        let result = GpuContext::init();
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), GpuError::NoGpu));
    }

    #[test]
    fn context_is_available_returns_false() {
        assert!(!GpuContext::is_available());
    }

    // ── KernelModule ────────────────────────────────────────────────────────

    #[test]
    fn kernel_module_load_returns_no_gpu() {
        let result = KernelModule::load();
        assert!(matches!(result.unwrap_err(), GpuError::NoGpu));
    }

    #[test]
    fn kernel_module_load_satsolver_returns_no_gpu() {
        let result = KernelModule::load_satsolver();
        assert!(matches!(result.unwrap_err(), GpuError::NoGpu));
    }

    // ── GpuBuffer ───────────────────────────────────────────────────────────

    #[test]
    fn buffer_alloc_zero_length() {
        let buf = GpuBuffer::<u8>::alloc(0).unwrap();
        assert_eq!(buf.len(), 0);
    }

    #[test]
    fn buffer_alloc_u8() {
        let buf = GpuBuffer::<u8>::alloc(100).unwrap();
        assert_eq!(buf.len(), 100);
        let data = buf.to_vec().unwrap();
        assert_eq!(data, vec![0u8; 100]);
    }

    #[test]
    fn buffer_alloc_f32() {
        let buf = GpuBuffer::<f32>::alloc(64).unwrap();
        assert_eq!(buf.len(), 64);
        let data = buf.to_vec().unwrap();
        assert!(data.iter().all(|&v| v == 0.0f32));
    }

    #[test]
    fn buffer_from_slice_roundtrip() {
        let input = vec![1i32, 2, 3, 4, 5];
        let buf = GpuBuffer::from_slice(&input).unwrap();
        assert_eq!(buf.len(), 5);
        let output = buf.to_vec().unwrap();
        assert_eq!(input, output);
    }

    #[test]
    fn buffer_from_slice_empty() {
        let buf = GpuBuffer::<u8>::from_slice(&[]).unwrap();
        assert_eq!(buf.len(), 0);
    }

    #[test]
    fn buffer_upload_ok() {
        let mut buf = GpuBuffer::<i32>::alloc(4).unwrap();
        buf.upload(&[10, 20, 30, 40]).unwrap();
        assert_eq!(buf.to_vec().unwrap(), vec![10, 20, 30, 40]);
    }

    #[test]
    fn buffer_upload_length_mismatch() {
        let mut buf = GpuBuffer::<i32>::alloc(4).unwrap();
        let result = buf.upload(&[1, 2]);
        assert!(result.is_err());
        match result.unwrap_err() {
            GpuError::MemoryError(msg) => {
                assert!(msg.contains("length mismatch"));
                assert!(msg.contains("4"));
                assert!(msg.contains("2"));
            }
            other => panic!("expected MemoryError, got: {other}"),
        }
    }

    #[test]
    fn buffer_zero_prefix_preserves_tail() {
        let mut buf = GpuBuffer::<f32>::from_slice(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        buf.zero_prefix(2).unwrap();
        assert_eq!(buf.to_vec().unwrap(), vec![0.0, 0.0, 3.0, 4.0]);
    }

    #[test]
    fn buffer_zero_prefix_rejects_oversize() {
        let mut buf = GpuBuffer::<f32>::from_slice(&[1.0, 2.0]).unwrap();
        let err = buf.zero_prefix(3).unwrap_err();
        match err {
            GpuError::MemoryError(msg) => assert!(msg.contains("zero_prefix")),
            other => panic!("expected MemoryError, got {other}"),
        }
        assert_eq!(buf.to_vec().unwrap(), vec![1.0, 2.0]);
    }

    #[test]
    fn buffer_as_device_ptr_non_null() {
        let buf = GpuBuffer::<u8>::alloc(16).unwrap();
        let ptr = buf.as_device_ptr();
        assert!(!ptr.is_null());
    }

    // ── GpuAccelerator ──────────────────────────────────────────────────────

    #[test]
    fn accelerator_new() {
        let acc = GpuAccelerator::new();
        assert!(!acc.is_ready());
        assert!(!acc.kernels_ready());
        assert_eq!(acc.selected_backend(), Backend::Cpu);
        let fb = acc.fallback().expect("CPU stub records a fallback");
        assert_eq!(fb.reason, FallbackReason::CudaFeatureNotBuilt);
        assert_eq!(fb.selected_backend, Backend::Cpu);
        assert!(!fb.detail.contains("/home/"));
        assert!(!acc.capabilities().cuda_built);
    }

    #[test]
    fn accelerator_default() {
        let acc = GpuAccelerator::default();
        assert!(!acc.is_ready());
        assert!(!acc.kernels_ready());
    }

    #[test]
    fn accelerator_require_gpu_fails_closed() {
        let err = match GpuAccelerator::require_gpu() {
            Err(e) => e,
            Ok(_) => panic!("require_gpu must fail on the CPU stub"),
        };
        assert_eq!(
            err.fallback_reason(),
            Some(FallbackReason::CudaFeatureNotBuilt)
        );
        match err {
            GpuError::Unavailable { reason, detail } => {
                assert_eq!(reason, FallbackReason::CudaFeatureNotBuilt);
                assert!(!detail.contains("/home/"));
            }
            other => panic!("expected Unavailable, got {other}"),
        }
    }

    #[test]
    fn accelerator_with_policy_prefer_gpu_is_cpu() {
        let acc = GpuAccelerator::with_policy(ExecutionPolicy::PreferGpu).unwrap();
        assert_eq!(acc.selected_backend(), Backend::Cpu);
        assert!(!acc.is_ready());
    }

    #[test]
    fn accelerator_kernels_returns_unavailable() {
        let acc = GpuAccelerator::new();
        assert_unavailable_not_built(acc.kernels().unwrap_err());
    }

    #[test]
    fn accelerator_synchronize_returns_unavailable() {
        let acc = GpuAccelerator::new();
        assert_unavailable_not_built(acc.synchronize().unwrap_err());
    }

    fn assert_unavailable_not_built(err: GpuError) {
        match err {
            GpuError::Unavailable { reason, detail } => {
                assert_eq!(reason, FallbackReason::CudaFeatureNotBuilt);
                assert!(!detail.contains("/home/"));
                assert!(!detail.contains("/Users/"));
            }
            other => panic!("expected Unavailable(cuda_feature_not_built), got {other}"),
        }
    }

    #[test]
    fn accelerator_satsolver_extract_returns_unavailable() {
        let acc = GpuAccelerator::new();
        let assignment = GpuBuffer::<u8>::alloc(10).unwrap();
        let best_walker = GpuBuffer::<i32>::alloc(1).unwrap();
        let mut output = GpuBuffer::<u8>::alloc(10).unwrap();
        assert_unavailable_not_built(
            acc.satsolver_extract(&assignment, &best_walker, &mut output, 10, 1)
                .unwrap_err(),
        );
    }

    #[test]
    fn accelerator_poisson_encode_returns_unavailable() {
        let acc = GpuAccelerator::new();
        let stimuli = GpuBuffer::<f32>::alloc(10).unwrap();
        let mut spikes = GpuBuffer::<u32>::alloc(10).unwrap();
        assert_unavailable_not_built(acc.poisson_encode(&stimuli, &mut spikes, 42).unwrap_err());
    }

    #[test]
    fn accelerator_satsolver_extract_async_returns_unavailable() {
        let acc = GpuAccelerator::new();
        let assignment = GpuBuffer::<u8>::alloc(10).unwrap();
        let best_walker = GpuBuffer::<i32>::alloc(1).unwrap();
        let mut output = GpuBuffer::<u8>::alloc(10).unwrap();
        assert_unavailable_not_built(
            acc.satsolver_extract_async(&assignment, &best_walker, &mut output, 10, 1)
                .unwrap_err(),
        );
    }

    #[test]
    fn accelerator_poisson_encode_async_returns_unavailable() {
        let acc = GpuAccelerator::new();
        let stimuli = GpuBuffer::<f32>::alloc(10).unwrap();
        let mut spikes = GpuBuffer::<u32>::alloc(10).unwrap();
        assert_unavailable_not_built(
            acc.poisson_encode_async(&stimuli, &mut spikes, 42)
                .unwrap_err(),
        );
    }

    #[test]
    fn accelerator_ternary_gemv_returns_unavailable() {
        let acc = GpuAccelerator::new();
        let w = GpuBuffer::<u32>::alloc(1).unwrap();
        let s = GpuBuffer::<f32>::alloc(1).unwrap();
        let x = GpuBuffer::<f32>::alloc(1).unwrap();
        let mut y = GpuBuffer::<f32>::alloc(1).unwrap();
        assert_unavailable_not_built(
            acc.ternary_gemv(&w, &s, &x, &mut y, 1, 1, 1, false)
                .unwrap_err(),
        );
    }

    #[test]
    fn accelerator_ternary_gemm_returns_unavailable() {
        let acc = GpuAccelerator::new();
        let w = GpuBuffer::<u32>::alloc(1).unwrap();
        let s = GpuBuffer::<f32>::alloc(1).unwrap();
        let b = GpuBuffer::<f32>::alloc(1).unwrap();
        let mut c = GpuBuffer::<f32>::alloc(1).unwrap();
        assert_unavailable_not_built(
            acc.ternary_gemm(&w, &s, &b, &mut c, 1, 1, 1, 1, false)
                .unwrap_err(),
        );
    }

    #[test]
    fn accelerator_ternary_gemv_async_returns_unavailable() {
        let acc = GpuAccelerator::new();
        let w = GpuBuffer::<u32>::alloc(1).unwrap();
        let s = GpuBuffer::<f32>::alloc(1).unwrap();
        let x = GpuBuffer::<f32>::alloc(1).unwrap();
        let mut y = GpuBuffer::<f32>::alloc(1).unwrap();
        assert_unavailable_not_built(
            acc.ternary_gemv_async(&w, &s, &x, &mut y, 1, 1, 1, false)
                .unwrap_err(),
        );
    }

    #[test]
    fn accelerator_ternary_gemm_async_returns_unavailable() {
        let acc = GpuAccelerator::new();
        let w = GpuBuffer::<u32>::alloc(1).unwrap();
        let s = GpuBuffer::<f32>::alloc(1).unwrap();
        let b = GpuBuffer::<f32>::alloc(1).unwrap();
        let mut c = GpuBuffer::<f32>::alloc(1).unwrap();
        assert_unavailable_not_built(
            acc.ternary_gemm_async(&w, &s, &b, &mut c, 1, 1, 1, 1, false)
                .unwrap_err(),
        );
    }

    #[cfg(feature = "saaq")]
    #[test]
    fn accelerator_temporal_methods_return_no_gpu() {
        let mut acc = GpuAccelerator::new();
        assert!(matches!(
            acc.ensure_temporal_state(16).unwrap_err(),
            GpuError::NoGpu
        ));
        assert!(matches!(
            acc.project_snapshot_current(SnapshotChannels::default(), 16)
                .unwrap_err(),
            GpuError::NoGpu
        ));
        assert!(matches!(
            acc.gif_step_weighted_tick(16).unwrap_err(),
            GpuError::NoGpu
        ));
        assert!(matches!(
            acc.reset_temporal_state().unwrap_err(),
            GpuError::NoGpu
        ));
        assert!(matches!(
            acc.load_synapse_weights_named("x", &[0.0]).unwrap_err(),
            GpuError::NoGpu
        ));
        assert!(matches!(
            acc.load_synapse_weights_f16_registered("x", &[0u16])
                .unwrap_err(),
            GpuError::NoGpu
        ));
        assert!(acc.synapse_signature().is_none());
        assert!(matches!(
            acc.temporal_spikes_to_vec(16).unwrap_err(),
            GpuError::NoGpu
        ));
        assert!(matches!(
            acc.temporal_membrane_to_vec(16).unwrap_err(),
            GpuError::NoGpu
        ));
        assert!(matches!(
            acc.temporal_adaptation_to_vec(16).unwrap_err(),
            GpuError::NoGpu
        ));
        assert!(matches!(
            acc.upload_temporal_input_spikes(&[0.0]).unwrap_err(),
            GpuError::NoGpu
        ));
        assert!(matches!(
            acc.saaq_find_best_walker(16).unwrap_err(),
            GpuError::NoGpu
        ));
    }

    // ── Property-based tests ────────────────────────────────────────────────

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn buffer_from_slice_roundtrip_prop(data in proptest::collection::vec(any::<i32>(), 0..=256)) {
            let buf = GpuBuffer::from_slice(&data).unwrap();
            prop_assert_eq!(buf.len(), data.len());
            prop_assert_eq!(buf.to_vec().unwrap(), data);
        }

        #[test]
        fn buffer_alloc_has_correct_len(n in 0usize..=1024) {
            let buf = GpuBuffer::<u8>::alloc(n).unwrap();
            prop_assert_eq!(buf.len(), n);
        }

        #[test]
        fn buffer_upload_roundtrip(
            // Use finite floats only — NaN != NaN breaks equality assertions.
            data in proptest::collection::vec(
                any::<f32>().prop_filter("finite", |v| v.is_finite()),
                1..=128
            )
        ) {
            let mut buf = GpuBuffer::<f32>::alloc(data.len()).unwrap();
            buf.upload(&data).unwrap();
            prop_assert_eq!(buf.to_vec().unwrap(), data);
        }
    }
}
