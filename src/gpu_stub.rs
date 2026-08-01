// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt;

pub type GpuResult<T> = Result<T, GpuError>;

#[derive(Debug)]
pub enum GpuError {
    NoGpu,
    InitFailed(String),
    ModuleLoadFailed(String),
    KernelNotFound(String),
    MemoryError(String),
    LaunchFailed(String),
    CudaError(String),
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuError::NoGpu => write!(f, "No GPU available (built without `cuda` feature)"),
            GpuError::InitFailed(s) => write!(f, "GPU init failed: {s}"),
            GpuError::ModuleLoadFailed(s) => write!(f, "PTX module load failed: {s}"),
            GpuError::KernelNotFound(s) => write!(f, "Kernel not found: {s}"),
            GpuError::MemoryError(s) => write!(f, "GPU memory error: {s}"),
            GpuError::LaunchFailed(s) => write!(f, "Kernel launch failed: {s}"),
            GpuError::CudaError(s) => write!(f, "CUDA error: {s}"),
        }
    }
}

impl std::error::Error for GpuError {}

#[derive(Debug)]
pub struct GpuContext;
impl GpuContext {
    pub fn init() -> GpuResult<Self> {
        Err(GpuError::NoGpu)
    }
    pub fn is_available() -> bool {
        false
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
    /// Upload into the first `data.len()` elements; leaves any tail untouched.
    pub fn upload_prefix(&mut self, data: &[T]) -> GpuResult<()> {
        if data.len() > self.data.len() {
            return Err(GpuError::MemoryError(format!(
                "upload_prefix: host len {} > buffer len {}",
                data.len(),
                self.data.len()
            )));
        }
        self.data[..data.len()].clone_from_slice(data);
        Ok(())
    }

    /// Zero the first `count` elements; leaves any tail untouched (host stub).
    pub fn zero_prefix(&mut self, count: usize) -> GpuResult<()>
    where
        T: Default + Copy,
    {
        if count > self.data.len() {
            return Err(GpuError::MemoryError(format!(
                "zero_prefix: count {count} > buffer len {}",
                self.data.len()
            )));
        }
        for slot in &mut self.data[..count] {
            *slot = T::default();
        }
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

pub struct GpuAccelerator;
impl GpuAccelerator {
    pub fn new() -> Self {
        Self
    }
    pub fn is_ready(&self) -> bool {
        false
    }
    pub fn kernels(&self) -> GpuResult<&KernelModule> {
        Err(GpuError::NoGpu)
    }
    pub fn satsolver_extract(
        &self,
        _: &GpuBuffer<u8>,
        _: &GpuBuffer<i32>,
        _: &mut GpuBuffer<u8>,
        _: i32,
        _: i32,
    ) -> GpuResult<()> {
        Err(GpuError::NoGpu)
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
        Err(GpuError::NoGpu)
    }
    pub fn poisson_encode(
        &self,
        _: &GpuBuffer<f32>,
        _: &mut GpuBuffer<u32>,
        _: u32,
    ) -> GpuResult<()> {
        Err(GpuError::NoGpu)
    }

    pub fn satsolver_extract_async(
        &self,
        _: &GpuBuffer<u8>,
        _: &GpuBuffer<i32>,
        _: &mut GpuBuffer<u8>,
        _: i32,
        _: i32,
    ) -> GpuResult<()> {
        Err(GpuError::NoGpu)
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
        Err(GpuError::NoGpu)
    }

    pub fn poisson_encode_async(
        &self,
        _: &GpuBuffer<f32>,
        _: &mut GpuBuffer<u32>,
        _: u32,
    ) -> GpuResult<()> {
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
        Err(GpuError::NoGpu)
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
        Err(GpuError::NoGpu)
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
        Err(GpuError::NoGpu)
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
        Err(GpuError::NoGpu)
    }

    pub fn synchronize(&self) -> GpuResult<()> {
        Err(GpuError::NoGpu)
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
    fn buffer_upload_prefix_preserves_tail() {
        let mut buf = GpuBuffer::<i32>::from_slice(&[1, 2, 3, 4]).unwrap();
        buf.upload_prefix(&[9, 8]).unwrap();
        assert_eq!(buf.to_vec().unwrap(), vec![9, 8, 3, 4]);
    }

    #[test]
    fn buffer_upload_prefix_rejects_oversize() {
        let mut buf = GpuBuffer::<i32>::from_slice(&[1, 2, 3, 4]).unwrap();
        let err = buf.upload_prefix(&[9, 8, 7, 6, 5]).unwrap_err();
        match err {
            GpuError::MemoryError(msg) => assert!(msg.contains("upload_prefix")),
            other => panic!("expected MemoryError, got {other}"),
        }
        assert_eq!(buf.to_vec().unwrap(), vec![1, 2, 3, 4]);
    }

    #[test]
    fn buffer_zero_prefix_preserves_tail() {
        let mut buf = GpuBuffer::<f32>::from_slice(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        buf.zero_prefix(2).unwrap();
        assert_eq!(buf.to_vec().unwrap(), vec![0.0, 0.0, 3.0, 4.0]);
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
    }

    #[test]
    fn accelerator_default() {
        let acc = GpuAccelerator;
        assert!(!acc.is_ready());
    }

    #[test]
    fn accelerator_kernels_returns_no_gpu() {
        let acc = GpuAccelerator::new();
        assert!(matches!(acc.kernels().unwrap_err(), GpuError::NoGpu));
    }

    #[test]
    fn accelerator_synchronize_returns_no_gpu() {
        let acc = GpuAccelerator::new();
        assert!(matches!(acc.synchronize().unwrap_err(), GpuError::NoGpu));
    }

    #[test]
    fn accelerator_satsolver_extract_returns_no_gpu() {
        let acc = GpuAccelerator::new();
        let assignment = GpuBuffer::<u8>::alloc(10).unwrap();
        let best_walker = GpuBuffer::<i32>::alloc(1).unwrap();
        let mut output = GpuBuffer::<u8>::alloc(10).unwrap();
        assert!(matches!(
            acc.satsolver_extract(&assignment, &best_walker, &mut output, 10, 1)
                .unwrap_err(),
            GpuError::NoGpu
        ));
    }

    #[test]
    fn accelerator_poisson_encode_returns_no_gpu() {
        let acc = GpuAccelerator::new();
        let stimuli = GpuBuffer::<f32>::alloc(10).unwrap();
        let mut spikes = GpuBuffer::<u32>::alloc(10).unwrap();
        assert!(matches!(
            acc.poisson_encode(&stimuli, &mut spikes, 42).unwrap_err(),
            GpuError::NoGpu
        ));
    }

    #[test]
    fn accelerator_satsolver_extract_async_returns_no_gpu() {
        let acc = GpuAccelerator::new();
        let assignment = GpuBuffer::<u8>::alloc(10).unwrap();
        let best_walker = GpuBuffer::<i32>::alloc(1).unwrap();
        let mut output = GpuBuffer::<u8>::alloc(10).unwrap();
        assert!(matches!(
            acc.satsolver_extract_async(&assignment, &best_walker, &mut output, 10, 1)
                .unwrap_err(),
            GpuError::NoGpu
        ));
    }

    #[test]
    fn accelerator_poisson_encode_async_returns_no_gpu() {
        let acc = GpuAccelerator::new();
        let stimuli = GpuBuffer::<f32>::alloc(10).unwrap();
        let mut spikes = GpuBuffer::<u32>::alloc(10).unwrap();
        assert!(matches!(
            acc.poisson_encode_async(&stimuli, &mut spikes, 42)
                .unwrap_err(),
            GpuError::NoGpu
        ));
    }

    #[test]
    fn accelerator_ternary_gemv_returns_no_gpu() {
        let acc = GpuAccelerator::new();
        let w = GpuBuffer::<u32>::alloc(1).unwrap();
        let s = GpuBuffer::<f32>::alloc(1).unwrap();
        let x = GpuBuffer::<f32>::alloc(1).unwrap();
        let mut y = GpuBuffer::<f32>::alloc(1).unwrap();
        assert!(matches!(
            acc.ternary_gemv(&w, &s, &x, &mut y, 1, 1, 1, false)
                .unwrap_err(),
            GpuError::NoGpu
        ));
    }

    #[test]
    fn accelerator_ternary_gemm_returns_no_gpu() {
        let acc = GpuAccelerator::new();
        let w = GpuBuffer::<u32>::alloc(1).unwrap();
        let s = GpuBuffer::<f32>::alloc(1).unwrap();
        let b = GpuBuffer::<f32>::alloc(1).unwrap();
        let mut c = GpuBuffer::<f32>::alloc(1).unwrap();
        assert!(matches!(
            acc.ternary_gemm(&w, &s, &b, &mut c, 1, 1, 1, 1, false)
                .unwrap_err(),
            GpuError::NoGpu
        ));
    }

    #[test]
    fn accelerator_ternary_gemv_async_returns_no_gpu() {
        let acc = GpuAccelerator::new();
        let w = GpuBuffer::<u32>::alloc(1).unwrap();
        let s = GpuBuffer::<f32>::alloc(1).unwrap();
        let x = GpuBuffer::<f32>::alloc(1).unwrap();
        let mut y = GpuBuffer::<f32>::alloc(1).unwrap();
        assert!(matches!(
            acc.ternary_gemv_async(&w, &s, &x, &mut y, 1, 1, 1, false)
                .unwrap_err(),
            GpuError::NoGpu
        ));
    }

    #[test]
    fn accelerator_ternary_gemm_async_returns_no_gpu() {
        let acc = GpuAccelerator::new();
        let w = GpuBuffer::<u32>::alloc(1).unwrap();
        let s = GpuBuffer::<f32>::alloc(1).unwrap();
        let b = GpuBuffer::<f32>::alloc(1).unwrap();
        let mut c = GpuBuffer::<f32>::alloc(1).unwrap();
        assert!(matches!(
            acc.ternary_gemm_async(&w, &s, &b, &mut c, 1, 1, 1, 1, false)
                .unwrap_err(),
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
