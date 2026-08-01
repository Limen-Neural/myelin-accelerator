// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

// ════════════════════════════════════════════════════════════════════
//  gpu/memory.rs — GPU device buffer wrapper
// ════════════════════════════════════════════════════════════════════

use crate::gpu::error::{GpuError, GpuResult};
use cust::memory::{CopyDestination, DeviceBuffer};

/// Owned device buffer of type `T`.
pub struct GpuBuffer<T: cust::memory::DeviceCopy> {
    inner: DeviceBuffer<T>,
    len: usize,
}

impl<T: cust::memory::DeviceCopy + Default + Clone> GpuBuffer<T> {
    /// Allocate a device buffer of `len` elements (uninitialised).
    pub fn alloc(len: usize) -> GpuResult<Self> {
        let inner = unsafe {
            DeviceBuffer::uninitialized(len)
                .map_err(|e| GpuError::MemoryError(format!("alloc({len}): {e:?}")))?
        };
        Ok(Self { inner, len })
    }

    /// Allocate and upload a host slice to device memory.
    pub fn from_slice(data: &[T]) -> GpuResult<Self> {
        let inner = DeviceBuffer::from_slice(data)
            .map_err(|e| GpuError::MemoryError(format!("from_slice: {e:?}")))?;
        Ok(Self {
            inner,
            len: data.len(),
        })
    }

    /// Download device data into a freshly-allocated `Vec<T>`.
    pub fn to_vec(&self) -> GpuResult<Vec<T>> {
        let mut host = vec![T::default(); self.len];
        self.inner
            .copy_to(&mut host)
            .map_err(|e| GpuError::MemoryError(format!("copy_to: {e:?}")))?;
        Ok(host)
    }

    /// Upload from a host slice (must be same length).
    pub fn upload(&mut self, data: &[T]) -> GpuResult<()> {
        self.inner
            .copy_from(data)
            .map_err(|e| GpuError::MemoryError(format!("upload: {e:?}")))
    }

    /// Upload into the first `data.len()` elements; leaves any tail untouched.
    ///
    /// Used for empty-product zero-fills when the device buffer may be larger
    /// than the logical output (pooled buffers).
    pub fn upload_prefix(&mut self, data: &[T]) -> GpuResult<()> {
        if data.len() > self.len {
            return Err(GpuError::MemoryError(format!(
                "upload_prefix: host len {} > device len {}",
                data.len(),
                self.len
            )));
        }
        if data.is_empty() {
            return Ok(());
        }
        if data.len() == self.len {
            return self.upload(data);
        }
        let mut prefix = self.inner.index(0..data.len());
        prefix
            .copy_from(data)
            .map_err(|e| GpuError::MemoryError(format!("upload_prefix: {e:?}")))
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the buffer contains zero elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Raw device pointer (for kernel launches via `cust`).
    pub fn as_device_ptr(&self) -> cust::memory::DevicePointer<T> {
        self.inner.as_device_ptr()
    }
}

impl GpuBuffer<f32> {
    /// Zero the first `count` elements on device; leaves any tail untouched.
    ///
    /// Uses a device memset (no host-sized staging buffer), so empty-K GEMM/GEMV
    /// zero-fills stay bounded in host memory for large `M*N`.
    pub fn zero_prefix(&mut self, count: usize) -> GpuResult<()> {
        if count > self.len {
            return Err(GpuError::MemoryError(format!(
                "zero_prefix: count {count} > device len {}",
                self.len
            )));
        }
        if count == 0 {
            return Ok(());
        }
        let mut prefix = self.inner.index(0..count);
        prefix
            .set_zero()
            .map_err(|e| GpuError::MemoryError(format!("zero_prefix: {e:?}")))
    }
}
