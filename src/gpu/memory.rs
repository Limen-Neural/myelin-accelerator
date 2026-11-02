// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

// ════════════════════════════════════════════════════════════════════
//  gpu/memory.rs — GPU device buffer wrapper
// ════════════════════════════════════════════════════════════════════

use crate::gpu::error::{GpuError, GpuResult};
use cust::memory::{CopyDestination, DeviceBuffer};
use cust::stream::Stream;

/// Owned device buffer of type `T`.
pub struct GpuBuffer<T: cust::memory::DeviceCopy> {
    inner: DeviceBuffer<T>,
    len: usize,
}

impl<T: cust::memory::DeviceCopy + Default + Clone> GpuBuffer<T> {
    /// Allocate a device buffer of `len` elements, initialised to `T::default()`.
    ///
    /// The buffer is readable immediately: [`Self::to_vec`] after `alloc`
    /// returns `len` copies of `T::default()` without requiring a prior
    /// [`Self::upload`], kernel launch, or memset. This matches the CPU-stub
    /// contract (`vec![T::default(); len]`).
    ///
    /// `alloc(0)` succeeds and yields an empty, readable buffer.
    ///
    /// # Initialisation cost
    ///
    /// Initialisation is performed by staging a `Vec<T>` of `len` default
    /// elements on the host and uploading it via [`DeviceBuffer::from_slice`]
    /// (host-to-device copy). This costs `len * size_of::<T>()` bytes of
    /// transient host memory plus one H2D transfer per `alloc`. The staging
    /// `Vec` is reserved fallibly so that an oversized request surfaces as
    /// [`GpuError::MemoryError`] instead of aborting the process, and the
    /// element-count arithmetic is checked against `isize::MAX` bytes (the
    /// allocation ceiling) before reserving.
    pub fn alloc(len: usize) -> GpuResult<Self> {
        // Check size arithmetic before any staging allocation, using the same
        // helper as the CPU stub so the overflow contract cannot drift
        // (GH #48). `DeviceCopy` implies `Sized`, so `size_of::<T>()` is the
        // exact per-element byte cost; the Rust allocator rejects any single
        // allocation exceeding `isize::MAX` bytes, so reject earlier with a
        // categorised error.
        crate::error::checked_alloc_bytes::<T>(len)?;

        // Fallible host reservation: an OOM here becomes a categorised error
        // rather than an allocation abort.
        let mut host: Vec<T> = Vec::new();
        host.try_reserve_exact(len).map_err(|e| {
            GpuError::MemoryError(format!("alloc({len}): host staging reserve failed: {e:?}"))
        })?;
        host.resize(len, T::default());

        Self::from_slice(&host)
    }

    /// Allocate a device buffer and upload `data` into it.
    ///
    /// The resulting buffer has `data.len()` elements holding exactly the
    /// uploaded values. `from_slice(&[])` succeeds and yields an empty buffer.
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

    /// Upload from a host slice, replacing the buffer's contents.
    ///
    /// `data.len()` must equal [`Self::len`]. On a length mismatch this returns
    /// [`GpuError::MemoryError`] *before* touching device memory, leaving the
    /// existing contents unchanged; the check is performed here rather than
    /// relying on the underlying `cust` copy, which panics on a length mismatch.
    /// The error payload matches the CPU stub exactly.
    ///
    /// On equal lengths the buffer contents are completely replaced. Uploading
    /// an empty slice into an empty buffer succeeds and is a no-op. Genuine
    /// device copy failures are surfaced as [`GpuError::MemoryError`].
    pub fn upload(&mut self, data: &[T]) -> GpuResult<()> {
        if data.len() != self.len {
            return Err(GpuError::MemoryError(format!(
                "upload: length mismatch, buffer has {} elements but input has {}",
                self.len,
                data.len()
            )));
        }
        self.inner
            .copy_from(data)
            .map_err(|e| GpuError::MemoryError(format!("upload: {e:?}")))
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
    /// This is a synchronous default-stream operation. Do not mix it with
    /// pending `GpuAccelerator::*_async` launches on the same buffer; synchronize
    /// the accelerator first. For stream-ordered work, use [`Self::zero_prefix_on`].
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

    /// Enqueue a prefix zero-fill on `stream`, leaving any tail untouched.
    ///
    /// # Safety
    ///
    /// The caller must keep this buffer and `stream` alive until the memset
    /// completes. Later work on the same stream is ordered after the memset;
    /// host access and work on other streams require synchronization or an
    /// explicit cross-stream dependency first.
    pub unsafe fn zero_prefix_on(&mut self, count: usize, stream: &Stream) -> GpuResult<()> {
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
        // SAFETY: the caller guarantees that the buffer and stream remain live
        // and that no unordered access to the prefix occurs before completion.
        unsafe { prefix.set_zero_async(stream) }
            .map_err(|e| GpuError::MemoryError(format!("zero_prefix_on: {e:?}")))
    }
}
