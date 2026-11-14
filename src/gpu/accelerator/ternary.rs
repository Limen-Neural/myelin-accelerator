// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::GpuAccelerator;
use crate::bitpacking::TERNARY_VALUES_PER_WORD;
use crate::capability::sanitize_diagnostic;
use crate::gpu::error::{GpuError, GpuResult};
use crate::gpu::memory::GpuBuffer;
use cust::launch;
use nvtx::{range_pop, range_push};
impl GpuAccelerator {
    /// Group-scaled ternary GEMV: `y = scale(W) @ x`.
    ///
    /// `packed_w` is row-major `M × ternary_word_count(K)` words.
    /// `scales` is `M × groups_per_row(K, group_size)` f32.
    /// When `skip_zeros` is true, zero trits are skipped on device.
    #[allow(clippy::too_many_arguments)]
    // @codescene(disable:"Excess Number of Function Arguments")
    pub fn ternary_gemv(
        &self,
        packed_w: &GpuBuffer<u32>,
        scales: &GpuBuffer<f32>,
        x: &GpuBuffer<f32>,
        y: &mut GpuBuffer<f32>,
        m: i32,
        k: i32,
        group_size: i32,
        skip_zeros: bool,
    ) -> GpuResult<()> {
        self.ternary_gemv_async(packed_w, scales, x, y, m, k, group_size, skip_zeros)?;
        self.synchronize()
    }

    /// Async variant of [`Self::ternary_gemv`]. All output writes are enqueued
    /// on the accelerator stream. When `m > 0`, the empty-K path waits for
    /// stream completion before returning. Calls with `m == 0` return
    /// immediately; other paths require [`Self::synchronize`] before reading
    /// or dropping any argument buffer.
    #[allow(clippy::too_many_arguments)]
    // @codescene(disable:"Excess Number of Function Arguments")
    pub fn ternary_gemv_async(
        &self,
        packed_w: &GpuBuffer<u32>,
        scales: &GpuBuffer<f32>,
        x: &GpuBuffer<f32>,
        y: &mut GpuBuffer<f32>,
        m: i32,
        k: i32,
        group_size: i32,
        skip_zeros: bool,
    ) -> GpuResult<()> {
        if m < 0 || k < 0 {
            return Err(GpuError::invalid_input(format!(
                "ternary_gemv: m and k must be >= 0, got m={m} k={k}"
            )));
        }
        if group_size <= 0 {
            return Err(GpuError::invalid_input(format!(
                "ternary_gemv: group_size must be > 0, got {group_size}"
            )));
        }

        let m_u = m as usize;
        let k_u = k as usize;
        Self::expect_len("y", y.len(), m_u)?;

        // Match host-ref: empty product is the zero vector (not "leave y untouched").
        if m == 0 {
            if !self.is_ready() {
                return Err(self.unavailable_error());
            }
            return Ok(());
        }
        if k == 0 {
            // Device memset — no host-sized staging buffer; preserve pooled tail.
            self.zero_output_prefix(y, m_u)?;
            return Ok(());
        }

        let group_u = group_size as usize;
        let words_per_row = k_u.div_ceil(TERNARY_VALUES_PER_WORD);
        let n_groups = k_u.div_ceil(group_u);

        Self::expect_len(
            "packed_w",
            packed_w.len(),
            m_u.saturating_mul(words_per_row),
        )?;
        Self::expect_len("scales", scales.len(), m_u.saturating_mul(n_groups))?;
        Self::expect_len("x", x.len(), k_u)?;

        let kernels = self.kernels()?;
        let func = kernels.get_function("ternary_gemv")?;
        let stream = self
            .stream
            .as_ref()
            .ok_or_else(|| self.unavailable_error())?;
        let block = 256u32;
        let grid = Self::ceil_div_u32(m as u32, block);
        let skip = if skip_zeros { 1i32 } else { 0i32 };

        range_push!("ternary_gemv");
        // SAFETY: launching the `ternary_gemv` kernel.
        // - ABI: the CUDA entry (cu/ternary_gemm.cu) is `(const u32* packed_w, const f32* scales,
        //   const f32* x, f32* y, i32 M, i32 K, i32 group_size, i32 skip_zeros)`. Arguments match in
        //   order and type: three read-only device pointers, the `y` write target, then the four
        //   `i32` scalars (`skip` is `0`/`1`).
        // - Bounds: `m,k >= 0` and `group_size > 0` are validated. The `m == 0` and `k == 0` cases
        //   returned earlier, so here `m,k >= 1`. Buffer lengths are checked via `expect_len`:
        //   `y >= m`, `packed_w >= m*ceil(k/TERNARY_VALUES_PER_WORD)`, `scales >= m*ceil(k/group)`,
        //   `x >= k`. These are exactly the strides the kernel indexes (one output row per thread,
        //   `words_per_row` words and `n_groups` scales per row), so all accesses are in bounds.
        // - Launch dims: `grid = ceil_div(m, 256)` covers all `m` output rows; 0 dynamic shared memory.
        // - Lifetime/aliasing: `packed_w`/`scales`/`x` are shared `&`, `y` is unique `&mut` (no host
        //   aliasing); all are caller-owned and outlive the borrow. The synchronous `ternary_gemv`
        //   wrapper synchronizes before returning; the async form requires the caller to synchronize
        //   before freeing/reusing the buffers. Context/module/stream are `Some` for as long as `self`.
        let launch_result = unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                packed_w.as_device_ptr(),
                scales.as_device_ptr(),
                x.as_device_ptr(),
                y.as_device_ptr(),
                m,
                k,
                group_size,
                skip,
            ))
        };
        range_pop!();
        launch_result.map_err(|e| {
            GpuError::LaunchFailed(sanitize_diagnostic(&format!("ternary_gemv launch: {e:?}")))
        })?;

        Ok(())
    }

    /// Group-scaled ternary GEMM: `C = scale(W) @ B`.
    ///
    /// `B` is `K × N` row-major f32; `C` is `M × N` row-major f32.
    #[allow(clippy::too_many_arguments)]
    // @codescene(disable:"Excess Number of Function Arguments")
    pub fn ternary_gemm(
        &self,
        packed_w: &GpuBuffer<u32>,
        scales: &GpuBuffer<f32>,
        b: &GpuBuffer<f32>,
        c: &mut GpuBuffer<f32>,
        m: i32,
        k: i32,
        n: i32,
        group_size: i32,
        skip_zeros: bool,
    ) -> GpuResult<()> {
        self.ternary_gemm_async(packed_w, scales, b, c, m, k, n, group_size, skip_zeros)?;
        self.synchronize()
    }

    /// Async variant of [`Self::ternary_gemm`]. All output writes are enqueued
    /// on the accelerator stream. When `m > 0 && n > 0`, the empty-K path
    /// waits for stream completion before returning. Calls with `m == 0 ||
    /// n == 0` return immediately; other paths require [`Self::synchronize`]
    /// before reading or dropping any argument buffer.
    #[allow(clippy::too_many_arguments)]
    // @codescene(disable:"Excess Number of Function Arguments", disable:"Large Method", disable:"Complex Method", disable:"Complex Conditional")
    pub fn ternary_gemm_async(
        &self,
        packed_w: &GpuBuffer<u32>,
        scales: &GpuBuffer<f32>,
        b: &GpuBuffer<f32>,
        c: &mut GpuBuffer<f32>,
        m: i32,
        k: i32,
        n: i32,
        group_size: i32,
        skip_zeros: bool,
    ) -> GpuResult<()> {
        if m < 0 || k < 0 || n < 0 {
            return Err(GpuError::invalid_input(format!(
                "ternary_gemm: m, k, n must be >= 0, got m={m} k={k} n={n}"
            )));
        }
        if group_size <= 0 {
            return Err(GpuError::invalid_input(format!(
                "ternary_gemm: group_size must be > 0, got {group_size}"
            )));
        }

        let m_u = m as usize;
        let k_u = k as usize;
        let n_u = n as usize;
        Self::expect_len("c", c.len(), m_u.saturating_mul(n_u))?;

        // Match host-ref: empty product zeros C (not "leave C untouched").
        if m == 0 || n == 0 {
            if !self.is_ready() {
                return Err(self.unavailable_error());
            }
            return Ok(());
        }
        if k == 0 {
            // Device memset — no host-sized staging buffer; preserve pooled tail.
            self.zero_output_prefix(c, m_u.saturating_mul(n_u))?;
            return Ok(());
        }

        let group_u = group_size as usize;
        let words_per_row = k_u.div_ceil(TERNARY_VALUES_PER_WORD);
        let n_groups = k_u.div_ceil(group_u);

        Self::expect_len(
            "packed_w",
            packed_w.len(),
            m_u.saturating_mul(words_per_row),
        )?;
        Self::expect_len("scales", scales.len(), m_u.saturating_mul(n_groups))?;
        Self::expect_len("b", b.len(), k_u.saturating_mul(n_u))?;

        let kernels = self.kernels()?;
        let func = kernels.get_function("ternary_gemm")?;
        let stream = self
            .stream
            .as_ref()
            .ok_or_else(|| self.unavailable_error())?;
        // m,n are nonnegative i32 after validation; product always fits u64.
        let total = (m as u64) * (n as u64);
        let block = 256u32;
        // Launch uses 1-D grid of u32 block indices over flattened M*N threads.
        let grid_u64 = total.div_ceil(block as u64);
        if grid_u64 > u32::MAX as u64 {
            return Err(GpuError::invalid_input(format!(
                "ternary_gemm: grid too large for 1-D launch ({grid_u64} blocks, M*N={total})"
            )));
        }
        let grid = grid_u64 as u32;
        let skip = if skip_zeros { 1i32 } else { 0i32 };

        range_push!("ternary_gemm");
        // SAFETY: launching the `ternary_gemm` kernel.
        // - ABI: the CUDA entry (cu/ternary_gemm.cu) is `(const u32* packed_w, const f32* scales,
        //   const f32* b, f32* c, i32 M, i32 K, i32 N, i32 group_size, i32 skip_zeros)`. Arguments
        //   match in order and type: three read-only device pointers, the `c` write target, then the
        //   five `i32` scalars (`skip` is `0`/`1`).
        // - Bounds: `m,k,n >= 0` and `group_size > 0` are validated. The `m == 0 || n == 0` and
        //   `k == 0` cases returned earlier, so here `m,k,n >= 1`. Buffer lengths are checked via
        //   `expect_len`: `c >= m*n`, `packed_w >= m*ceil(k/TERNARY_VALUES_PER_WORD)`,
        //   `scales >= m*ceil(k/group)`, `b >= k*n`. The kernel maps one flattened `M*N` output
        //   element per thread and indexes those same strides, so all accesses are in bounds.
        // - Launch dims: `grid = ceil_div(M*N, 256)` computed in u64 and rejected above if it would
        //   exceed `u32::MAX`, so the 1-D grid covers every output element without overflow;
        //   0 dynamic shared memory.
        // - Lifetime/aliasing: `packed_w`/`scales`/`b` are shared `&`, `c` is unique `&mut` (no host
        //   aliasing); all are caller-owned and outlive the borrow. The synchronous `ternary_gemm`
        //   wrapper synchronizes before returning; the async form requires the caller to synchronize
        //   before freeing/reusing the buffers. Context/module/stream are `Some` for as long as `self`.
        let launch_result = unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                packed_w.as_device_ptr(),
                scales.as_device_ptr(),
                b.as_device_ptr(),
                c.as_device_ptr(),
                m,
                k,
                n,
                group_size,
                skip,
            ))
        };
        range_pop!();
        launch_result.map_err(|e| {
            GpuError::LaunchFailed(sanitize_diagnostic(&format!("ternary_gemm launch: {e:?}")))
        })?;

        Ok(())
    }
}
