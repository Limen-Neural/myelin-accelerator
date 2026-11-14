// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::GpuAccelerator;
use crate::capability::sanitize_diagnostic;
use crate::gpu::error::{GpuError, GpuResult};
use crate::gpu::memory::GpuBuffer;
use cust::launch;
impl GpuAccelerator {
    pub fn poisson_encode(
        &self,
        stimuli: &GpuBuffer<f32>,
        spikes: &mut GpuBuffer<u32>,
        seed: u32,
    ) -> GpuResult<()> {
        self.poisson_encode_async(stimuli, spikes, seed)?;
        self.synchronize()
    }

    /// Async variant of [`Self::poisson_encode`]: enqueues the launch on the
    /// internal stream and returns without waiting.
    ///
    /// An empty stimulus is a defined no-op on a ready accelerator: it
    /// succeeds without launching a kernel (a zero-grid launch would fail
    /// with `InvalidValue`) and leaves any pooled output tail untouched.
    /// Readiness is still required, so a CPU-fallback accelerator reports
    /// `Unavailable` here exactly as it does for non-empty inputs, keeping
    /// the sync/async contract coherent. The caller must call
    /// [`Self::synchronize`] before reading `spikes` on the host or
    /// dropping/reusing either buffer.
    pub fn poisson_encode_async(
        &self,
        stimuli: &GpuBuffer<f32>,
        spikes: &mut GpuBuffer<u32>,
        seed: u32,
    ) -> GpuResult<()> {
        let n = stimuli.len();
        Self::expect_len("spikes", spikes.len(), n)?;

        // The CUDA kernel takes the element count as `int`. Reject `n > i32::MAX`
        // before the cast: `n as i32` would otherwise wrap negative and the
        // kernel's `tid >= n` guard would make every thread skip, silently
        // producing no output. `n as u32` for the grid is safe once `n` fits i32.
        let n_i32 = i32::try_from(n).map_err(|_| {
            GpuError::invalid_input(format!(
                "poisson_encode: element count {n} exceeds i32::MAX kernel limit"
            ))
        })?;

        if n == 0 {
            // Defined empty-input contract (GH #48): the spikes-length check
            // above already ran, so an invalid buffer relationship still fails
            // before this no-op return. Readiness is required so a
            // CPU-fallback accelerator stays fail-closed (matching non-empty
            // inputs and the sync wrapper, which must synchronize). No launch
            // occurs and pooled tails are untouched.
            if !self.is_ready() {
                return Err(self.unavailable_error());
            }
            return Ok(());
        }

        let kernels = self.kernels()?;
        let func = kernels.get_function("poisson_encode")?;
        let stream = self
            .stream
            .as_ref()
            .ok_or_else(|| self.unavailable_error())?;

        let block = 256;
        let grid = Self::ceil_div_u32(n as u32, block);

        // SAFETY: launching the `poisson_encode` kernel.
        // - ABI: the CUDA entry (cu/spiking_network.cu) is
        //   `(const float* stimuli, unsigned int* spikes, int n, unsigned int seed)`. Arguments match:
        //   `stimuli` is a read-only f32 device pointer, `spikes` is the u32 write target, `n_i32`
        //   is the element count, and `seed` is the u32 RNG seed.
        // - Bounds: `spikes.len() == stimuli.len() == n` is enforced above via `expect_len`, and
        //   `n` is checked to fit `i32` via `i32::try_from` (so `n_i32 >= 0` and the grid cast
        //   `n as u32` cannot wrap). The kernel guards `tid >= n`, so every `stimuli[tid]` read and
        //   `spikes[tid]` write is in bounds.
        // - Launch dims: `grid = ceil_div(n, 256)` covers all `n` threads; 0 dynamic shared memory
        //   (the kernel declares none).
        // - Lifetime/aliasing: `stimuli` is shared `&`, `spikes` is unique `&mut` (no host aliasing);
        //   both are caller-owned and outlive the borrow. The synchronous `poisson_encode` wrapper
        //   synchronizes before returning; the async form requires the caller to synchronize before
        //   freeing/reusing the buffers. Context/module/stream are `Some` for as long as `self`.
        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                stimuli.as_device_ptr(),
                spikes.as_device_ptr(),
                n_i32,
                seed,
            ))
            .map_err(|e| {
                GpuError::LaunchFailed(sanitize_diagnostic(&format!(
                    "poisson_encode launch: {e:?}"
                )))
            })?;
        }

        Ok(())
    }
}
