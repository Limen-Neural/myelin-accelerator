// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::GpuAccelerator;
use crate::capability::sanitize_diagnostic;
use crate::gpu::error::{GpuError, GpuResult};
use crate::gpu::memory::GpuBuffer;
use cust::launch;
struct StdpTraceLaunch<'a> {
    pre_spikes: &'a GpuBuffer<f32>,
    post_spikes: &'a GpuBuffer<f32>,
    pre_traces: &'a mut GpuBuffer<f32>,
    post_traces: &'a mut GpuBuffer<f32>,
    n_post: i32,
    n_pre: i32,
    dt_ms: f32,
}

impl GpuAccelerator {
    /// Update STDP weights and eligibility traces in one ordered operation.
    ///
    /// The weight kernel reads old traces and computes the current-step trace
    /// values locally. The trace kernel then writes each trace once. Both
    /// launches use this accelerator's stream, so they cannot race each other.
    /// If either launch fails, queued work is synchronized before returning;
    /// an error can still leave weights or traces partially updated.
    #[allow(clippy::too_many_arguments)]
    // @codescene(disable:"Excess Number of Function Arguments")
    // qlty-ignore: qlty:function-parameters
    pub fn stdp_update(
        &self,
        weights: &mut GpuBuffer<f32>,
        pre_spikes: &GpuBuffer<f32>,
        post_spikes: &GpuBuffer<f32>,
        pre_traces: &mut GpuBuffer<f32>,
        post_traces: &mut GpuBuffer<f32>,
        n_post: i32,
        n_pre: i32,
        dt_ms: f32,
    ) -> GpuResult<()> {
        let update_result = self.stdp_update_async(
            weights,
            pre_spikes,
            post_spikes,
            pre_traces,
            post_traces,
            n_post,
            n_pre,
            dt_ms,
        );
        let sync_result = self.synchronize();
        update_result?;
        sync_result
    }

    /// Enqueue both STDP kernels on the accelerator stream without waiting.
    ///
    /// Call [`Self::synchronize`] before reading, reusing, or dropping any
    /// argument buffer. Concurrent launches on another stream must not touch
    /// these weights, spikes, or traces until this operation completes.
    /// A launch error can leave the preceding weight launch queued; synchronize
    /// before reusing the buffers, and do not assume an atomic update.
    #[allow(clippy::too_many_arguments)]
    // @codescene(disable:"Excess Number of Function Arguments")
    // qlty-ignore: qlty:function-parameters
    pub fn stdp_update_async(
        &self,
        weights: &mut GpuBuffer<f32>,
        pre_spikes: &GpuBuffer<f32>,
        post_spikes: &GpuBuffer<f32>,
        pre_traces: &mut GpuBuffer<f32>,
        post_traces: &mut GpuBuffer<f32>,
        n_post: i32,
        n_pre: i32,
        dt_ms: f32,
    ) -> GpuResult<()> {
        Self::validate_stdp_buffers(
            weights,
            pre_spikes,
            post_spikes,
            pre_traces,
            post_traces,
            n_post,
            n_pre,
            dt_ms,
        )?;
        if n_post == 0 && n_pre == 0 {
            return Ok(());
        }

        let kernels = self.kernels()?;
        let stream = self
            .stream
            .as_ref()
            .ok_or_else(|| self.unavailable_error())?;

        if n_post > 0 && n_pre > 0 {
            let update_weights = kernels.get_function("stdp_update_weights")?;
            let blocks = ((n_post as u64) * (n_pre as u64)).div_ceil(256);
            if blocks > i32::MAX as u64 {
                return Err(GpuError::invalid_input(
                    "stdp_update: weight grid exceeds CUDA grid.x limit".to_string(),
                ));
            }
            let grid = blocks as u32;
            // SAFETY: every matrix/trace/spike buffer was length-checked; the
            // kernel bounds-checks both global indices, and the stream and
            // buffers remain alive until the caller synchronizes.
            unsafe {
                launch!(update_weights<<<grid, 256u32, 0u32, stream>>>(
                    weights.as_device_ptr(),
                    pre_spikes.as_device_ptr(),
                    post_spikes.as_device_ptr(),
                    pre_traces.as_device_ptr(),
                    post_traces.as_device_ptr(),
                    n_post,
                    n_pre,
                    dt_ms
                ))
                .map_err(|e| {
                    GpuError::LaunchFailed(sanitize_diagnostic(&format!(
                        "stdp_update_weights launch: {e:?}"
                    )))
                })?;
            }
        }

        self.launch_stdp_traces(StdpTraceLaunch {
            pre_spikes,
            post_spikes,
            pre_traces,
            post_traces,
            n_post,
            n_pre,
            dt_ms,
        })
    }

    fn launch_stdp_traces(&self, args: StdpTraceLaunch<'_>) -> GpuResult<()> {
        let update_traces = self.kernels()?.get_function("stdp_update_traces")?;
        let stream = self
            .stream
            .as_ref()
            .ok_or_else(|| self.unavailable_error())?;
        let grid = Self::ceil_div_u32(args.n_post.max(args.n_pre) as u32, 256);
        // SAFETY: each global thread owns at most one index of each trace
        // buffer. The accelerator stream orders this after the weight update.
        unsafe {
            launch!(update_traces<<<grid, 256u32, 0u32, stream>>>(
                args.pre_spikes.as_device_ptr(),
                args.post_spikes.as_device_ptr(),
                args.pre_traces.as_device_ptr(),
                args.post_traces.as_device_ptr(),
                args.n_post,
                args.n_pre,
                args.dt_ms
            ))
            .map_err(|e| {
                GpuError::LaunchFailed(sanitize_diagnostic(&format!(
                    "stdp_update_traces launch: {e:?}"
                )))
            })?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    // @codescene(disable:"Excess Number of Function Arguments")
    // qlty-ignore: qlty:function-parameters
    fn validate_stdp_buffers(
        weights: &GpuBuffer<f32>,
        pre_spikes: &GpuBuffer<f32>,
        post_spikes: &GpuBuffer<f32>,
        pre_traces: &GpuBuffer<f32>,
        post_traces: &GpuBuffer<f32>,
        n_post: i32,
        n_pre: i32,
        dt_ms: f32,
    ) -> GpuResult<()> {
        let invalid_shape = n_post < 0 || n_pre < 0;
        let invalid_time = !dt_ms.is_finite() || dt_ms < 0.0;
        if invalid_shape || invalid_time {
            return Err(GpuError::invalid_input(
                "stdp_update: n_post/n_pre and finite dt_ms must be nonnegative".to_string(),
            ));
        }
        let post = n_post as usize;
        let pre = n_pre as usize;
        Self::expect_len("weights", weights.len(), post.saturating_mul(pre))?;
        Self::expect_len("pre_spikes", pre_spikes.len(), pre)?;
        Self::expect_len("post_spikes", post_spikes.len(), post)?;
        Self::expect_len("pre_traces", pre_traces.len(), pre)?;
        Self::expect_len("post_traces", post_traces.len(), post)?;
        Ok(())
    }
}
