// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::gpu::accelerator::GpuAccelerator;
use crate::gpu::error::GpuResult;
use crate::gpu::memory::GpuBuffer;
use crate::routing::ROUTING_BLOCK_SIZE;
use cust::function::Function;
use cust::stream::Stream;

mod entropy;
mod fused;
mod selection;
mod softmax;
mod top_k;

/// Per-block partial buffers shared by the multi-pass routing launches.
pub(super) struct RoutingScratch {
    pub(super) entropy_sum: GpuBuffer<f32>,
    pub(super) entropy_max: GpuBuffer<f32>,
    pub(super) scores: GpuBuffer<f32>,
    pub(super) indices: GpuBuffer<u32>,
}

impl GpuAccelerator {
    fn routing_function<'a>(&'a self, name: &str) -> GpuResult<Function<'a>> {
        self.kernels()?.get_function(name)
    }

    fn routing_launch_parts<'a>(&'a self, name: &str) -> GpuResult<(Function<'a>, &'a Stream)> {
        Ok((self.routing_function(name)?, self.ready_stream()?))
    }

    fn with_routing_scratch<T>(
        &self,
        partial_len: usize,
        f: impl FnOnce(&RoutingScratch, &Stream) -> GpuResult<T>,
    ) -> GpuResult<T> {
        self.ensure_routing_scratch(partial_len)?;
        let scratch = self.routing_scratch.borrow();
        let scratch = scratch.as_ref().ok_or_else(|| self.unavailable_error())?;
        let stream = self.ready_stream()?;
        f(scratch, stream)
    }

    fn finish_launch(&self, launched: GpuResult<()>) -> GpuResult<()> {
        let synchronized = self.synchronize();
        launched?;
        synchronized
    }

    /// Blocks covering `n_nodes`. Dimensions were checked against `i32::MAX`.
    fn routing_grid(n_nodes: usize) -> u32 {
        n_nodes.div_ceil(ROUTING_BLOCK_SIZE) as u32
    }
}
