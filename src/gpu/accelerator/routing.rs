// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::GpuAccelerator;
use crate::gpu::error::{GpuError, GpuResult};
use crate::gpu::memory::GpuBuffer;
use crate::gpu::routing_ops::RoutingScratch;
impl GpuAccelerator {
    pub(crate) fn ensure_routing_scratch(&self, partial_len: usize) -> GpuResult<()> {
        self._ctx
            .as_ref()
            .ok_or_else(|| self.unavailable_error())?
            .require_current()?;
        let mut slot = self.routing_scratch.borrow_mut();
        if slot.as_ref().is_none_or(|s| s.scores.len() < partial_len) {
            // Replaced buffers may still be referenced by queued launches.
            self.synchronize()?;
            *slot = Some(RoutingScratch {
                entropy_sum: GpuBuffer::alloc(partial_len)?,
                entropy_max: GpuBuffer::alloc(partial_len)?,
                scores: GpuBuffer::alloc(partial_len)?,
                indices: GpuBuffer::alloc(partial_len)?,
            });
        }
        Ok(())
    }
}
