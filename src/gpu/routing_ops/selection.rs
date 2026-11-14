use crate::gpu::accelerator::GpuAccelerator;
use crate::gpu::error::GpuResult;
use crate::routing::{AdjustedSelectionRequest, ROUTING_BLOCK_SIZE, SelectionInputs};
use cust::launch;

impl GpuAccelerator {
    /// Select the node with the highest `state - scale * adaptation`, using a
    /// per-block pass and a reduction pass.
    ///
    /// Writes [`crate::routing::NO_SELECTION`] and `-inf` when no node is
    /// eligible. This is the default selection path; see `docs/ROUTING.md`.
    pub fn adjusted_argmax(&self, request: AdjustedSelectionRequest<'_>) -> GpuResult<()> {
        self.finish_launch(self.adjusted_argmax_async(request))
    }

    /// Enqueue [`Self::adjusted_argmax`]; synchronize before using the buffers.
    /// An error after the first launch can leave that launch queued.
    pub fn adjusted_argmax_async(&self, request: AdjustedSelectionRequest<'_>) -> GpuResult<()> {
        let AdjustedSelectionRequest {
            state,
            adaptation,
            selected_index,
            selected_score,
            scale,
        } = request;
        let n = {
            let inputs = SelectionInputs::new(state, adaptation, selected_index, selected_score);
            crate::routing::validate_selection_buffers(&inputs, scale)?
        };
        let pass1 = self.routing_function("adjusted_argmax_pass1")?;
        let pass2 = self.routing_function("indexed_argmax_reduce_pass2")?;
        let grid = Self::routing_grid(n as usize);
        self.with_routing_scratch(grid.max(1) as usize, move |scratch, stream| {
            // SAFETY: pass 1 is `(const float*, const float*, float*, unsigned*,
            // int, float)`; state and adaptation hold n elements, the kernel guards
            // tid >= n, and scratch holds at least grid partials. Pass 2 is
            // `(const float*, const unsigned*, unsigned*, float*, int)` and reads
            // exactly grid partials, so grid == 0 writes the no-selection result
            // without reading scratch. Both launches share this stream.
            unsafe {
                if grid > 0 {
                    launch!(pass1<<<grid, ROUTING_BLOCK_SIZE as u32, 0, stream>>>(
                        state.as_device_ptr(),
                        adaptation.as_device_ptr(),
                        scratch.scores.as_device_ptr(),
                        scratch.indices.as_device_ptr(),
                        n,
                        scale,
                    ))
                    .map_err(|e| Self::launch_error("adjusted_argmax_pass1", e))?;
                }
                launch!(pass2<<<1u32, ROUTING_BLOCK_SIZE as u32, 0, stream>>>(
                    scratch.scores.as_device_ptr(),
                    scratch.indices.as_device_ptr(),
                    selected_index.as_device_ptr(),
                    selected_score.as_device_ptr(),
                    grid as i32,
                ))
            }
            .map_err(|e| Self::launch_error("indexed_argmax_reduce_pass2", e))
        })
    }

    /// One-launch variant of [`Self::adjusted_argmax`] with the same results
    /// and no partial buffers. It occupies a single block at any `n`; prefer it
    /// only at the small sizes identified in `docs/ROUTING.md`.
    pub fn adjusted_argmax_single_block(
        &self,
        request: AdjustedSelectionRequest<'_>,
    ) -> GpuResult<()> {
        self.finish_launch(self.adjusted_argmax_single_block_async(request))
    }

    /// Enqueue [`Self::adjusted_argmax_single_block`]; synchronize before using
    /// the buffers.
    pub fn adjusted_argmax_single_block_async(
        &self,
        request: AdjustedSelectionRequest<'_>,
    ) -> GpuResult<()> {
        let AdjustedSelectionRequest {
            state,
            adaptation,
            selected_index,
            selected_score,
            scale,
        } = request;
        let inputs = SelectionInputs::new(state, adaptation, selected_index, selected_score);
        let n = crate::routing::validate_selection_buffers(&inputs, scale)?;
        let (func, stream) = self.routing_launch_parts("adjusted_argmax_single_block")?;
        // SAFETY: the entry is `(const float*, const float*, unsigned*, float*,
        // int, float)`. Threads stride below n, which both inputs hold, and only
        // thread 0 writes one scalar to each output, which were length-checked.
        unsafe {
            launch!(func<<<1u32, ROUTING_BLOCK_SIZE as u32, 0, stream>>>(
                state.as_device_ptr(),
                adaptation.as_device_ptr(),
                selected_index.as_device_ptr(),
                selected_score.as_device_ptr(),
                n,
                scale,
            ))
        }
        .map_err(|e| Self::launch_error("adjusted_argmax_single_block", e))
    }
}
