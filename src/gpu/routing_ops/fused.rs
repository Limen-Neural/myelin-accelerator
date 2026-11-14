use crate::gpu::accelerator::GpuAccelerator;
use crate::gpu::error::GpuResult;
use crate::gpu::memory::GpuBuffer;
use crate::routing::{
    FusedRoutingInputs, FusedRoutingOutputs, ROUTING_BLOCK_SIZE, RoutingParams,
    validate_fused_buffers,
};
use cust::launch;

impl GpuAccelerator {
    /// Run softmax, entropy, top-k, and adjusted-score selection in two
    /// launches without materializing a probability matrix.
    ///
    /// Inputs are `n_nodes × n_routes` scores and `n_nodes` state/adaptation
    /// values. Partial buffers are held by the accelerator. The contract is
    /// documented in [`crate::routing`].
    pub fn fused_routing_select(
        &self,
        scores: &GpuBuffer<f32>,
        state: &GpuBuffer<f32>,
        adaptation: &GpuBuffer<f32>,
        outputs: &mut FusedRoutingOutputs,
        params: &RoutingParams,
    ) -> GpuResult<()> {
        self.finish_launch(
            self.fused_routing_select_async(scores, state, adaptation, outputs, params),
        )
    }

    /// Enqueue [`Self::fused_routing_select`]; synchronize before using the
    /// buffers. An error after the first launch can leave that launch queued.
    pub fn fused_routing_select_async(
        &self,
        scores: &GpuBuffer<f32>,
        state: &GpuBuffer<f32>,
        adaptation: &GpuBuffer<f32>,
        outputs: &mut FusedRoutingOutputs,
        params: &RoutingParams,
    ) -> GpuResult<()> {
        let inputs = FusedRoutingInputs::new(scores, state, adaptation);
        validate_fused_buffers(&inputs, outputs, params)?;
        let pass1 = self.routing_function("fused_routing_pass1")?;
        let pass2 = self.routing_function("fused_routing_pass2")?;
        let grid = Self::routing_grid(params.n_nodes);
        let top_k_probabilities = outputs
            .top_k_probabilities
            .as_ref()
            .map_or_else(cust::memory::DevicePointer::null, |b| b.as_device_ptr());
        self.with_routing_scratch(grid.max(1) as usize, move |scratch, stream| {
            // SAFETY: pass 1 is `(const float*, const float*, const float*, int*,
            // float*, float*, float*, float*, unsigned*, int, int, int, int, float)`.
            // validate_fused_buffers checked every input and output length and that
            // top_k is in 1..=MAX_ROUTING_TOP_K; the kernel guards tid >= n_nodes,
            // only writes probabilities through a non-null pointer, and writes grid
            // partials into scratch. Pass 2 reads exactly grid partials, so
            // grid == 0 writes the empty result without reading scratch. Both
            // launches share this stream.
            unsafe {
                if grid > 0 {
                    launch!(pass1<<<grid, ROUTING_BLOCK_SIZE as u32, 0, stream>>>(
                        scores.as_device_ptr(),
                        state.as_device_ptr(),
                        adaptation.as_device_ptr(),
                        outputs.top_k_indices.as_device_ptr(),
                        top_k_probabilities,
                        scratch.entropy_sum.as_device_ptr(),
                        scratch.entropy_max.as_device_ptr(),
                        scratch.scores.as_device_ptr(),
                        scratch.indices.as_device_ptr(),
                        params.n_nodes as i32,
                        params.n_routes as i32,
                        params.top_k as i32,
                        params.score_kind.as_kernel_arg(),
                        params.adaptation_scale,
                    ))
                    .map_err(|e| Self::launch_error("fused_routing_pass1", e))?;
                }
                launch!(pass2<<<1u32, ROUTING_BLOCK_SIZE as u32, 0, stream>>>(
                    scratch.entropy_sum.as_device_ptr(),
                    scratch.entropy_max.as_device_ptr(),
                    scratch.scores.as_device_ptr(),
                    scratch.indices.as_device_ptr(),
                    outputs.entropy_sum.as_device_ptr(),
                    outputs.entropy_max.as_device_ptr(),
                    outputs.selected_index.as_device_ptr(),
                    outputs.selected_score.as_device_ptr(),
                    grid as i32,
                ))
            }
            .map_err(|e| Self::launch_error("fused_routing_pass2", e))
        })
    }
}
