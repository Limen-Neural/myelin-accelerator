use crate::gpu::accelerator::GpuAccelerator;
use crate::gpu::error::GpuResult;
use crate::routing::{
    ROUTING_BLOCK_SIZE, RoutingEntropyRequest, RoutingShape, validate_entropy_buffers,
};
use cust::launch;

impl GpuAccelerator {
    /// Sum and maximum of per-node entropy from a probability matrix
    /// (composable path).
    pub fn routing_entropy_reduce(&self, request: RoutingEntropyRequest<'_>) -> GpuResult<()> {
        self.finish_launch(self.routing_entropy_reduce_async(request))
    }

    /// Enqueue [`Self::routing_entropy_reduce`]; synchronize before using the
    /// buffers. An error after the first launch can leave that launch queued.
    pub fn routing_entropy_reduce_async(
        &self,
        request: RoutingEntropyRequest<'_>,
    ) -> GpuResult<()> {
        let RoutingEntropyRequest {
            probabilities,
            entropy_sum,
            entropy_max,
            n_nodes,
            n_routes,
        } = request;
        let shape = RoutingShape::new(n_nodes, n_routes)?;
        let (nodes, routes, _) =
            validate_entropy_buffers(probabilities, entropy_sum, entropy_max, shape)?;
        if n_nodes == 0 {
            self.zero_output_prefix(entropy_sum, 1)?;
            return self.zero_output_prefix(entropy_max, 1);
        }

        let pass1 = self.routing_function("routing_entropy_reduce_pass1")?;
        let pass2 = self.routing_function("latent_reduce_pass2")?;
        let grid = Self::routing_grid(n_nodes);
        self.with_routing_scratch(grid as usize, move |scratch, stream| {
            // SAFETY: pass 1 is `(const float*, float*, float*, int, int)` and
            // writes grid partials; scratch holds at least grid elements. Pass 2 is
            // `(const float*, const float*, float*, float*, int)` and writes one
            // scalar to each output. Both launches share this stream, so pass 2
            // reads completed partials. Scratch lives on `self`.
            unsafe {
                launch!(pass1<<<grid, ROUTING_BLOCK_SIZE as u32, 0, stream>>>(
                    probabilities.as_device_ptr(),
                    scratch.entropy_sum.as_device_ptr(),
                    scratch.entropy_max.as_device_ptr(),
                    nodes,
                    routes,
                ))
                .map_err(|e| Self::launch_error("routing_entropy_reduce_pass1", e))?;
                launch!(pass2<<<1u32, ROUTING_BLOCK_SIZE as u32, 0, stream>>>(
                    scratch.entropy_sum.as_device_ptr(),
                    scratch.entropy_max.as_device_ptr(),
                    entropy_sum.as_device_ptr(),
                    entropy_max.as_device_ptr(),
                    grid as i32,
                ))
            }
            .map_err(|e| Self::launch_error("latent_reduce_pass2", e))
        })
    }
}
