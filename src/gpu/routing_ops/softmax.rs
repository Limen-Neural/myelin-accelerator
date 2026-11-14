use crate::gpu::accelerator::GpuAccelerator;
use crate::gpu::error::GpuResult;
use crate::routing::{
    ROUTING_BLOCK_SIZE, RoutingShape, RoutingSoftmaxRequest, validate_softmax_buffers,
};
use cust::launch;

impl GpuAccelerator {
    /// Row-normalize a score matrix into `probabilities` (composable path).
    pub fn routing_softmax(&self, request: RoutingSoftmaxRequest<'_>) -> GpuResult<()> {
        self.finish_launch(self.routing_softmax_async(request))
    }

    /// Enqueue [`Self::routing_softmax`]; synchronize before using the buffers.
    pub fn routing_softmax_async(&self, request: RoutingSoftmaxRequest<'_>) -> GpuResult<()> {
        let RoutingSoftmaxRequest {
            scores,
            probabilities,
            n_nodes,
            n_routes,
            score_kind,
        } = request;
        let shape = RoutingShape::new(n_nodes, n_routes)?;
        let (nodes, routes, elements) = validate_softmax_buffers(scores, probabilities, shape)?;
        if elements == 0 {
            return Ok(());
        }

        let (func, stream) = self.routing_launch_parts("routing_softmax")?;
        let grid = Self::routing_grid(n_nodes);
        // SAFETY: the entry is `(const float*, float*, int, int, int)`. Both
        // buffers hold at least n_nodes * n_routes elements, the kernel guards
        // tid >= n_nodes, and the buffers outlive the stream work per the
        // synchronize contract.
        unsafe {
            launch!(func<<<grid, ROUTING_BLOCK_SIZE as u32, 0, stream>>>(
                scores.as_device_ptr(),
                probabilities.as_device_ptr(),
                nodes,
                routes,
                score_kind.as_kernel_arg(),
            ))
        }
        .map_err(|e| Self::launch_error("routing_softmax", e))
    }
}
