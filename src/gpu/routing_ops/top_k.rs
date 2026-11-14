use crate::gpu::accelerator::GpuAccelerator;
use crate::gpu::error::GpuResult;
use crate::routing::{
    ROUTING_BLOCK_SIZE, RoutingShape, RoutingTopKRequest, validate_top_k_buffers,
};
use cust::launch;

impl GpuAccelerator {
    /// Per-row top-k route indices (composable path), padded with `-1`.
    pub fn routing_top_k(&self, request: RoutingTopKRequest<'_>) -> GpuResult<()> {
        self.finish_launch(self.routing_top_k_async(request))
    }

    /// Enqueue [`Self::routing_top_k`]; synchronize before using the buffers.
    pub fn routing_top_k_async(&self, request: RoutingTopKRequest<'_>) -> GpuResult<()> {
        let RoutingTopKRequest {
            scores,
            top_k_indices,
            n_nodes,
            n_routes,
            top_k,
            score_kind,
        } = request;
        let shape = RoutingShape::new(n_nodes, n_routes)?;
        let (nodes, routes, _) = validate_top_k_buffers(scores, top_k_indices, shape, top_k)?;
        if n_nodes == 0 {
            return Ok(());
        }

        let (func, stream) = self.routing_launch_parts("routing_top_k")?;
        let grid = Self::routing_grid(n_nodes);
        // SAFETY: the entry is `(const float*, int*, int, int, int, int)`.
        // Buffer lengths were checked above, top_k is at most
        // MAX_ROUTING_TOP_K, and the kernel guards tid >= n_nodes. The buffers
        // outlive the stream work per the synchronize contract.
        unsafe {
            launch!(func<<<grid, ROUTING_BLOCK_SIZE as u32, 0, stream>>>(
                scores.as_device_ptr(),
                top_k_indices.as_device_ptr(),
                nodes,
                routes,
                top_k as i32,
                score_kind.as_kernel_arg(),
            ))
        }
        .map_err(|e| Self::launch_error("routing_top_k", e))
    }
}
