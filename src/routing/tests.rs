use super::*;

fn params(n_nodes: usize, n_routes: usize, top_k: usize) -> RoutingParams {
    RoutingParams {
        n_nodes,
        n_routes,
        top_k,
        score_kind: RoutingScoreKind::Logits,
        adaptation_scale: 0.5,
    }
}

fn invalid(result: GpuResult<impl std::fmt::Debug>) -> bool {
    matches!(result, Err(GpuError::InvalidInput(_)))
}

mod reference;
mod selection;
mod softmax_entropy;
mod top_k;
mod traffic;
mod validation;
