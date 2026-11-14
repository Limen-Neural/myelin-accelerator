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

#[test]
fn softmax_handles_special_logits() {
    use RoutingScoreKind::Logits;
    assert_eq!(
        softmax_row(&[1.0, f32::INFINITY, 2.0], Logits),
        vec![0.0, 1.0, 0.0]
    );
    assert_eq!(
        softmax_row(&[1.0, f32::INFINITY, f32::INFINITY], Logits),
        vec![0.0, 0.5, 0.5]
    );
    assert_eq!(softmax_row(&[f32::NAN, 0.0], Logits), vec![0.0, 1.0]);
    assert_eq!(
        softmax_row(&[f32::NEG_INFINITY, f32::NAN], Logits),
        vec![0.5, 0.5]
    );
}

#[test]
fn entropy_has_closed_form_special_rows() {
    use RoutingScoreKind::Logits;
    assert_eq!(
        entropy_row(&[f32::INFINITY, f32::INFINITY, 1.0], Logits),
        1.0
    );
    assert_eq!(entropy_row(&[f32::INFINITY, 1.0], Logits), 0.0);
    assert_eq!(entropy_row(&[f32::NAN; 4], Logits), 2.0);
    assert_eq!(entropy_row(&[], Logits), 0.0);
    assert!((entropy_row(&[3.0; 4], Logits) - 2.0).abs() < 1e-5);
}

#[test]
fn logit_entropy_matches_direct_definition() {
    let row = [0.3f32, -1.2, 2.5, 0.0, -7.0];
    let direct: f32 = softmax_row(&row, RoutingScoreKind::Logits)
        .iter()
        .map(|&p| -p * p.log2())
        .sum();
    let got = entropy_row(&row, RoutingScoreKind::Logits);
    assert!((got - direct).abs() < 1e-5, "{got} vs {direct}");
}

#[test]
fn extreme_logit_gap_stays_finite() {
    let row = [f32::MAX, -f32::MAX, 0.0];
    let got = entropy_row(&row, RoutingScoreKind::Logits);
    assert_eq!(got, 0.0);
    assert_eq!(
        softmax_row(&row, RoutingScoreKind::Logits),
        vec![1.0, 0.0, 0.0]
    );
}

#[test]
fn probability_rows_sanitize_out_of_range_values() {
    use RoutingScoreKind::Probabilities;
    assert_eq!(
        softmax_row(&[0.5, f32::NAN, f32::INFINITY, -1.0], Probabilities),
        vec![0.5, 0.0, 0.0, 0.0]
    );
    assert_eq!(
        entropy_row(&[0.5, f32::NAN, f32::INFINITY, -1.0], Probabilities),
        0.5
    );
}

#[test]
fn top_k_ranks_values_then_lower_index() {
    use RoutingScoreKind::{Logits, Probabilities};
    assert_eq!(top_k_row(&[0.1, f32::NAN, 0.9], 2, Logits), vec![2, 0]);
    assert_eq!(top_k_row(&[f32::NAN, f32::NAN], 2, Logits), vec![0, 1]);
    assert_eq!(top_k_row(&[1.0, 1.0, 1.0], 2, Logits), vec![0, 1]);
    assert_eq!(top_k_row(&[1.0], 3, Logits), vec![0, -1, -1]);
    assert_eq!(top_k_row(&[], 2, Logits), vec![-1, -1]);
    assert_eq!(
        top_k_row(&[f32::INFINITY, 0.25, f32::NAN], 2, Probabilities),
        vec![1, 0]
    );
}

#[test]
fn top_k_treats_signed_zero_as_equal_and_keeps_lower_index() {
    use RoutingScoreKind::{Logits, Probabilities};
    assert_eq!(top_k_row(&[-0.0, 0.0, -1.0], 3, Logits), vec![0, 1, 2]);
    assert_eq!(
        top_k_row(&[0.0, -0.0, -1.0], 3, Probabilities),
        vec![0, 1, 2]
    );
}

#[test]
fn top_k_order_survives_probability_underflow() {
    // Both losing routes round to probability 0; their logits still rank.
    assert_eq!(
        top_k_row(&[0.0, -500.0, -400.0], 3, RoutingScoreKind::Logits),
        vec![0, 2, 1]
    );
}

#[test]
fn adjusted_score_is_single_rounded() {
    let (state, adaptation, scale) = (1.0f32 + f32::EPSILON, 1.0f32 + f32::EPSILON, 1.0f32);
    assert_eq!(adjusted_score(state, adaptation, scale), 0.0);
    let (a, s) = (1.0f32 + f32::EPSILON, 1.0f32 - f32::EPSILON / 2.0);
    let product = a * s;
    assert_eq!(adjusted_score(product, a, s), (-s).mul_add(a, product));
    assert_ne!(adjusted_score(product, a, s), 0.0);
}

#[test]
fn selection_resolves_ties_and_ineligible_scores() {
    let pick = |s: &[f32], a: &[f32], scale| adjusted_argmax(s, a, scale).unwrap();
    assert_eq!(pick(&[3.0, 1.0, 3.0], &[0.0; 3], 0.5).index, Some(0));
    assert_eq!(
        pick(&[1.0, 4.0, 2.0], &[0.0, f32::NAN, 0.0], 1.0).index,
        Some(2)
    );
    assert_eq!(pick(&[2.0, 2.0], &[0.0, 10.0], 0.5).index, Some(0));
    assert_eq!(pick(&[-1e32, -1e31, -1e32], &[0.0; 3], 0.0).index, Some(1));
}

#[test]
fn selection_handles_empty_and_no_eligible_node() {
    let pick = |s: &[f32], a: &[f32], scale| adjusted_argmax(s, a, scale).unwrap();
    let none = pick(&[f32::NAN, f32::NEG_INFINITY], &[0.0; 2], 1.0);
    assert_eq!(none.index, None);
    assert_eq!(none.score, f32::NEG_INFINITY);

    let empty = pick(&[], &[], 1.0);
    assert_eq!(empty.index, None);

    // inf - inf is NaN, so that node is ineligible rather than winning.
    let inf = pick(&[f32::INFINITY, 1.0], &[f32::INFINITY, 0.0], 1.0);
    assert_eq!(inf.index, Some(1));
}

#[test]
fn selection_raw_conversion_maps_sentinel() {
    assert_eq!(
        AdjustedSelection::from_raw(NO_SELECTION, f32::NEG_INFINITY).index,
        None
    );
    assert_eq!(AdjustedSelection::from_raw(7, 1.0).index, Some(7));
}

#[test]
fn routing_params_reject_invalid_values() {
    assert!(invalid(params(1, 1, 0).validate()));
    assert!(invalid(params(1, 1, MAX_ROUTING_TOP_K + 1).validate()));
    assert!(invalid(params(i32::MAX as usize + 1, 1, 1).validate()));
    assert!(invalid(params(1, i32::MAX as usize + 1, 1).validate()));

    let mut bad_scale = params(1, 1, 1);
    bad_scale.adaptation_scale = f32::NAN;
    assert!(invalid(bad_scale.validate()));
}

#[test]
fn routing_inputs_reject_invalid_values() {
    assert!(invalid(adjusted_argmax(&[1.0], &[1.0], f32::INFINITY)));
    assert!(invalid(adjusted_argmax(&[1.0], &[], 1.0)));

    let p = params(2, 2, 1);
    assert!(invalid(routing_reference(
        &[0.0; 3], &[0.0; 2], &[0.0; 2], &p
    )));
    assert!(invalid(routing_reference(
        &[0.0; 4], &[0.0; 1], &[0.0; 1], &p
    )));
    assert!(invalid(routing_reference(
        &[0.0; 4], &[0.0; 2], &[0.0; 1], &p
    )));
}

#[test]
fn reference_handles_empty_dimensions() {
    let empty = routing_reference(&[], &[], &[], &params(0, 4, 2)).unwrap();
    assert!(empty.top_k_indices.is_empty());
    assert_eq!(empty.summary.entropy_sum, 0.0);
    assert_eq!(empty.summary.selection.index, None);

    let no_routes = routing_reference(&[], &[1.0, 4.0, 2.0], &[0.0; 3], &params(3, 0, 2)).unwrap();
    assert_eq!(no_routes.summary.selection.index, Some(1));
    assert_eq!(no_routes.summary.entropy_max, 0.0);
    assert_eq!(no_routes.top_k_indices, vec![-1; 6]);
    assert_eq!(no_routes.top_k_probabilities, vec![0.0; 6]);
}

#[test]
fn reference_composes_row_operations() {
    let scores = [0.0f32, 1.0, 2.0, 5.0, 5.0, f32::NAN];
    let state = [1.0f32, 2.0];
    let adaptation = [0.0f32, 4.0];
    let got = routing_reference(&scores, &state, &adaptation, &params(2, 3, 2)).unwrap();

    assert_eq!(got.top_k_indices, vec![2, 1, 0, 1]);
    assert_eq!(got.summary.selection.index, Some(0));
    assert_eq!(got.summary.selection.score, 1.0);
    let row0 = softmax_row(&scores[..3], RoutingScoreKind::Logits);
    assert_eq!(got.top_k_probabilities[0], row0[2]);
    assert_eq!(got.top_k_probabilities[2], 0.5);
    let h0 = entropy_row(&scores[..3], RoutingScoreKind::Logits);
    assert!((got.summary.entropy_sum - (h0 + 1.0)).abs() < 1e-5);
    assert_eq!(got.summary.entropy_max, h0.max(1.0));
}

#[test]
fn traffic_counts_composable_path() {
    use analytical::composable;
    // 256 nodes, 4096 routes, k=8: M = 4 MiB, one block.
    let m = 4_194_304u64;
    let c = composable(256, 4096, 8);
    assert_eq!(c.bytes_read, 5 * m + 2048 + 16);
    assert_eq!(c.bytes_written, m + 8192 + 16 + 16);
    assert_eq!(c.launches, 6);
    assert_eq!(composable(0, 16, 4).bytes_total(), 0);
}

#[test]
fn traffic_counts_fused_path() {
    use analytical::fused;
    let m = 4_194_304u64;
    let f = fused(256, 4096, 8);
    assert_eq!(f.bytes_read, 2 * m + 2048 + 16);
    assert_eq!(f.bytes_written, 8192 + 16 + 16);
    assert_eq!(f.launches, 2);
    assert_eq!(fused(0, 16, 4).launches, 0);
}
