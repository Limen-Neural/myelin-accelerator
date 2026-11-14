// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::*;
use super::params;

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
