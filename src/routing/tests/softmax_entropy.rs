// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::*;

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
    assert_eq!(entropy_row(&[0.5, 1.0e-8], Probabilities), 0.5);
}
