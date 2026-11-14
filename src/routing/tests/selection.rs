// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::*;

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
