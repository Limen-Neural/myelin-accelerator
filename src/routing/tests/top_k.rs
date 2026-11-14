use super::super::*;

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
