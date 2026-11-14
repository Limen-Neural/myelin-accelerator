use super::super::*;
use super::{invalid, params};

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
