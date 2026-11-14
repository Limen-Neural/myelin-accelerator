// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Device differential tests for routing reductions and adjusted-score
//! selection, compared with the CPU reference in `myelin_accelerator::routing`.

#![cfg(feature = "cuda")]

use myelin_accelerator::oracle::CaseRng;
use myelin_accelerator::routing::{
    AdjustedSelection, AdjustedSelectionRequest, ENTROPY_ABS_TOL, FusedRoutingOutputs,
    FusedRoutingRequest, MAX_ROUTING_TOP_K, PROBABILITY_ABS_TOL, RoutingEntropyRequest,
    RoutingParams, RoutingReference, RoutingScoreKind, RoutingSoftmaxRequest, RoutingTopKRequest,
    adjusted_argmax, entropy_row, routing_reference,
};
use myelin_accelerator::{GpuAccelerator, GpuBuffer, GpuError};

const SCALE: f32 = 0.22;

fn accelerator() -> GpuAccelerator {
    GpuAccelerator::require_gpu().expect("routing goldens require a usable GPU")
}

fn params(n_nodes: usize, n_routes: usize, top_k: usize) -> RoutingParams {
    RoutingParams {
        n_nodes,
        n_routes,
        top_k,
        score_kind: RoutingScoreKind::Logits,
        adaptation_scale: SCALE,
    }
}

struct Inputs {
    scores: Vec<f32>,
    state: Vec<f32>,
    adaptation: Vec<f32>,
}

/// Seeded inputs with distinct-enough values and no subnormals.
fn seeded(seed: u64, n_nodes: usize, n_routes: usize) -> Inputs {
    let mut rng = CaseRng::new(seed);
    Inputs {
        scores: (0..n_nodes * n_routes)
            .map(|_| rng.next_unit_f32() * 8.0)
            .collect(),
        state: (0..n_nodes).map(|_| rng.next_unit_f32() * 4.0).collect(),
        adaptation: (0..n_nodes).map(|_| rng.next_rate_f32()).collect(),
    }
}

struct DeviceRun {
    top_k_indices: Vec<i32>,
    top_k_probabilities: Vec<f32>,
    entropy_sum: f32,
    entropy_max: f32,
    selection: AdjustedSelection,
}

fn run_fused(acc: &GpuAccelerator, inputs: &Inputs, params: &RoutingParams) -> DeviceRun {
    let scores = GpuBuffer::from_slice(&inputs.scores).unwrap();
    let state = GpuBuffer::from_slice(&inputs.state).unwrap();
    let adaptation = GpuBuffer::from_slice(&inputs.adaptation).unwrap();
    let mut outputs = FusedRoutingOutputs::alloc(params, true).unwrap();
    acc.fused_routing_select(FusedRoutingRequest {
        scores: &scores,
        state: &state,
        adaptation: &adaptation,
        outputs: &mut outputs,
        params,
    })
    .unwrap();
    let summary = outputs.read_summary().unwrap();
    DeviceRun {
        top_k_indices: outputs.top_k_indices.to_vec().unwrap(),
        top_k_probabilities: outputs
            .top_k_probabilities
            .as_ref()
            .unwrap()
            .to_vec()
            .unwrap(),
        entropy_sum: summary.entropy_sum,
        entropy_max: summary.entropy_max,
        selection: summary.selection,
    }
}

fn assert_matches_reference(got: &DeviceRun, expected: &RoutingReference, params: &RoutingParams) {
    let shape = format!("{}x{} k={}", params.n_nodes, params.n_routes, params.top_k);
    assert_eq!(got.top_k_indices, expected.top_k_indices, "{shape}");
    assert_eq!(
        got.selection.index, expected.summary.selection.index,
        "{shape}"
    );
    assert_eq!(
        got.selection.score, expected.summary.selection.score,
        "{shape}"
    );
    for (i, (g, e)) in got
        .top_k_probabilities
        .iter()
        .zip(&expected.top_k_probabilities)
        .enumerate()
    {
        assert!(
            (g - e).abs() <= PROBABILITY_ABS_TOL,
            "{shape} probability {i}: device {g}, reference {e}"
        );
    }
    let sum_tol = ENTROPY_ABS_TOL * params.n_nodes.max(1) as f32;
    assert!(
        (got.entropy_sum - expected.summary.entropy_sum).abs() <= sum_tol,
        "{shape} entropy_sum: device {}, reference {}",
        got.entropy_sum,
        expected.summary.entropy_sum
    );
    assert!(
        (got.entropy_max - expected.summary.entropy_max).abs() <= ENTROPY_ABS_TOL,
        "{shape} entropy_max: device {}, reference {}",
        got.entropy_max,
        expected.summary.entropy_max
    );
}

fn select_both(
    acc: &GpuAccelerator,
    state: &[f32],
    adaptation: &[f32],
    scale: f32,
) -> (AdjustedSelection, AdjustedSelection) {
    let d_state = GpuBuffer::from_slice(state).unwrap();
    let d_adaptation = GpuBuffer::from_slice(adaptation).unwrap();
    let mut index = GpuBuffer::<u32>::alloc(1).unwrap();
    let mut score = GpuBuffer::<f32>::alloc(1).unwrap();
    acc.adjusted_argmax(AdjustedSelectionRequest {
        state: &d_state,
        adaptation: &d_adaptation,
        selected_index: &mut index,
        selected_score: &mut score,
        scale,
    })
    .unwrap();
    let two_pass =
        AdjustedSelection::from_raw(index.to_vec().unwrap()[0], score.to_vec().unwrap()[0]);
    acc.adjusted_argmax_single_block(AdjustedSelectionRequest {
        state: &d_state,
        adaptation: &d_adaptation,
        selected_index: &mut index,
        selected_score: &mut score,
        scale,
    })
    .unwrap();
    let single =
        AdjustedSelection::from_raw(index.to_vec().unwrap()[0], score.to_vec().unwrap()[0]);
    (two_pass, single)
}

fn assert_selection_paths(acc: &GpuAccelerator, state: &[f32], adaptation: &[f32], scale: f32) {
    let expected = adjusted_argmax(state, adaptation, scale).unwrap();
    let (two_pass, single) = select_both(acc, state, adaptation, scale);
    assert_eq!(two_pass, expected, "two-pass, n={}", state.len());
    assert_eq!(single, expected, "single-block, n={}", state.len());
}

#[test]
#[ignore] // requires GPU + driver ≥ 570
fn fused_matches_reference_across_block_counts() {
    let acc = accelerator();
    for &(n_nodes, n_routes, top_k) in &[
        (1usize, 1usize, 1usize),
        (31, 7, 3),
        (256, 16, 4),
        (257, 16, 4),
        (2048, 16, 4),
        (1024, 128, MAX_ROUTING_TOP_K),
        (9000, 8, 2),
        (64, 4096, MAX_ROUTING_TOP_K),
    ] {
        for seed in [11u64, 12] {
            let p = params(n_nodes, n_routes, top_k);
            let inputs = seeded(seed + n_nodes as u64, n_nodes, n_routes);
            let got = run_fused(&acc, &inputs, &p);
            let expected =
                routing_reference(&inputs.scores, &inputs.state, &inputs.adaptation, &p).unwrap();
            assert_matches_reference(&got, &expected, &p);
        }
    }
}

#[test]
#[ignore] // requires GPU + driver ≥ 570
fn fused_special_values_match_reference() {
    let acc = accelerator();
    let (inf, neg, nan) = (f32::INFINITY, f32::NEG_INFINITY, f32::NAN);
    let rows: [[f32; 4]; 8] = [
        [nan, 0.0, 1.0, 2.0],
        [1.0, inf, 2.0, inf],
        [neg, neg, neg, neg],
        [nan, nan, nan, nan],
        [nan, neg, nan, neg],
        [3.0, 3.0, 3.0, 3.0],
        [f32::MAX, -f32::MAX, 0.0, 1.0],
        [0.0, -500.0, -400.0, -600.0],
    ];
    let inputs = Inputs {
        scores: rows.iter().flatten().copied().collect(),
        state: vec![1.0, 5.0, nan, 5.0, inf, neg, 2.0, 0.0],
        adaptation: vec![0.0, 0.0, 0.0, 0.0, inf, 0.0, nan, 0.0],
    };
    let p = params(rows.len(), 4, 4);
    let got = run_fused(&acc, &inputs, &p);
    let expected =
        routing_reference(&inputs.scores, &inputs.state, &inputs.adaptation, &p).unwrap();
    assert_matches_reference(&got, &expected, &p);
    assert_eq!(got.selection.index, Some(1), "tie keeps the lower index");
}

#[test]
#[ignore] // requires GPU + driver ≥ 570
fn fused_probability_rows_match_reference() {
    let acc = accelerator();
    let inputs = Inputs {
        scores: vec![
            0.5,
            0.25,
            0.25,
            0.0,
            f32::NAN,
            f32::INFINITY,
            -1.0,
            0.75,
            2.0,
            2.0,
            f32::NEG_INFINITY,
            1e-9,
        ],
        state: vec![0.0, 1.0, 2.0],
        adaptation: vec![0.0, 0.0, 0.0],
    };
    let mut p = params(3, 4, 3);
    p.score_kind = RoutingScoreKind::Probabilities;
    let got = run_fused(&acc, &inputs, &p);
    let expected =
        routing_reference(&inputs.scores, &inputs.state, &inputs.adaptation, &p).unwrap();
    assert_matches_reference(&got, &expected, &p);
}

#[test]
#[ignore] // requires GPU + driver ≥ 570
fn fused_handles_empty_dimensions() {
    let acc = accelerator();

    let p = params(0, 4, 2);
    let inputs = Inputs {
        scores: vec![],
        state: vec![],
        adaptation: vec![],
    };
    let got = run_fused(&acc, &inputs, &p);
    assert_eq!(got.entropy_sum, 0.0);
    assert_eq!(got.entropy_max, 0.0);
    assert_eq!(got.selection.index, None);
    assert_eq!(got.selection.score, f32::NEG_INFINITY);

    let p = params(3, 0, 2);
    let inputs = Inputs {
        scores: vec![],
        state: vec![1.0, 4.0, 2.0],
        adaptation: vec![0.0; 3],
    };
    let got = run_fused(&acc, &inputs, &p);
    assert_eq!(got.selection.index, Some(1));
    assert_eq!(got.entropy_max, 0.0);
    assert_eq!(got.top_k_indices, vec![-1; 6]);
    assert_eq!(got.top_k_probabilities, vec![0.0; 6]);
}

#[test]
#[ignore] // requires GPU + driver ≥ 570
fn fused_pads_when_top_k_exceeds_routes() {
    let acc = accelerator();
    let p = params(2, 2, 4);
    let inputs = seeded(5, 2, 2);
    let got = run_fused(&acc, &inputs, &p);
    let expected =
        routing_reference(&inputs.scores, &inputs.state, &inputs.adaptation, &p).unwrap();
    assert_matches_reference(&got, &expected, &p);
    assert_eq!(&got.top_k_indices[2..4], &[-1, -1]);
}

#[test]
#[ignore] // requires GPU + driver ≥ 570
fn fused_without_probabilities_matches_with_probabilities() {
    let acc = accelerator();
    let p = params(600, 32, 4);
    let inputs = seeded(21, 600, 32);
    let with = run_fused(&acc, &inputs, &p);

    let scores = GpuBuffer::from_slice(&inputs.scores).unwrap();
    let state = GpuBuffer::from_slice(&inputs.state).unwrap();
    let adaptation = GpuBuffer::from_slice(&inputs.adaptation).unwrap();
    let mut outputs = FusedRoutingOutputs::alloc(&p, false).unwrap();
    acc.fused_routing_select(FusedRoutingRequest {
        scores: &scores,
        state: &state,
        adaptation: &adaptation,
        outputs: &mut outputs,
        params: &p,
    })
    .unwrap();
    let summary = outputs.read_summary().unwrap();

    assert_eq!(outputs.top_k_indices.to_vec().unwrap(), with.top_k_indices);
    assert_eq!(summary.selection, with.selection);
    assert_eq!(summary.entropy_sum.to_bits(), with.entropy_sum.to_bits());
    assert_eq!(summary.entropy_max.to_bits(), with.entropy_max.to_bits());
}

#[test]
#[ignore] // requires GPU + driver ≥ 570
fn composable_path_matches_fused_results() {
    let acc = accelerator();
    for &(n_nodes, n_routes, top_k) in &[(2048usize, 16usize, 4usize), (9000, 8, 2), (64, 1024, 8)]
    {
        let p = params(n_nodes, n_routes, top_k);
        let inputs = seeded(31 + n_nodes as u64, n_nodes, n_routes);
        let fused = run_fused(&acc, &inputs, &p);

        let scores = GpuBuffer::from_slice(&inputs.scores).unwrap();
        let state = GpuBuffer::from_slice(&inputs.state).unwrap();
        let adaptation = GpuBuffer::from_slice(&inputs.adaptation).unwrap();
        let mut probabilities = GpuBuffer::<f32>::alloc(n_nodes * n_routes).unwrap();
        let mut top_k_indices = GpuBuffer::<i32>::alloc(n_nodes * top_k).unwrap();
        let mut entropy_sum = GpuBuffer::<f32>::alloc(1).unwrap();
        let mut entropy_max = GpuBuffer::<f32>::alloc(1).unwrap();
        let mut index = GpuBuffer::<u32>::alloc(1).unwrap();
        let mut score = GpuBuffer::<f32>::alloc(1).unwrap();

        acc.routing_softmax(RoutingSoftmaxRequest {
            scores: &scores,
            probabilities: &mut probabilities,
            n_nodes,
            n_routes,
            score_kind: p.score_kind,
        })
        .unwrap();
        acc.routing_top_k(RoutingTopKRequest {
            scores: &scores,
            top_k_indices: &mut top_k_indices,
            n_nodes,
            n_routes,
            top_k,
            score_kind: p.score_kind,
        })
        .unwrap();
        acc.routing_entropy_reduce(RoutingEntropyRequest {
            probabilities: &probabilities,
            entropy_sum: &mut entropy_sum,
            entropy_max: &mut entropy_max,
            n_nodes,
            n_routes,
        })
        .unwrap();
        acc.adjusted_argmax(AdjustedSelectionRequest {
            state: &state,
            adaptation: &adaptation,
            selected_index: &mut index,
            selected_score: &mut score,
            scale: SCALE,
        })
        .unwrap();

        assert_eq!(top_k_indices.to_vec().unwrap(), fused.top_k_indices);
        assert_eq!(
            AdjustedSelection::from_raw(index.to_vec().unwrap()[0], score.to_vec().unwrap()[0]),
            fused.selection
        );
        let sum = entropy_sum.to_vec().unwrap()[0];
        let max = entropy_max.to_vec().unwrap()[0];
        assert!(
            (sum - fused.entropy_sum).abs() <= ENTROPY_ABS_TOL * n_nodes as f32,
            "{n_nodes}x{n_routes}: composable sum {sum}, fused {}",
            fused.entropy_sum
        );
        assert!(
            (max - fused.entropy_max).abs() <= ENTROPY_ABS_TOL,
            "{n_nodes}x{n_routes}: composable max {max}, fused {}",
            fused.entropy_max
        );
    }
}

#[test]
#[ignore] // requires GPU + driver ≥ 570
fn composable_entropy_sanitizes_nonfinite_probabilities() {
    let acc = accelerator();
    let (n_nodes, n_routes) = (3usize, 4usize);
    let probabilities = vec![
        0.5,
        f32::NAN,
        f32::INFINITY,
        -1.0,
        0.25,
        0.25,
        0.25,
        0.25,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        0.0,
    ];
    let probs = GpuBuffer::from_slice(&probabilities).unwrap();
    let mut entropy_sum = GpuBuffer::<f32>::alloc(1).unwrap();
    let mut entropy_max = GpuBuffer::<f32>::alloc(1).unwrap();

    acc.routing_entropy_reduce(RoutingEntropyRequest {
        probabilities: &probs,
        entropy_sum: &mut entropy_sum,
        entropy_max: &mut entropy_max,
        n_nodes,
        n_routes,
    })
    .unwrap();

    let row_entropy = |node: usize| {
        entropy_row(
            &probabilities[node * n_routes..(node + 1) * n_routes],
            RoutingScoreKind::Probabilities,
        )
    };
    let expected_sum: f32 = (0..n_nodes).map(row_entropy).sum();
    let expected_max = (0..n_nodes).map(row_entropy).fold(0.0f32, f32::max);

    let sum = entropy_sum.to_vec().unwrap()[0];
    let max = entropy_max.to_vec().unwrap()[0];
    assert!(
        (sum - expected_sum).abs() <= ENTROPY_ABS_TOL * n_nodes as f32,
        "entropy_sum: device {sum}, reference {expected_sum}"
    );
    assert!(
        (max - expected_max).abs() <= ENTROPY_ABS_TOL,
        "entropy_max: device {max}, reference {expected_max}"
    );
}

#[test]
#[ignore] // requires GPU + driver ≥ 570
fn selection_paths_match_reference() {
    let acc = accelerator();
    for &n in &[1usize, 2, 255, 256, 257, 2048, 8192, 8193, 20_000] {
        let inputs = seeded(41 + n as u64, n, 0);
        assert_selection_paths(&acc, &inputs.state, &inputs.adaptation, SCALE);
    }
}

#[test]
#[ignore] // requires GPU + driver ≥ 570
fn selection_tie_spanning_blocks_keeps_lower_index() {
    let acc = accelerator();
    let n = 2048;
    let mut state = vec![0.0f32; n];
    let adaptation = vec![0.0f32; n];
    state[1500] = 9.0;
    state[300] = 9.0;
    state[1900] = 9.0;
    assert_selection_paths(&acc, &state, &adaptation, SCALE);
    assert_eq!(
        select_both(&acc, &state, &adaptation, SCALE).0.index,
        Some(300)
    );
}

#[test]
#[ignore] // requires GPU + driver ≥ 570
fn selection_reduces_more_than_32_partials() {
    let acc = accelerator();
    let n = 9000;
    let mut state = vec![0.0f32; n];
    let adaptation = vec![0.0f32; n];
    state[8800] = 7.0;
    assert_selection_paths(&acc, &state, &adaptation, SCALE);
}

#[test]
#[ignore] // requires GPU + driver ≥ 570
fn selection_special_values_and_no_selection() {
    let acc = accelerator();
    let (inf, neg, nan) = (f32::INFINITY, f32::NEG_INFINITY, f32::NAN);

    assert_selection_paths(&acc, &[1.0, 4.0, 2.0], &[0.0, nan, 0.0], 1.0);
    assert_selection_paths(&acc, &[-1e32, -1e31, -1e32], &[0.0; 3], 0.0);
    assert_selection_paths(&acc, &[inf, 1.0], &[inf, 0.0], 1.0);
    assert_selection_paths(&acc, &[1.0, inf, inf], &[0.0, 0.0, 0.0], 1.0);

    let none = adjusted_argmax(&[nan, neg], &[0.0; 2], 1.0).unwrap();
    assert_eq!(none.index, None);
    assert_selection_paths(&acc, &[nan, neg], &[0.0; 2], 1.0);
    assert_selection_paths(&acc, &[], &[], 1.0);
}

#[test]
#[ignore] // requires GPU + driver ≥ 570
fn scratch_grows_and_async_path_orders_launches() {
    let acc = accelerator();
    for &n_nodes in &[256usize, 9000, 512] {
        let p = params(n_nodes, 8, 2);
        let inputs = seeded(51 + n_nodes as u64, n_nodes, 8);
        let scores = GpuBuffer::from_slice(&inputs.scores).unwrap();
        let state = GpuBuffer::from_slice(&inputs.state).unwrap();
        let adaptation = GpuBuffer::from_slice(&inputs.adaptation).unwrap();
        let mut outputs = FusedRoutingOutputs::alloc(&p, false).unwrap();
        acc.fused_routing_select_async(FusedRoutingRequest {
            scores: &scores,
            state: &state,
            adaptation: &adaptation,
            outputs: &mut outputs,
            params: &p,
        })
        .unwrap();
        acc.synchronize().unwrap();
        let expected =
            routing_reference(&inputs.scores, &inputs.state, &inputs.adaptation, &p).unwrap();
        assert_eq!(
            outputs.read_summary().unwrap().selection,
            expected.summary.selection
        );
        assert_eq!(
            outputs.top_k_indices.to_vec().unwrap(),
            expected.top_k_indices
        );
    }
}

#[test]
#[ignore] // requires GPU + driver ≥ 570
fn invalid_inputs_fail_before_launch() {
    let acc = accelerator();
    let p = params(2, 2, 1);
    let good = GpuBuffer::from_slice(&[0.0f32; 4]).unwrap();
    let short = GpuBuffer::from_slice(&[0.0f32; 1]).unwrap();
    let state = GpuBuffer::from_slice(&[0.0f32; 2]).unwrap();
    let mut outputs = FusedRoutingOutputs::alloc(&p, false).unwrap();

    let result = acc.fused_routing_select(FusedRoutingRequest {
        scores: &short,
        state: &state,
        adaptation: &state,
        outputs: &mut outputs,
        params: &p,
    });
    assert!(matches!(result, Err(GpuError::InvalidInput(_))));
    let result = acc.fused_routing_select(FusedRoutingRequest {
        scores: &good,
        state: &short,
        adaptation: &state,
        outputs: &mut outputs,
        params: &p,
    });
    assert!(matches!(result, Err(GpuError::InvalidInput(_))));

    let mut bad_k = p;
    bad_k.top_k = MAX_ROUTING_TOP_K + 1;
    let result = acc.fused_routing_select(FusedRoutingRequest {
        scores: &good,
        state: &state,
        adaptation: &state,
        outputs: &mut outputs,
        params: &bad_k,
    });
    assert!(matches!(result, Err(GpuError::InvalidInput(_))));

    let mut index = GpuBuffer::<u32>::alloc(1).unwrap();
    let mut score = GpuBuffer::<f32>::alloc(1).unwrap();
    let result = acc.adjusted_argmax(AdjustedSelectionRequest {
        state: &state,
        adaptation: &short,
        selected_index: &mut index,
        selected_score: &mut score,
        scale: 1.0,
    });
    assert!(matches!(result, Err(GpuError::InvalidInput(_))));
    let result = acc.adjusted_argmax(AdjustedSelectionRequest {
        state: &state,
        adaptation: &state,
        selected_index: &mut index,
        selected_score: &mut score,
        scale: f32::NAN,
    });
    assert!(matches!(result, Err(GpuError::InvalidInput(_))));
}
