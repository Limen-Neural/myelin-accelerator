// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

//! GPU goldens for fused routing / SAAQ kernels (requires CUDA + sm_120 driver).

#![cfg(all(feature = "cuda", feature = "saaq"))]

use myelin_accelerator::fused::{
    GIF_ADAPTATION_SCALE, RoutingSaaqInput, entropy_row, fused_routing_saaq, saaq_best_walker,
    softmax_row, top_k_indices,
};
use myelin_accelerator::{GpuAccelerator, GpuBuffer};

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn fixture(n_nodes: usize, n_routes: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut scores = Vec::with_capacity(n_nodes * n_routes);
    let mut membrane = Vec::with_capacity(n_nodes);
    let mut adaptation = Vec::with_capacity(n_nodes);
    for i in 0..n_nodes {
        membrane.push((i as f32) * 0.05 - 0.1 * (i % 7) as f32);
        adaptation.push(((i * 3) % 11) as f32 * 0.1);
        for r in 0..n_routes {
            scores.push((i as f32) * 0.02 - (r as f32) * 0.15 + ((i + r) % 5) as f32 * 0.01);
        }
    }
    (scores, membrane, adaptation)
}

#[test]
#[ignore] // requires GPU + driver ≥ 570
fn saaq_unfused_matches_host() {
    let acc = GpuAccelerator::new();
    assert!(acc.is_ready(), "GPU not ready for SAAQ golden");

    let n = 2048usize;
    let (mut membrane, mut adaptation) = (vec![0.0f32; n], vec![0.0f32; n]);
    membrane[17] = 2.0;
    membrane[5 * 256 + 9] = 4.5;
    adaptation[5 * 256 + 9] = 1.0; // still wins: 4.5 - 0.22 = 4.28 > 2.0

    let expected = saaq_best_walker(&membrane, &adaptation, GIF_ADAPTATION_SCALE);
    let d_m = GpuBuffer::from_slice(&membrane).unwrap();
    let d_a = GpuBuffer::from_slice(&adaptation).unwrap();
    let mut d_w = GpuBuffer::<u32>::alloc(1).unwrap();
    acc.saaq_select(&d_m, &d_a, &mut d_w, GIF_ADAPTATION_SCALE)
        .expect("saaq_select");
    assert_eq!(d_w.to_vec().unwrap()[0], expected);
}

#[test]
#[ignore] // requires GPU + driver ≥ 570
fn saaq_unfused_reduces_more_than_32_partials() {
    let acc = GpuAccelerator::new();
    assert!(acc.is_ready(), "GPU not ready for SAAQ golden");

    // 36 blocks of 256 → n_partials = 36, which the old <<<1,32>>> pass2 dropped.
    let n = 9000usize;
    let mut membrane = vec![0.0f32; n];
    let adaptation = vec![0.0f32; n];
    membrane[8500] = 10.0;
    let expected = saaq_best_walker(&membrane, &adaptation, GIF_ADAPTATION_SCALE);
    assert_eq!(expected, 8500);

    let d_m = GpuBuffer::from_slice(&membrane).unwrap();
    let d_a = GpuBuffer::from_slice(&adaptation).unwrap();
    let mut d_w = GpuBuffer::<u32>::alloc(1).unwrap();
    acc.saaq_select(&d_m, &d_a, &mut d_w, GIF_ADAPTATION_SCALE)
        .expect("saaq_select large");
    assert_eq!(d_w.to_vec().unwrap()[0], expected);
    acc.saaq_select_fused(&d_m, &d_a, &mut d_w, GIF_ADAPTATION_SCALE)
        .expect("saaq_select_fused large");
    assert_eq!(d_w.to_vec().unwrap()[0], expected);
}

#[test]
#[ignore] // requires GPU + driver ≥ 570
fn saaq_empty_writes_walker_zero() {
    let acc = GpuAccelerator::new();
    assert!(acc.is_ready(), "GPU not ready");

    let d_m = GpuBuffer::<f32>::alloc(0).unwrap();
    let d_a = GpuBuffer::<f32>::alloc(0).unwrap();
    let mut d_w = GpuBuffer::<u32>::from_slice(&[u32::MAX]).unwrap();
    acc.saaq_select(&d_m, &d_a, &mut d_w, GIF_ADAPTATION_SCALE)
        .unwrap();
    assert_eq!(d_w.to_vec().unwrap()[0], 0);
    d_w.upload(&[u32::MAX]).unwrap();
    acc.saaq_select_fused(&d_m, &d_a, &mut d_w, GIF_ADAPTATION_SCALE)
        .unwrap();
    assert_eq!(d_w.to_vec().unwrap()[0], 0);
}

#[test]
#[ignore] // requires GPU + driver ≥ 570
fn saaq_fused_matches_unfused() {
    let acc = GpuAccelerator::new();
    assert!(acc.is_ready(), "GPU not ready");

    let (_, membrane, adaptation) = fixture(512, 8);
    let d_m = GpuBuffer::from_slice(&membrane).unwrap();
    let d_a = GpuBuffer::from_slice(&adaptation).unwrap();
    let mut unfused = GpuBuffer::<u32>::alloc(1).unwrap();
    let mut fused = GpuBuffer::<u32>::alloc(1).unwrap();
    acc.saaq_select(&d_m, &d_a, &mut unfused, GIF_ADAPTATION_SCALE)
        .unwrap();
    acc.saaq_select_fused(&d_m, &d_a, &mut fused, GIF_ADAPTATION_SCALE)
        .unwrap();
    assert_eq!(unfused.to_vec().unwrap()[0], fused.to_vec().unwrap()[0]);
    assert_eq!(
        fused.to_vec().unwrap()[0],
        saaq_best_walker(&membrane, &adaptation, GIF_ADAPTATION_SCALE)
    );
}

#[test]
#[ignore] // requires GPU + driver ≥ 570
fn saaq_tie_breaks_to_lower_index() {
    let acc = GpuAccelerator::new();
    assert!(acc.is_ready(), "GPU not ready");

    let n = 2048usize;
    let mut membrane = vec![0.0f32; n];
    let adaptation = vec![0.0f32; n];
    membrane[11] = 3.0;
    membrane[3 * 256 + 4] = 3.0;

    let d_m = GpuBuffer::from_slice(&membrane).unwrap();
    let d_a = GpuBuffer::from_slice(&adaptation).unwrap();
    let mut d_w = GpuBuffer::<u32>::alloc(1).unwrap();
    acc.saaq_select(&d_m, &d_a, &mut d_w, GIF_ADAPTATION_SCALE)
        .unwrap();
    assert_eq!(d_w.to_vec().unwrap()[0], 11);
    acc.saaq_select_fused(&d_m, &d_a, &mut d_w, GIF_ADAPTATION_SCALE)
        .unwrap();
    assert_eq!(d_w.to_vec().unwrap()[0], 11);
}

#[test]
#[ignore] // requires GPU + driver ≥ 570
fn routing_softmax_and_entropy_match_host() {
    let acc = GpuAccelerator::new();
    assert!(acc.is_ready(), "GPU not ready");

    let n_nodes = 64usize;
    let n_routes = 8usize;
    let (scores, _, _) = fixture(n_nodes, n_routes);
    let mut expected_probs = Vec::new();
    for node in 0..n_nodes {
        expected_probs.extend(softmax_row(
            &scores[node * n_routes..node * n_routes + n_routes],
        ));
    }

    let d_scores = GpuBuffer::from_slice(&scores).unwrap();
    let mut d_probs = GpuBuffer::<f32>::alloc(n_nodes * n_routes).unwrap();
    acc.routing_softmax(
        &d_scores,
        &mut d_probs,
        n_nodes as i32,
        n_routes as i32,
        true,
    )
    .unwrap();
    let got = d_probs.to_vec().unwrap();
    assert!(max_abs_diff(&got, &expected_probs) < 1e-5);

    let mut d_sum = GpuBuffer::<f32>::alloc(1).unwrap();
    let mut d_max = GpuBuffer::<f32>::alloc(1).unwrap();
    acc.routing_entropy_reduce(
        &d_probs,
        &mut d_sum,
        &mut d_max,
        n_nodes as i32,
        n_routes as i32,
    )
    .unwrap();
    let (h_sum, h_max) =
        myelin_accelerator::fused::routing_entropy(&expected_probs, n_nodes, n_routes);
    assert!((d_sum.to_vec().unwrap()[0] - h_sum).abs() < 1e-4);
    assert!((d_max.to_vec().unwrap()[0] - h_max).abs() < 1e-4);
}

#[test]
#[ignore] // requires GPU + driver ≥ 570
fn routing_saaq_fused_matches_host() {
    let acc = GpuAccelerator::new();
    assert!(acc.is_ready(), "GPU not ready");

    let n_nodes = 128usize;
    let n_routes = 16usize;
    let top_k = 4usize;
    let (scores, membrane, adaptation) = fixture(n_nodes, n_routes);
    let expected = fused_routing_saaq(&RoutingSaaqInput {
        scores: &scores,
        membrane: &membrane,
        adaptation: &adaptation,
        n_nodes,
        n_routes,
        top_k,
        adaptation_scale: GIF_ADAPTATION_SCALE,
        scores_are_logits: true,
    });

    let d_scores = GpuBuffer::from_slice(&scores).unwrap();
    let d_m = GpuBuffer::from_slice(&membrane).unwrap();
    let d_a = GpuBuffer::from_slice(&adaptation).unwrap();
    let mut d_topk = GpuBuffer::<i32>::alloc(n_nodes * top_k).unwrap();
    let mut d_sum = GpuBuffer::<f32>::alloc(1).unwrap();
    let mut d_max = GpuBuffer::<f32>::alloc(1).unwrap();
    let mut d_w = GpuBuffer::<u32>::alloc(1).unwrap();

    acc.routing_saaq_fused(
        &d_scores,
        &d_m,
        &d_a,
        &mut d_topk,
        &mut d_sum,
        &mut d_max,
        &mut d_w,
        n_nodes as i32,
        n_routes as i32,
        top_k as i32,
        GIF_ADAPTATION_SCALE,
        true,
    )
    .expect("routing_saaq_fused");

    let got_walker = d_w.to_vec().unwrap()[0];
    let got_sum = d_sum.to_vec().unwrap()[0];
    let got_max = d_max.to_vec().unwrap()[0];
    let got_topk = d_topk.to_vec().unwrap();

    assert_eq!(got_walker, expected.best_walker);
    assert_eq!(got_topk, expected.top_k_indices);

    // Device-vs-device: fused entropy should match the unfused GPU path
    // (same log2f / softmax family), not only the host reference.
    let mut d_probs = GpuBuffer::<f32>::alloc(n_nodes * n_routes).unwrap();
    let mut u_sum = GpuBuffer::<f32>::alloc(1).unwrap();
    let mut u_max = GpuBuffer::<f32>::alloc(1).unwrap();
    acc.routing_softmax(
        &d_scores,
        &mut d_probs,
        n_nodes as i32,
        n_routes as i32,
        true,
    )
    .unwrap();
    acc.routing_entropy_reduce(
        &d_probs,
        &mut u_sum,
        &mut u_max,
        n_nodes as i32,
        n_routes as i32,
    )
    .unwrap();
    let unfused_sum = u_sum.to_vec().unwrap()[0];
    let unfused_max = u_max.to_vec().unwrap()[0];
    assert!(
        (got_sum - unfused_sum).abs() < 1e-4,
        "fused vs unfused GPU entropy_sum: {got_sum} vs {unfused_sum}"
    );
    assert!(
        (got_max - unfused_max).abs() < 1e-4,
        "fused vs unfused GPU entropy_max: {got_max} vs {unfused_max}"
    );

    let host_tol = 1e-3f32;
    assert!(
        (got_sum - expected.entropy_sum).abs() < host_tol,
        "fused vs host entropy_sum: {got_sum} vs {}",
        expected.entropy_sum
    );
    assert!(
        (got_max - expected.entropy_max).abs() < host_tol,
        "fused vs host entropy_max: {got_max} vs {}",
        expected.entropy_max
    );
}

#[test]
#[ignore] // requires GPU + driver ≥ 570
fn routing_saaq_fused_zero_nodes_writes_defined_outputs() {
    let acc = GpuAccelerator::new();
    assert!(acc.is_ready(), "GPU not ready");

    let d_scores = GpuBuffer::<f32>::from_slice(&[]).unwrap();
    let d_m = GpuBuffer::<f32>::from_slice(&[]).unwrap();
    let d_a = GpuBuffer::<f32>::from_slice(&[]).unwrap();
    let mut d_topk = GpuBuffer::<i32>::from_slice(&[]).unwrap();
    let mut d_sum = GpuBuffer::<f32>::from_slice(&[42.0]).unwrap();
    let mut d_max = GpuBuffer::<f32>::from_slice(&[42.0]).unwrap();
    let mut d_w = GpuBuffer::<u32>::from_slice(&[u32::MAX]).unwrap();

    acc.routing_saaq_fused(
        &d_scores,
        &d_m,
        &d_a,
        &mut d_topk,
        &mut d_sum,
        &mut d_max,
        &mut d_w,
        0,
        0,
        1,
        GIF_ADAPTATION_SCALE,
        true,
    )
    .unwrap();
    assert_eq!(d_sum.to_vec().unwrap()[0], 0.0);
    assert_eq!(d_max.to_vec().unwrap()[0], 0.0);
    assert_eq!(d_w.to_vec().unwrap()[0], 0);
}

#[test]
#[ignore] // requires GPU + driver ≥ 570
fn routing_saaq_fused_zero_routes_still_selects_saaq() {
    let acc = GpuAccelerator::new();
    assert!(acc.is_ready(), "GPU not ready");

    let membrane = [1.0f32, 4.0, 2.0];
    let adaptation = [0.0f32, 0.0, 0.0];
    let d_scores = GpuBuffer::<f32>::from_slice(&[]).unwrap();
    let d_m = GpuBuffer::from_slice(&membrane).unwrap();
    let d_a = GpuBuffer::from_slice(&adaptation).unwrap();
    let mut d_topk = GpuBuffer::<i32>::alloc(6).unwrap();
    let mut d_sum = GpuBuffer::<f32>::alloc(1).unwrap();
    let mut d_max = GpuBuffer::<f32>::alloc(1).unwrap();
    let mut d_w = GpuBuffer::<u32>::alloc(1).unwrap();

    acc.routing_saaq_fused(
        &d_scores,
        &d_m,
        &d_a,
        &mut d_topk,
        &mut d_sum,
        &mut d_max,
        &mut d_w,
        3,
        0,
        2,
        GIF_ADAPTATION_SCALE,
        true,
    )
    .unwrap();
    assert_eq!(d_w.to_vec().unwrap()[0], 1);
    assert_eq!(d_sum.to_vec().unwrap()[0], 0.0);
    assert_eq!(d_max.to_vec().unwrap()[0], 0.0);
    assert_eq!(d_topk.to_vec().unwrap(), vec![-1, -1, -1, -1, -1, -1]);
}

#[test]
#[ignore] // requires GPU + driver ≥ 570
fn saaq_nan_scale_matches_host_walker_zero() {
    let acc = GpuAccelerator::new();
    assert!(acc.is_ready(), "GPU not ready");

    let membrane = [1.0f32, 4.0, 2.0];
    let adaptation = [0.0f32, 0.0, 0.0];
    let expected = saaq_best_walker(&membrane, &adaptation, f32::NAN);
    assert_eq!(expected, 0);

    let d_m = GpuBuffer::from_slice(&membrane).unwrap();
    let d_a = GpuBuffer::from_slice(&adaptation).unwrap();
    let mut d_w = GpuBuffer::<u32>::from_slice(&[u32::MAX]).unwrap();
    acc.saaq_select(&d_m, &d_a, &mut d_w, f32::NAN).unwrap();
    assert_eq!(d_w.to_vec().unwrap()[0], 0);
    d_w.upload(&[u32::MAX]).unwrap();
    acc.saaq_select_fused(&d_m, &d_a, &mut d_w, f32::NAN)
        .unwrap();
    assert_eq!(d_w.to_vec().unwrap()[0], 0);
}

#[test]
#[ignore] // requires GPU + driver ≥ 570
fn saaq_mixed_nan_adaptation_skips_invalid_walker() {
    let acc = GpuAccelerator::new();
    assert!(acc.is_ready(), "GPU not ready");

    let membrane = [1.0f32, 4.0, 2.0];
    let adaptation = [0.0, f32::NAN, 0.0];
    let expected = saaq_best_walker(&membrane, &adaptation, 1.0);
    assert_eq!(expected, 2);

    let d_m = GpuBuffer::from_slice(&membrane).unwrap();
    let d_a = GpuBuffer::from_slice(&adaptation).unwrap();
    let mut d_w = GpuBuffer::<u32>::from_slice(&[u32::MAX]).unwrap();
    acc.saaq_select(&d_m, &d_a, &mut d_w, 1.0).unwrap();
    assert_eq!(d_w.to_vec().unwrap()[0], expected);
    d_w.upload(&[u32::MAX]).unwrap();
    acc.saaq_select_fused(&d_m, &d_a, &mut d_w, 1.0).unwrap();
    assert_eq!(d_w.to_vec().unwrap()[0], expected);
}

fn fused_gpu_matches_host_row(scores: &[f32], n_routes: i32, top_k: i32) {
    let acc = GpuAccelerator::new();
    assert!(acc.is_ready(), "GPU not ready");

    let membrane = [0.0f32];
    let adaptation = [0.0f32];
    let expected = fused_routing_saaq(&RoutingSaaqInput {
        scores,
        membrane: &membrane,
        adaptation: &adaptation,
        n_nodes: 1,
        n_routes: n_routes as usize,
        top_k: top_k as usize,
        adaptation_scale: GIF_ADAPTATION_SCALE,
        scores_are_logits: true,
    });

    let d_scores = GpuBuffer::from_slice(scores).unwrap();
    let d_m = GpuBuffer::from_slice(&membrane).unwrap();
    let d_a = GpuBuffer::from_slice(&adaptation).unwrap();
    let mut d_topk = GpuBuffer::<i32>::alloc(top_k as usize).unwrap();
    let mut d_sum = GpuBuffer::<f32>::alloc(1).unwrap();
    let mut d_max = GpuBuffer::<f32>::alloc(1).unwrap();
    let mut d_w = GpuBuffer::<u32>::alloc(1).unwrap();
    acc.routing_saaq_fused(
        &d_scores,
        &d_m,
        &d_a,
        &mut d_topk,
        &mut d_sum,
        &mut d_max,
        &mut d_w,
        1,
        n_routes,
        top_k,
        GIF_ADAPTATION_SCALE,
        true,
    )
    .unwrap();
    assert_eq!(d_topk.to_vec().unwrap(), expected.top_k_indices);
    assert!((d_sum.to_vec().unwrap()[0] - expected.entropy_sum).abs() < 1e-5);
    assert!((d_max.to_vec().unwrap()[0] - expected.entropy_max).abs() < 1e-5);
}

#[test]
#[ignore] // requires GPU + driver ≥ 570
fn routing_pos_inf_logit_matches_host_topk() {
    let scores = [1.0f32, f32::INFINITY, f32::INFINITY];
    let p = softmax_row(&scores);
    assert!((p[0] - 0.0).abs() < 1e-6);
    assert!((p[1] - 0.5).abs() < 1e-6);
    assert!((p[2] - 0.5).abs() < 1e-6);
    assert_eq!(top_k_indices(&p, 2), vec![1, 2]);
    assert!((entropy_row(&p) - 1.0).abs() < 1e-5);
    fused_gpu_matches_host_row(&scores, 3, 2);
}

#[test]
#[ignore] // requires GPU + driver ≥ 570
fn routing_mixed_nan_logit_matches_host_topk() {
    let scores = [f32::NAN, 0.0];
    let p = softmax_row(&scores);
    assert!((p[0] - 0.0).abs() < 1e-6);
    assert!((p[1] - 1.0).abs() < 1e-6);
    assert_eq!(top_k_indices(&p, 2), vec![1, 0]);
    fused_gpu_matches_host_row(&scores, 2, 2);
}
