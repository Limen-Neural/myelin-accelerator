// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Raw-kernel regressions for LIM-1305. Run with `--features cuda -- --ignored`.

#![cfg(feature = "cuda")]

use cust::launch;
use cust::stream::{Stream, StreamFlags};
use myelin_accelerator::{GpuAccelerator, GpuBuffer, GpuContext, KernelModule};

#[test]
#[ignore] // requires GPU + driver >= 570
fn stdp_accelerator_advances_weights_and_traces() {
    let acc = GpuAccelerator::require_gpu().unwrap();
    let mut weights = GpuBuffer::from_slice(&[1.0f32; 4]).unwrap();
    let pre_spikes = GpuBuffer::from_slice(&[1.0f32, 0.0]).unwrap();
    let post_spikes = GpuBuffer::from_slice(&[1.0f32, 0.0]).unwrap();
    let mut pre_traces = GpuBuffer::from_slice(&[1.0f32; 2]).unwrap();
    let mut post_traces = GpuBuffer::from_slice(&[1.0f32; 2]).unwrap();

    acc.stdp_update(
        &mut weights,
        &pre_spikes,
        &post_spikes,
        &mut pre_traces,
        &mut post_traces,
        2,
        2,
        20.0,
    )
    .unwrap();

    let decay = (-1.0f32).exp();
    assert!((pre_traces.to_vec().unwrap()[0] - (1.0 + decay)).abs() < 1e-5);
    assert!((post_traces.to_vec().unwrap()[1] - decay).abs() < 1e-5);
    assert!((weights.to_vec().unwrap()[0] - (1.0 - 0.002 * (1.0 + decay))).abs() < 1e-5);
}

#[test]
#[ignore] // requires GPU + driver >= 570
fn stdp_accelerator_handles_post_extent_past_grid_y_limit() {
    let acc = GpuAccelerator::require_gpu().unwrap();
    let n_post = 1_048_561usize;
    let mut weights = GpuBuffer::from_slice(&vec![1.0f32; n_post]).unwrap();
    let pre_spikes = GpuBuffer::from_slice(&[0.0f32]).unwrap();
    let mut post_spike_data = vec![0.0f32; n_post];
    post_spike_data[n_post - 1] = 1.0;
    let post_spikes = GpuBuffer::from_slice(&post_spike_data).unwrap();
    let mut pre_traces = GpuBuffer::from_slice(&[1.0f32]).unwrap();
    let mut post_traces = GpuBuffer::from_slice(&vec![1.0f32; n_post]).unwrap();

    acc.stdp_update(
        &mut weights,
        &pre_spikes,
        &post_spikes,
        &mut pre_traces,
        &mut post_traces,
        n_post as i32,
        1,
        20.0,
    )
    .unwrap();

    let got = post_traces.to_vec().unwrap();
    let decay = (-1.0f32).exp();
    assert!((got[0] - decay).abs() < 1e-5);
    assert!((got[n_post - 1] - (1.0 + decay)).abs() < 1e-5);
    assert!((weights.to_vec().unwrap()[n_post - 1] - (1.0 + 0.01 * decay)).abs() < 1e-5);
}

#[test]
#[ignore] // requires GPU + driver >= 570
fn satsolver_aux_skips_out_of_range_literals() {
    let acc = GpuAccelerator::require_gpu().unwrap();
    let assignment = GpuBuffer::from_slice(&[1u8, 0u8]).unwrap();
    let mut flags = GpuBuffer::<u8>::alloc(2).unwrap();
    let scores = GpuBuffer::from_slice(&[1i32]).unwrap();
    let mut best_score = GpuBuffer::<i32>::alloc(1).unwrap();
    let mut best_walker = GpuBuffer::<i32>::alloc(1).unwrap();
    // One clause has a valid satisfying literal after an invalid one.
    // The second clause has no valid literal and is unsatisfied.
    let clauses = GpuBuffer::from_slice(&[198i32, 0, -1, 198]).unwrap();

    acc.satsolver_aux_reduce_best(
        &assignment,
        &mut flags,
        &scores,
        &mut best_score,
        &mut best_walker,
        &clauses,
        1,
        2,
        2,
        2,
    )
    .unwrap();
    assert_eq!(flags.to_vec().unwrap(), [1, 0]);
}

#[test]
#[ignore] // requires GPU + driver >= 570
fn satsolver_init_scores_mixed_and_all_invalid_clauses() {
    let _ctx = GpuContext::init().unwrap();
    let kernels = KernelModule::load().unwrap();
    let init = kernels.get_function("satsolver_init").unwrap();
    let stream = Stream::new(StreamFlags::DEFAULT, None).unwrap();
    let assignment = GpuBuffer::<u8>::alloc(1).unwrap();
    let scores = GpuBuffer::<i32>::alloc(1).unwrap();
    let clauses = GpuBuffer::from_slice(&[198i32, 0, -1, 198]).unwrap();

    // SAFETY: one walker, one variable, and two clauses of length two fit
    // exactly in the buffers; all buffers live through synchronization.
    unsafe {
        launch!(init<<<1u32, 32u32, 0u32, stream>>>(
            assignment.as_device_ptr(),
            scores.as_device_ptr(),
            clauses.as_device_ptr(),
            1i32, 1i32, 2i32, 2i32, 7u32
        ))
        .unwrap();
    }
    stream.synchronize().unwrap();
    let x0 = assignment.to_vec().unwrap()[0];
    assert!(x0 <= 1);
    assert_eq!(scores.to_vec().unwrap(), [if x0 == 1 { 1 } else { 2 }]);
}

#[test]
#[ignore] // requires GPU + driver >= 570
fn satsolver_step_skips_invalid_literal_in_both_flip_branches() {
    for seed in [0u32, 1u32] {
        run_single_clause_step(&[198, 0], 1, seed, 1, 0);
    }
}

#[test]
#[ignore] // requires GPU + driver >= 570
fn satsolver_step_can_select_unsat_clause_past_64() {
    let _ctx = GpuContext::init().unwrap();
    let kernels = KernelModule::load().unwrap();
    let step = kernels.get_function("satsolver_step").unwrap();
    let stream = Stream::new(StreamFlags::DEFAULT, None).unwrap();
    let assignment = GpuBuffer::from_slice(&[0u8]).unwrap();
    let scores = GpuBuffer::from_slice(&[65i32]).unwrap();
    // The first 64 clauses contain only invalid literals. Clause 64 is x0.
    // Seed 36 selects rank 64 of the 65 unsatisfied clauses; x0 must flip.
    let mut clauses = vec![198i32; 64];
    clauses.push(0);
    let clauses = GpuBuffer::from_slice(&clauses).unwrap();

    // SAFETY: one walker owns its assignment/score, all buffers cover the
    // declared dimensions, and they outlive the synchronized launch.
    unsafe {
        launch!(step<<<1u32, 32u32, 0u32, stream>>>(
            assignment.as_device_ptr(),
            scores.as_device_ptr(),
            clauses.as_device_ptr(),
            1i32, 1i32, 65i32, 1i32, 36u32
        ))
        .unwrap();
    }
    stream.synchronize().unwrap();
    assert_eq!(assignment.to_vec().unwrap(), [1]);
    assert_eq!(scores.to_vec().unwrap(), [64]);
}

#[test]
#[ignore] // requires GPU + driver >= 570
fn satsolver_step_all_invalid_literals_stay_unsatisfied() {
    for seed in [0u32, 1u32] {
        run_single_clause_step(&[-1], 0, seed, 0, 1);
    }
}

fn run_single_clause_step(
    literals: &[i32],
    initial_score: i32,
    seed: u32,
    expected_assignment: u8,
    expected_score: i32,
) {
    let _ctx = GpuContext::init().unwrap();
    let kernels = KernelModule::load().unwrap();
    let step = kernels.get_function("satsolver_step").unwrap();
    let stream = Stream::new(StreamFlags::DEFAULT, None).unwrap();
    let assignment = GpuBuffer::from_slice(&[0u8]).unwrap();
    let scores = GpuBuffer::from_slice(&[initial_score]).unwrap();
    let clauses = GpuBuffer::from_slice(literals).unwrap();

    // Seed 0 takes greedy selection; seed 1 takes random selection. All
    // buffers match the declared one-walker, one-variable shape and live
    // through synchronization.
    unsafe {
        launch!(step<<<1u32, 32u32, 0u32, stream>>>(
            assignment.as_device_ptr(),
            scores.as_device_ptr(),
            clauses.as_device_ptr(),
            1i32, 1i32, 1i32, literals.len() as i32, seed
        ))
        .unwrap();
    }
    stream.synchronize().unwrap();
    assert_eq!(
        assignment.to_vec().unwrap(),
        [expected_assignment],
        "seed={seed}"
    );
    assert_eq!(scores.to_vec().unwrap(), [expected_score], "seed={seed}");
}

#[test]
#[ignore] // requires GPU + driver >= 570; suitable for compute-sanitizer racecheck
fn stdp_traces_have_one_writer_across_multiple_blocks() {
    let _ctx = GpuContext::init().unwrap();
    let kernels = KernelModule::load().unwrap();
    let update_weights = kernels.get_function("stdp_update_weights").unwrap();
    let update_traces = kernels.get_function("stdp_update_traces").unwrap();
    let stream = Stream::new(StreamFlags::DEFAULT, None).unwrap();
    let (n_pre, n_post) = (33usize, 17usize);
    let weights = GpuBuffer::from_slice(&vec![1.0f32; n_pre * n_post]).unwrap();
    let mut pre_spikes = vec![0.0f32; n_pre];
    let mut post_spikes = vec![0.0f32; n_post];
    pre_spikes[0] = 1.0;
    post_spikes[0] = 1.0;
    let pre_spikes = GpuBuffer::from_slice(&pre_spikes).unwrap();
    let post_spikes = GpuBuffer::from_slice(&post_spikes).unwrap();
    let pre_traces = GpuBuffer::from_slice(&vec![1.0f32; n_pre]).unwrap();
    let post_traces = GpuBuffer::from_slice(&vec![1.0f32; n_post]).unwrap();

    // SAFETY: matrix/vector dimensions match; no buffer is reused until sync.
    unsafe {
        launch!(update_weights<<<5u32, 128u32, 0u32, stream>>>(
            weights.as_device_ptr(),
            pre_spikes.as_device_ptr(),
            post_spikes.as_device_ptr(),
            pre_traces.as_device_ptr(),
            post_traces.as_device_ptr(),
            n_post as i32, n_pre as i32, 20.0f32
        ))
        .unwrap();
    }
    stream.synchronize().unwrap();
    assert_eq!(pre_traces.to_vec().unwrap(), vec![1.0; n_pre]);
    assert_eq!(post_traces.to_vec().unwrap(), vec![1.0; n_post]);

    // SAFETY: each 1-D thread owns one trace index; same stream orders this
    // after the weight kernel, with all buffers kept alive until sync.
    unsafe {
        launch!(update_traces<<<2u32, 32u32, 0u32, stream>>>(
            pre_spikes.as_device_ptr(),
            post_spikes.as_device_ptr(),
            pre_traces.as_device_ptr(),
            post_traces.as_device_ptr(),
            n_post as i32, n_pre as i32, 20.0f32
        ))
        .unwrap();
    }
    stream.synchronize().unwrap();
    let decay = (-1.0f32).exp();
    let pre = pre_traces.to_vec().unwrap();
    let post = post_traces.to_vec().unwrap();
    for (i, &trace) in pre.iter().enumerate() {
        assert!((trace - (decay + f32::from(i == 0))).abs() < 1e-5);
    }
    for (i, &trace) in post.iter().enumerate() {
        assert!((trace - (decay + f32::from(i == 0))).abs() < 1e-5);
    }
    let got = weights.to_vec().unwrap();
    let expected = 1.0f32 - 0.002 * (1.0 + decay);
    assert!((got[0] - expected).abs() < 1e-5);
    assert!((got[n_pre * n_post - 1] - 1.0).abs() < 1e-5);
}
