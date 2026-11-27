// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

//! GPU evidence for the stable kernel wrappers added in GH #63.
#![cfg(feature = "cuda")]

use myelin_accelerator::{GpuAccelerator, GpuBuffer};

fn eval_clause(assignment: &[u8], clause: &[i32], n_vars: usize) -> bool {
    clause.iter().any(|&lit| {
        let var = (lit as usize) >> 1;
        var < n_vars && (assignment[var] ^ ((lit & 1) as u8)) != 0
    })
}

fn unsat_scores(
    assignment: &[u8],
    clauses: &[i32],
    walkers: usize,
    vars: usize,
    width: usize,
) -> Vec<i32> {
    (0..walkers)
        .map(|w| {
            let row = &assignment[w * vars..(w + 1) * vars];
            clauses
                .chunks_exact(width)
                .filter(|clause| !eval_clause(row, clause, vars))
                .count() as i32
        })
        .collect()
}

#[test]
#[ignore] // requires GPU + driver >= 570
fn lif_rate_and_reset_public_wrappers_match_oracles() {
    use myelin_accelerator::oracle::{lif_step_oracle, lif_step_weighted_oracle};

    let acc = GpuAccelerator::require_gpu().unwrap();

    let mut expected_mem = vec![0.2f32, 0.9, 0.5];
    let current = vec![0.5f32, 0.2, 0.9];
    let mut expected_ref = vec![0u32, 0, 1];
    let expected_spikes = lif_step_oracle(&mut expected_mem, &current, &mut expected_ref);

    let mut membrane = GpuBuffer::from_slice(&[0.2f32, 0.9, 0.5]).unwrap();
    let current_gpu = GpuBuffer::from_slice(&current).unwrap();
    let mut refractory = GpuBuffer::from_slice(&[0u32, 0, 1]).unwrap();
    let mut spikes = GpuBuffer::<u32>::alloc(3).unwrap();
    acc.lif_step(&mut membrane, &current_gpu, &mut refractory, &mut spikes)
        .unwrap();
    assert_eq!(membrane.to_vec().unwrap(), expected_mem);
    assert_eq!(refractory.to_vec().unwrap(), expected_ref);
    assert_eq!(spikes.to_vec().unwrap(), expected_spikes);

    let weights = vec![0.5f32, 0.25, -0.5, 0.75, 1.0, 0.5];
    let input = vec![1.0f32, 0.0, 1.0];
    let mut expected_mem = vec![0.1f32, 0.4];
    let mut expected_ref = vec![0u32, 0];
    let expected_spikes =
        lif_step_weighted_oracle(&mut expected_mem, &weights, &input, &mut expected_ref, 3);
    let mut membrane = GpuBuffer::from_slice(&[0.1f32, 0.4]).unwrap();
    let weights_gpu = GpuBuffer::from_slice(&weights).unwrap();
    let input_gpu = GpuBuffer::from_slice(&input).unwrap();
    let mut refractory = GpuBuffer::from_slice(&[0u32, 0]).unwrap();
    let mut spikes = GpuBuffer::<u32>::alloc(2).unwrap();
    acc.lif_step_weighted(
        &mut membrane,
        &weights_gpu,
        &input_gpu,
        &mut refractory,
        &mut spikes,
    )
    .unwrap();
    assert_eq!(membrane.to_vec().unwrap(), expected_mem);
    assert_eq!(refractory.to_vec().unwrap(), expected_ref);
    assert_eq!(spikes.to_vec().unwrap(), expected_spikes);

    acc.spike_rate(&mut membrane, &input_gpu, 0.5).unwrap();
    let rates = membrane.to_vec().unwrap();
    for (&rate, (&v, &spike)) in rates
        .iter()
        .zip(expected_mem.iter().zip(input.iter().take(2)))
    {
        assert_eq!(rate.to_bits(), (0.5f32.mul_add(v, 0.5 * spike)).to_bits());
    }
    acc.reset_membrane(&mut membrane, -0.25).unwrap();
    assert_eq!(membrane.to_vec().unwrap(), [-0.25, -0.25]);
}

#[test]
#[ignore] // requires GPU + driver >= 570
fn cosine_public_wrappers_agree_on_scores_and_indices() {
    let acc = GpuAccelerator::require_gpu().unwrap();
    let queries = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0];
    let keys = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
    let queries_gpu = GpuBuffer::from_slice(&queries).unwrap();
    let keys_gpu = GpuBuffer::from_slice(&keys).unwrap();
    let mut batched = GpuBuffer::<f32>::alloc(6).unwrap();
    acc.cosine_similarity_batched(&queries_gpu, &keys_gpu, &mut batched, 2, 3, 3)
        .unwrap();

    let mut selected = GpuBuffer::<f32>::alloc(6).unwrap();
    let mut indices = GpuBuffer::<i32>::alloc(4).unwrap();
    acc.cosine_similarity_top_k(
        &queries_gpu,
        &keys_gpu,
        &mut selected,
        &mut indices,
        2,
        3,
        3,
        2,
    )
    .unwrap();
    assert_eq!(batched.to_vec().unwrap(), selected.to_vec().unwrap());
    assert_eq!(indices.to_vec().unwrap(), [0, 2, 1, 2]);
}

#[test]
#[ignore] // requires GPU + driver >= 570
fn walksat_public_wrappers_produce_valid_states() {
    let acc = GpuAccelerator::require_gpu().unwrap();
    let (walkers, vars, width) = (4usize, 3usize, 2usize);
    let clauses_host = [0i32, 2, 1, 4, 3, 5];
    let clause_count = clauses_host.len() / width;
    let mut assignment = GpuBuffer::<u8>::alloc(walkers * vars).unwrap();
    let mut scores = GpuBuffer::<i32>::alloc(walkers).unwrap();
    let clauses = GpuBuffer::from_slice(&clauses_host).unwrap();

    acc.satsolver_init(
        &mut assignment,
        &mut scores,
        &clauses,
        walkers as i32,
        vars as i32,
        clause_count as i32,
        width as i32,
        17,
    )
    .unwrap();
    let mut assignment_host = assignment.to_vec().unwrap();
    assert!(assignment_host.iter().all(|v| *v <= 1));
    assert_eq!(
        scores.to_vec().unwrap(),
        unsat_scores(&assignment_host, &clauses_host, walkers, vars, width)
    );

    acc.satsolver_step(
        &mut assignment,
        &mut scores,
        &clauses,
        walkers as i32,
        vars as i32,
        clause_count as i32,
        width as i32,
        18,
    )
    .unwrap();
    assignment_host = assignment.to_vec().unwrap();
    assert!(assignment_host.iter().all(|v| *v <= 1));
    let scores_host = scores.to_vec().unwrap();
    assert_eq!(
        scores_host,
        unsat_scores(&assignment_host, &clauses_host, walkers, vars, width)
    );

    let mut solved = GpuBuffer::<u8>::alloc(walkers).unwrap();
    acc.satsolver_check_solution(&scores, &mut solved).unwrap();
    let expected: Vec<u8> = scores_host.iter().map(|s| u8::from(*s == 0)).collect();
    assert_eq!(solved.to_vec().unwrap(), expected);
}
