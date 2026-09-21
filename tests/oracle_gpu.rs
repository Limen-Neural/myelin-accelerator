// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Device vs CPU-oracle differential tests (requires CUDA + sm_120 driver).

#![cfg(feature = "cuda")]

use myelin_accelerator::bitpacking::{pack_ternary_matrix, uniform_group_scales};
use myelin_accelerator::oracle::{
    BOUNDARY_LENS, CASE_SEEDS, CaseRng, TERNARY_ABS_TOL, TERNARY_REL_TOL, assert_exact, assert_f32,
    fill_sat_scores, pin_poisson_boundary_stimuli, poisson_encode_oracle,
    satsolver_aux_reduce_best_oracle, satsolver_extract_oracle, ternary_gemm_oracle,
    ternary_gemv_oracle,
};
use myelin_accelerator::{GpuAccelerator, GpuBuffer};

fn require_gpu() -> GpuAccelerator {
    let acc = GpuAccelerator::new();
    assert!(
        acc.is_ready(),
        "GPU not ready for oracle differential tests"
    );
    acc
}

// ── poisson_encode ──────────────────────────────────────────────────────────

#[test]
#[ignore] // requires GPU + driver ≥ 570
fn poisson_encode_matches_oracle_golden() {
    let acc = require_gpu();
    let stim = vec![1.0f32, 0.0, 0.2, 0.3, -0.0, 1.5];
    let expected = poisson_encode_oracle(&stim, 0);
    let d_stim = GpuBuffer::from_slice(&stim).unwrap();
    let mut d_spikes = GpuBuffer::<u32>::alloc(stim.len()).unwrap();
    acc.poisson_encode(&d_stim, &mut d_spikes, 0)
        .expect("poisson_encode");
    let got = d_spikes.to_vec().unwrap();
    assert_exact(&got, &expected, 0, "n=6 seed=0 golden");
}

#[test]
#[ignore] // requires GPU + driver ≥ 570
fn poisson_encode_matches_oracle_seeded_boundaries() {
    let acc = require_gpu();
    for &seed in CASE_SEEDS {
        for &n in BOUNDARY_LENS {
            // A 0-thread launch is not a useful device comparison; CPU covers n=0.
            if n == 0 {
                continue;
            }
            let mut rng = CaseRng::new(seed.wrapping_mul(0x1000_0001) ^ n as u64);
            let mut stim = vec![0.0f32; n];
            for rate in &mut stim {
                *rate = rng.next_rate_f32();
            }
            pin_poisson_boundary_stimuli(&mut stim);
            let expected = poisson_encode_oracle(&stim, seed as u32);
            let d_stim = GpuBuffer::from_slice(&stim).unwrap();
            let mut d_spikes = GpuBuffer::<u32>::alloc(n).unwrap();
            acc.poisson_encode(&d_stim, &mut d_spikes, seed as u32)
                .expect("poisson_encode seeded");
            let got = d_spikes.to_vec().unwrap();
            assert_exact(&got, &expected, seed, &format!("n={n}"));
        }
    }
}

// ── satsolver extract / aux reduce ──────────────────────────────────────────

#[test]
#[ignore] // requires GPU + driver ≥ 570
fn satsolver_extract_matches_oracle_golden() {
    let acc = require_gpu();
    let assignment = vec![1u8, 0, 1, 0, 0, 0];
    let d_asgn = GpuBuffer::from_slice(&assignment).unwrap();
    let d_bw = GpuBuffer::from_slice(&[1i32]).unwrap();
    let mut d_out = GpuBuffer::<u8>::alloc(3).unwrap();
    acc.satsolver_extract(&d_asgn, &d_bw, &mut d_out, 3, 2)
        .expect("extract");
    let got = d_out.to_vec().unwrap();
    let expected = satsolver_extract_oracle(&assignment, 1, 3, 2);
    assert_exact(&got, &expected, 0, "n_vars=3 n_walkers=2 bw=1");
}

#[test]
#[ignore] // requires GPU + driver ≥ 570
fn satsolver_extract_out_of_range_walker_falls_back() {
    let acc = require_gpu();
    let assignment = vec![1u8, 1, 0, 0];
    let d_asgn = GpuBuffer::from_slice(&assignment).unwrap();
    let d_bw = GpuBuffer::from_slice(&[-3i32]).unwrap();
    let mut d_out = GpuBuffer::<u8>::alloc(2).unwrap();
    acc.satsolver_extract(&d_asgn, &d_bw, &mut d_out, 2, 2)
        .expect("extract fallback");
    let got = d_out.to_vec().unwrap();
    let expected = satsolver_extract_oracle(&assignment, -3, 2, 2);
    assert_exact(&got, &expected, 0, "n_vars=2 n_walkers=2 bw=-3");
}

#[test]
#[ignore] // requires GPU + driver ≥ 570
fn satsolver_aux_reduce_best_matches_oracle_golden() {
    let acc = require_gpu();
    let assignment = vec![1u8, 0, 1, 0, 0, 0];
    let clauses = vec![0i32, 2, 1, 4];
    let scores = vec![0i32, 1];
    let expected = satsolver_aux_reduce_best_oracle(&assignment, &scores, &clauses, 2, 3, 2, 2);

    let d_asgn = GpuBuffer::from_slice(&assignment).unwrap();
    let mut d_flags = GpuBuffer::<u8>::alloc(4).unwrap();
    let d_scores = GpuBuffer::from_slice(&scores).unwrap();
    let mut d_best_score = GpuBuffer::from_slice(&[i32::MAX]).unwrap();
    let mut d_best_walker = GpuBuffer::from_slice(&[-1i32]).unwrap();
    let d_clauses = GpuBuffer::from_slice(&clauses).unwrap();

    acc.satsolver_aux_reduce_best(
        &d_asgn,
        &mut d_flags,
        &d_scores,
        &mut d_best_score,
        &mut d_best_walker,
        &d_clauses,
        2,
        3,
        2,
        2,
    )
    .expect("aux_reduce_best");

    let flags = d_flags.to_vec().unwrap();
    let best_score = d_best_score.to_vec().unwrap();
    let best_walker = d_best_walker.to_vec().unwrap();
    assert_exact(&flags, &expected.sat_flags, 0, "n_walkers=2 n_clauses=2");
    assert_exact(&best_score, &[expected.best_score], 0, "best_score");
    assert_exact(&best_walker, &[expected.best_walker], 0, "best_walker");
}

#[test]
#[ignore] // requires GPU + driver ≥ 570
fn satsolver_aux_reduce_best_matches_oracle_seeded() {
    let acc = require_gpu();
    for &seed in CASE_SEEDS {
        for &(n_walkers, n_vars, n_clauses, clause_len) in
            &[(1, 1, 1, 1), (3, 7, 5, 2), (33, 4, 8, 3), (257, 2, 1, 1)]
        {
            let mut rng = CaseRng::new(seed ^ n_walkers as u64);
            let mut assignment = vec![0u8; n_walkers * n_vars];
            for slot in &mut assignment {
                *slot = rng.next_bit();
            }
            let scores = fill_sat_scores(&mut rng, n_walkers);
            let mut clauses = vec![0i32; n_clauses * clause_len];
            for c in 0..n_clauses {
                for l in 0..clause_len {
                    let var = (rng.next_u32() as usize) % n_vars.max(1);
                    let neg = rng.next_bit() as i32;
                    clauses[c * clause_len + l] = ((var as i32) << 1) | neg;
                }
            }
            let expected = satsolver_aux_reduce_best_oracle(
                &assignment,
                &scores,
                &clauses,
                n_walkers,
                n_vars,
                n_clauses,
                clause_len,
            );
            let shape = format!(
                "n_walkers={n_walkers} n_vars={n_vars} n_clauses={n_clauses} clause_len={clause_len}"
            );

            let d_asgn = GpuBuffer::from_slice(&assignment).unwrap();
            let mut d_flags = GpuBuffer::<u8>::alloc(n_walkers * n_clauses).unwrap();
            let d_scores = GpuBuffer::from_slice(&scores).unwrap();
            let mut d_best_score = GpuBuffer::from_slice(&[i32::MAX]).unwrap();
            let mut d_best_walker = GpuBuffer::from_slice(&[-1i32]).unwrap();
            let d_clauses = GpuBuffer::from_slice(&clauses).unwrap();

            acc.satsolver_aux_reduce_best(
                &d_asgn,
                &mut d_flags,
                &d_scores,
                &mut d_best_score,
                &mut d_best_walker,
                &d_clauses,
                n_walkers as i32,
                n_vars as i32,
                n_clauses as i32,
                clause_len as i32,
            )
            .expect("aux_reduce_best seeded");

            let flags = d_flags.to_vec().unwrap();
            let best_score = d_best_score.to_vec().unwrap()[0];
            let best_walker = d_best_walker.to_vec().unwrap()[0];
            assert_exact(&flags, &expected.sat_flags, seed, &shape);
            assert_exact(&[best_score], &[expected.best_score], seed, &shape);
            assert_exact(&[best_walker], &[expected.best_walker], seed, &shape);

            let mut d_out = GpuBuffer::<u8>::alloc(n_vars).unwrap();
            let d_bw = GpuBuffer::from_slice(&[best_walker]).unwrap();
            acc.satsolver_extract(&d_asgn, &d_bw, &mut d_out, n_vars as i32, n_walkers as i32)
                .expect("extract seeded");
            let extracted = d_out.to_vec().unwrap();
            let expected_asgn =
                satsolver_extract_oracle(&assignment, best_walker, n_vars, n_walkers);
            assert_exact(&extracted, &expected_asgn, seed, &shape);
        }
    }
}

// ── packed ternary ──────────────────────────────────────────────────────────

#[test]
#[ignore] // requires GPU + driver ≥ 570
fn ternary_gemv_matches_oracle_golden() {
    let acc = require_gpu();
    let w: Vec<i8> = vec![1, 0, -1, 1, 0, 1, 0, -1];
    let packed = pack_ternary_matrix(&w, 2, 4);
    let scales = uniform_group_scales(2, 4, 4, 2.0);
    let x = vec![1.0f32, 2.0, 3.0, 4.0];
    let expected = ternary_gemv_oracle(&packed, &scales, &x, 2, 4, 4, false);

    let d_w = GpuBuffer::from_slice(&packed).unwrap();
    let d_s = GpuBuffer::from_slice(&scales).unwrap();
    let d_x = GpuBuffer::from_slice(&x).unwrap();
    let mut d_y = GpuBuffer::<f32>::alloc(2).unwrap();
    acc.ternary_gemv(&d_w, &d_s, &d_x, &mut d_y, 2, 4, 4, false)
        .expect("ternary_gemv");
    let got = d_y.to_vec().unwrap();
    assert_f32(
        &got,
        &expected,
        TERNARY_ABS_TOL,
        TERNARY_REL_TOL,
        0,
        "m=2 k=4 group=4",
    );
}

#[test]
#[ignore] // requires GPU + driver ≥ 570
fn ternary_kernels_match_oracle_seeded_boundaries() {
    let acc = require_gpu();
    for &seed in CASE_SEEDS {
        for &(m, k, n, group, skip) in &[
            (1usize, 1usize, 1usize, 1usize, false),
            (8, 20, 3, 8, false),
            (3, 17, 1, 16, true),
            (5, 33, 4, 7, true),
            (1, 257, 2, 32, false),
        ] {
            let mut rng = CaseRng::new(seed ^ ((m as u64) << 24) ^ (k as u64));
            let mut weights = vec![0i8; m * k];
            for w in &mut weights {
                *w = rng.next_trit();
            }
            let packed = pack_ternary_matrix(&weights, m, k);
            let gpr = k.div_ceil(group);
            let mut scales = vec![0.0f32; m * gpr];
            for s in &mut scales {
                *s = rng.next_unit_f32() * 0.5 + 0.75;
            }
            let mut x = vec![0.0f32; k];
            for (i, slot) in x.iter_mut().enumerate() {
                *slot = rng.next_unit_f32();
                if i == 0 {
                    *slot = 0.0;
                }
                if i == 1 && k > 1 {
                    *slot = -0.0;
                }
            }
            let expected_y = ternary_gemv_oracle(&packed, &scales, &x, m, k, group, skip);
            let shape = format!("m={m} k={k} n={n} group={group} skip={skip}");

            let d_w = GpuBuffer::from_slice(&packed).unwrap();
            let d_s = GpuBuffer::from_slice(&scales).unwrap();
            let d_x = GpuBuffer::from_slice(&x).unwrap();
            let mut d_y = GpuBuffer::<f32>::alloc(m).unwrap();
            acc.ternary_gemv(
                &d_w,
                &d_s,
                &d_x,
                &mut d_y,
                m as i32,
                k as i32,
                group as i32,
                skip,
            )
            .expect("ternary_gemv seeded");
            let got_y = d_y.to_vec().unwrap();
            assert_f32(
                &got_y,
                &expected_y,
                TERNARY_ABS_TOL,
                TERNARY_REL_TOL,
                seed,
                &shape,
            );

            let mut b = vec![0.0f32; k * n];
            for slot in &mut b {
                *slot = rng.next_unit_f32();
            }
            let expected_c = ternary_gemm_oracle(&packed, &scales, &b, m, k, n, group, skip);
            let d_b = GpuBuffer::from_slice(&b).unwrap();
            let mut d_c = GpuBuffer::<f32>::alloc(m * n).unwrap();
            acc.ternary_gemm(
                &d_w,
                &d_s,
                &d_b,
                &mut d_c,
                m as i32,
                k as i32,
                n as i32,
                group as i32,
                skip,
            )
            .expect("ternary_gemm seeded");
            let got_c = d_c.to_vec().unwrap();
            assert_f32(
                &got_c,
                &expected_c,
                TERNARY_ABS_TOL,
                TERNARY_REL_TOL,
                seed,
                &shape,
            );
        }
    }
}

#[test]
#[ignore] // requires GPU + driver ≥ 570
fn ternary_empty_k_matches_oracle_zeros() {
    let acc = require_gpu();
    let m = 4usize;
    let expected = ternary_gemv_oracle(&[], &[], &[], m, 0, 1, false);
    let packed = GpuBuffer::<u32>::alloc(0).unwrap();
    let scales = GpuBuffer::<f32>::alloc(0).unwrap();
    let x = GpuBuffer::<f32>::alloc(0).unwrap();
    let mut y = GpuBuffer::from_slice(&vec![7.0f32; m]).unwrap();
    acc.ternary_gemv(&packed, &scales, &x, &mut y, m as i32, 0, 1, false)
        .expect("empty-K gemv");
    let got = y.to_vec().unwrap();
    assert_f32(&got, &expected, 0.0, 0.0, 0, "m=4 k=0");

    let n = 2usize;
    let expected_c = ternary_gemm_oracle(&[], &[], &[], m, 0, n, 1, false);
    let b = GpuBuffer::<f32>::alloc(0).unwrap();
    let mut c = GpuBuffer::from_slice(&vec![9.0f32; m * n]).unwrap();
    acc.ternary_gemm(
        &packed, &scales, &b, &mut c, m as i32, 0, n as i32, 1, false,
    )
    .expect("empty-K gemm");
    let got_c = c.to_vec().unwrap();
    assert_f32(&got_c, &expected_c, 0.0, 0.0, 0, "m=4 k=0 n=2");
}
