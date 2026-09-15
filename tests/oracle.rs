// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

//! CPU oracle suite: goldens, seeded cases, and mismatch reporting.
//!
//! Runs in CPU-only CI (`cargo test --locked`). GPU comparison lives in
//! `tests/oracle_gpu.rs` and is capability-gated.

use myelin_accelerator::bitpacking::{
    pack_ternary_matrix, ternary_gemm_ref, ternary_gemv_ref, uniform_group_scales,
};
use myelin_accelerator::oracle::{
    BOUNDARY_LENS, CASE_SEEDS, COSINE_ABS_TOL, COSINE_REL_TOL, CaseRng, SHIP_EPS, TERNARY_ABS_TOL,
    TERNARY_REL_TOL, assert_exact, assert_f32, check_exact, check_f32,
    cosine_similarity_batched_oracle, f32_close, lcg_next, lcg_unit, poisson_encode_oracle,
    satsolver_aux_reduce_best_oracle, satsolver_extract_oracle, ternary_gemm_oracle,
    ternary_gemv_oracle,
};

// ── LCG / encoding goldens ──────────────────────────────────────────────────

#[test]
fn lcg_golden_seed_zero() {
    // Hand-traced Numerical Recipes step from state 0.
    assert_eq!(lcg_next(0), 1_013_904_223);
    assert_eq!(lcg_next(1_013_904_223), 1_196_435_762);
    let r = lcg_unit(1_196_435_762);
    assert!((r - (4_673_577.0 / 16_777_216.0)).abs() < 1e-12);
    assert!((0.0..1.0).contains(&r));
}

#[test]
fn poisson_encode_golden_single_channel() {
    // seed=0, n=1: r ≈ 0.2786 after two LCG steps.
    let r = {
        let s1 = lcg_next(0);
        lcg_unit(lcg_next(s1))
    };
    assert_exact(
        &poisson_encode_oracle(&[1.0], 0),
        &[1],
        0,
        "n=1 stim=1 seed=0",
    );
    assert_exact(
        &poisson_encode_oracle(&[0.0], 0),
        &[0],
        0,
        "n=1 stim=0 seed=0",
    );
    assert_exact(
        &poisson_encode_oracle(&[0.2], 0),
        &[u32::from(r < 0.2)],
        0,
        "n=1 stim=0.2 seed=0",
    );
    assert_exact(
        &poisson_encode_oracle(&[0.3], 0),
        &[u32::from(r < 0.3)],
        0,
        "n=1 stim=0.3 seed=0",
    );
}

#[test]
fn poisson_encode_empty_and_clamp_and_signed_zero() {
    assert_exact(&poisson_encode_oracle(&[], 7), &[], 7, "n=0");
    // Rates outside [0,1] clamp; -0.0 is a zero threshold → never fires.
    let spikes = poisson_encode_oracle(&[-1.0, -0.0, 2.0, 1.0], 0);
    assert_eq!(spikes[0], 0, "negative rate clamps to 0");
    assert_eq!(spikes[1], 0, "signed zero does not fire");
    assert_eq!(spikes[2], poisson_encode_oracle(&[1.0], 2)[0]);
    assert_eq!(spikes[3], poisson_encode_oracle(&[1.0], 3)[0]);
}

#[test]
fn poisson_encode_boundary_lengths_and_seeds() {
    for &seed in CASE_SEEDS {
        for &n in BOUNDARY_LENS {
            let mut rng = CaseRng::new(seed.wrapping_mul(0x1000_0001) ^ n as u64);
            let mut stim = vec![0.0f32; n];
            for rate in &mut stim {
                *rate = rng.next_rate_f32();
            }
            if n > 0 {
                stim[0] = 0.0;
            }
            if n > 1 {
                stim[1] = 1.0;
            }
            if n > 2 {
                stim[2] = -0.0;
            }
            if n > 3 {
                stim[3] = 1.5;
            }
            let spikes = poisson_encode_oracle(&stim, seed as u32);
            assert_eq!(spikes.len(), n, "seed={seed} n={n} truncated spikes");
            for &s in &spikes {
                assert!(s == 0 || s == 1, "seed={seed} n={n} non-binary spike");
            }
            let shape = format!("n={n}");
            assert_exact(
                &spikes,
                &poisson_encode_oracle(&stim, seed as u32),
                seed,
                &shape,
            );
        }
    }
}

// ── Cosine routing goldens ──────────────────────────────────────────────────

#[test]
fn cosine_similarity_golden_orthogonal_and_identical() {
    let queries = vec![1.0f32, 0.0, 0.0, 1.0];
    let keys = vec![1.0f32, 0.0, 0.0, 1.0];
    let out = cosine_similarity_batched_oracle(&queries, &keys, 2, 2, 2);
    assert_eq!(out.len(), 4);
    let denom_same = 1.0f32 + SHIP_EPS;
    assert_f32(
        &[out[0], out[3]],
        &[1.0 / denom_same, 1.0 / denom_same],
        COSINE_ABS_TOL,
        COSINE_REL_TOL,
        0,
        "q=2 k=2 dim=2 identical",
    );
    assert_f32(
        &[out[1], out[2]],
        &[0.0 / denom_same, 0.0 / denom_same],
        COSINE_ABS_TOL,
        COSINE_REL_TOL,
        0,
        "q=2 k=2 dim=2 orthogonal",
    );
}

#[test]
fn cosine_similarity_empty_shapes() {
    let out = cosine_similarity_batched_oracle(&[], &[], 0, 0, 4);
    assert!(out.is_empty());
    let q = vec![1.0f32, 2.0];
    let out = cosine_similarity_batched_oracle(&q, &[], 1, 0, 2);
    assert!(out.is_empty(), "n_keys=0 must not hide a truncated row");
    let out = cosine_similarity_batched_oracle(&[], &[1.0, 2.0], 0, 1, 2);
    assert!(out.is_empty());
    // dim=0: 0 / eps
    let out = cosine_similarity_batched_oracle(&[], &[], 1, 1, 0);
    assert_f32(
        &out,
        &[0.0 / SHIP_EPS],
        COSINE_ABS_TOL,
        COSINE_REL_TOL,
        0,
        "dim=0",
    );
}

#[test]
fn cosine_similarity_seeded_and_non_multiples() {
    for &seed in CASE_SEEDS {
        for &(nq, nk, dim) in &[(1, 1, 1), (1, 3, 17), (3, 1, 31), (2, 5, 33)] {
            let mut rng =
                CaseRng::new(seed ^ ((nq as u64) << 16) ^ ((nk as u64) << 8) ^ dim as u64);
            let mut queries = vec![0.0f32; nq * dim];
            let mut keys = vec![0.0f32; nk * dim];
            for q in &mut queries {
                *q = rng.next_unit_f32();
            }
            for k in &mut keys {
                *k = rng.next_unit_f32();
            }
            if dim > 0 {
                queries[0] = 0.0;
                keys[0] = -0.0;
            }
            let out = cosine_similarity_batched_oracle(&queries, &keys, nq, nk, dim);
            let shape = format!("n_queries={nq} n_keys={nk} dim={dim}");
            assert_eq!(out.len(), nq * nk, "seed={seed} {shape} truncated");
            for &v in &out {
                assert!(v.is_finite(), "seed={seed} {shape} non-finite cosine {v}");
            }
            let again = cosine_similarity_batched_oracle(&queries, &keys, nq, nk, dim);
            assert_f32(&out, &again, 0.0, 0.0, seed, &shape);
        }
    }
}

// ── SAT extract / reduce goldens ────────────────────────────────────────────

fn sat_golden_formula() -> (Vec<u8>, Vec<i32>, Vec<i32>) {
    // (x0 ∨ x1) ∧ (¬x0 ∨ x2)
    let clauses = vec![0, 2, 1, 4];
    let assignment = vec![
        1u8, 0, 1, // walker 0: both clauses sat
        0, 0, 0, // walker 1: first clause unsat
    ];
    let scores = vec![0i32, 1];
    (assignment, scores, clauses)
}

#[test]
fn satsolver_extract_golden() {
    let (assignment, _, _) = sat_golden_formula();
    assert_exact(
        &satsolver_extract_oracle(&assignment, 0, 3, 2),
        &[1, 0, 1],
        0,
        "n_vars=3 n_walkers=2 bw=0",
    );
    assert_exact(
        &satsolver_extract_oracle(&assignment, 1, 3, 2),
        &[0, 0, 0],
        0,
        "n_vars=3 n_walkers=2 bw=1",
    );
    assert_exact(
        &satsolver_extract_oracle(&assignment, -1, 3, 2),
        &[1, 0, 1],
        0,
        "n_vars=3 n_walkers=2 bw=-1 fallback",
    );
    assert_exact(
        &satsolver_extract_oracle(&assignment, 99, 3, 2),
        &[1, 0, 1],
        0,
        "n_vars=3 n_walkers=2 bw=99 fallback",
    );
    assert_exact(
        &satsolver_extract_oracle(&assignment, 0, 0, 2),
        &[],
        0,
        "n_vars=0",
    );
}

#[test]
fn satsolver_aux_reduce_best_golden() {
    let (assignment, scores, clauses) = sat_golden_formula();
    let got = satsolver_aux_reduce_best_oracle(&assignment, &scores, &clauses, 2, 3, 2, 2);
    assert_exact(&got.sat_flags, &[1, 1, 0, 1], 0, "n_walkers=2 n_clauses=2");
    assert_eq!(got.best_score, 0);
    assert_eq!(got.best_walker, 0);

    // Scores are the reduction input, not recomputed from assignments.
    let scores_tie = vec![7i32, 7];
    let tied = satsolver_aux_reduce_best_oracle(&assignment, &scores_tie, &clauses, 2, 3, 2, 2);
    assert_eq!(tied.best_score, 7);
    assert_eq!(tied.best_walker, 0, "equal scores: lower walker index wins");

    let scores_flip = vec![4i32, 1];
    let flipped = satsolver_aux_reduce_best_oracle(&assignment, &scores_flip, &clauses, 2, 3, 2, 2);
    assert_eq!(flipped.best_score, 1);
    assert_eq!(flipped.best_walker, 1);
}

#[test]
fn satsolver_aux_empty_clauses_and_zero_len() {
    let assignment = vec![1u8, 0, 1, 0];
    let scores = vec![3i32, 1];
    let got = satsolver_aux_reduce_best_oracle(&assignment, &scores, &[], 2, 2, 0, 0);
    assert!(got.sat_flags.is_empty());
    assert_eq!(got.best_score, 1);
    assert_eq!(got.best_walker, 1);

    // clause_len=0 → every clause unsatisfied
    let dummy_clauses = vec![];
    let got = satsolver_aux_reduce_best_oracle(&assignment, &scores, &dummy_clauses, 1, 2, 3, 0);
    assert_exact(
        &got.sat_flags,
        &[0, 0, 0],
        0,
        "n_walkers=1 n_clauses=3 clause_len=0",
    );
}

#[test]
fn satsolver_seeded_walkers_not_block_multiples() {
    for &seed in CASE_SEEDS {
        for &(n_walkers, n_vars, n_clauses, clause_len) in
            &[(1, 1, 1, 1), (3, 7, 5, 2), (33, 4, 8, 3), (257, 2, 1, 1)]
        {
            let mut rng = CaseRng::new(seed ^ n_walkers as u64);
            let mut assignment = vec![0u8; n_walkers * n_vars];
            for slot in &mut assignment {
                *slot = rng.next_bit();
            }
            let mut scores = vec![0i32; n_walkers];
            for (w, score) in scores.iter_mut().enumerate() {
                *score = (rng.next_u32() % 17) as i32;
                if w == n_walkers / 2 {
                    *score = i32::MIN / 4;
                }
            }
            let mut clauses = vec![0i32; n_clauses * clause_len];
            for c in 0..n_clauses {
                for l in 0..clause_len {
                    let var = (rng.next_u32() as usize) % n_vars.max(1);
                    let neg = rng.next_bit() as i32;
                    clauses[c * clause_len + l] = ((var as i32) << 1) | neg;
                }
            }
            let got = satsolver_aux_reduce_best_oracle(
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
            assert_eq!(
                got.sat_flags.len(),
                n_walkers * n_clauses,
                "seed={seed} {shape}"
            );
            let extracted =
                satsolver_extract_oracle(&assignment, got.best_walker, n_vars, n_walkers);
            assert_eq!(
                extracted.len(),
                n_vars,
                "seed={seed} {shape} extract truncated"
            );
            let again = satsolver_aux_reduce_best_oracle(
                &assignment,
                &scores,
                &clauses,
                n_walkers,
                n_vars,
                n_clauses,
                clause_len,
            );
            assert_eq!(got, again);
        }
    }
}

// ── Packed ternary goldens ──────────────────────────────────────────────────

#[test]
fn ternary_gemv_oracle_golden_2x4() {
    let w: Vec<i8> = vec![1, 0, -1, 1, 0, 1, 0, -1];
    let packed = pack_ternary_matrix(&w, 2, 4);
    let scales = uniform_group_scales(2, 4, 4, 2.0);
    let x = vec![1.0f32, 2.0, 3.0, 4.0];
    for skip in [false, true] {
        let y = ternary_gemv_oracle(&packed, &scales, &x, 2, 4, 4, skip);
        assert_f32(&y, &[4.0, -4.0], 1e-6, 0.0, 0, "m=2 k=4 group=4");
        let host = ternary_gemv_ref(&packed, &scales, &x, 2, 4, 4, skip);
        assert_f32(&y, &host, 0.0, 0.0, 0, "oracle vs bitpacking ref");
    }
}

#[test]
fn ternary_gemm_oracle_golden_matches_gemv_columns() {
    let w: Vec<i8> = vec![1, 0, -1, 1, 0, 1, 0, -1];
    let packed = pack_ternary_matrix(&w, 2, 4);
    let scales = uniform_group_scales(2, 4, 2, 1.5);
    let n = 3usize;
    let b: Vec<f32> = vec![1.0, 0.0, 2.0, 2.0, 1.0, 0.0, 3.0, 2.0, 1.0, 4.0, 3.0, 2.0];
    let c = ternary_gemm_oracle(&packed, &scales, &b, 2, 4, n, 2, true);
    for col in 0..n {
        let x: Vec<f32> = (0..4).map(|kk| b[kk * n + col]).collect();
        let y = ternary_gemv_oracle(&packed, &scales, &x, 2, 4, 2, false);
        assert_f32(
            &[c[col]],
            &[y[0]],
            1e-6,
            0.0,
            0,
            &format!("col={col} row=0"),
        );
        assert_f32(
            &[c[n + col]],
            &[y[1]],
            1e-6,
            0.0,
            0,
            &format!("col={col} row=1"),
        );
    }
    let host = ternary_gemm_ref(&packed, &scales, &b, 2, 4, n, 2, true);
    assert_f32(&c, &host, 0.0, 0.0, 0, "oracle vs bitpacking gemm ref");
}

#[test]
fn ternary_oracles_empty_k_zero_output() {
    let y = ternary_gemv_oracle(&[], &[], &[], 4, 0, 1, false);
    assert_f32(&y, &[0.0; 4], 0.0, 0.0, 0, "m=4 k=0");
    let c = ternary_gemm_oracle(&[], &[], &[], 3, 0, 2, 1, false);
    assert_f32(&c, &[0.0; 6], 0.0, 0.0, 0, "m=3 k=0 n=2");
    let y0 = ternary_gemv_oracle(&[], &[], &[1.0], 0, 1, 1, false);
    assert!(y0.is_empty());
}

#[test]
fn ternary_oracles_k_not_multiple_of_16_and_extreme_values() {
    for &seed in CASE_SEEDS {
        for &(m, k, n, group) in &[(1, 1, 1, 1), (8, 20, 3, 8), (3, 17, 1, 16), (5, 33, 4, 7)] {
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
            if k > 2 {
                x[2] = f32::MIN_POSITIVE;
            }
            let y = ternary_gemv_oracle(&packed, &scales, &x, m, k, group, false);
            let y_skip = ternary_gemv_oracle(&packed, &scales, &x, m, k, group, true);
            let shape = format!("m={m} k={k} group={group}");
            assert_eq!(y.len(), m, "seed={seed} {shape}");
            assert_f32(&y, &y_skip, TERNARY_ABS_TOL, TERNARY_REL_TOL, seed, &shape);
            let host = ternary_gemv_ref(&packed, &scales, &x, m, k, group, false);
            assert_f32(&y, &host, 0.0, 0.0, seed, &shape);

            let mut b = vec![0.0f32; k * n];
            for slot in &mut b {
                *slot = rng.next_unit_f32();
            }
            let c = ternary_gemm_oracle(&packed, &scales, &b, m, k, n, group, false);
            assert_eq!(c.len(), m * n, "seed={seed} m={m} k={k} n={n} truncated");
            let host_c = ternary_gemm_ref(&packed, &scales, &b, m, k, n, group, true);
            assert_f32(
                &c,
                &host_c,
                TERNARY_ABS_TOL,
                TERNARY_REL_TOL,
                seed,
                &format!("m={m} k={k} n={n} group={group}"),
            );
        }
    }
}

// ── Mismatch reporting ──────────────────────────────────────────────────────

#[test]
fn mismatch_reports_first_index_seed_and_shape() {
    let err = check_exact(&[1u32, 2, 9], &[1u32, 2, 3], 42, "n=3").unwrap_err();
    assert!(err.contains("index 2"), "{err}");
    assert!(err.contains("seed=42"), "{err}");
    assert!(err.contains("n=3"), "{err}");
    assert!(err.contains("expected 3"), "{err}");
    assert!(err.contains("got 9"), "{err}");
}

#[test]
fn length_mismatch_is_not_hidden_by_zip() {
    let err = check_exact(&[1u8, 2], &[1u8, 2, 3], 7, "n_vars=3").unwrap_err();
    assert!(err.contains("length mismatch"), "{err}");
    assert!(err.contains("truncated"), "{err}");
    assert!(err.contains("seed=7"), "{err}");
    assert!(err.contains("n_vars=3"), "{err}");
}

#[test]
fn nan_never_compares_equal() {
    assert!(!f32_close(f32::NAN, f32::NAN, 1.0, 1.0));
    assert!(!f32_close(f32::NAN, 0.0, 1.0, 1.0));
    let err = check_f32(&[f32::NAN], &[f32::NAN], 1.0, 1.0, 99, "m=1").unwrap_err();
    assert!(err.contains("index 0"), "{err}");
    assert!(err.contains("seed=99"), "{err}");
    assert!(f32_close(0.0, -0.0, 0.0, 0.0));
    assert!(f32_close(f32::INFINITY, f32::INFINITY, 0.0, 0.0));
    assert!(!f32_close(f32::INFINITY, f32::NEG_INFINITY, 1e9, 1.0));
}
