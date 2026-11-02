// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

//! GH #47 / LIM-1465: compare the two real CUDA score matrices, then select
//! expected indices from the batched output (never from a copied formula).
#![cfg(feature = "cuda")]

use cust::launch;
use cust::stream::{Stream, StreamFlags};
use myelin_accelerator::{GpuBuffer, GpuContext, KernelModule};

struct CosineCase<'a> {
    queries: &'a [f32],
    keys: &'a [f32],
    dim: usize,
}

impl<'a> CosineCase<'a> {
    fn new(queries: &'a [f32], keys: &'a [f32], dim: usize) -> Self {
        Self { queries, keys, dim }
    }

    fn compare(&self, top_k: usize, block: u32) -> (Vec<f32>, Vec<i32>) {
        let Self { queries, keys, dim } = *self;
        let nq = queries.len() / dim;
        let nk = keys.len() / dim;
        assert_eq!(queries.len(), nq * dim);
        assert_eq!(keys.len(), nk * dim);
        let _ctx = GpuContext::init().unwrap();
        let kernels = KernelModule::load().unwrap();
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();
        let queries = GpuBuffer::from_slice(queries).unwrap();
        let keys = GpuBuffer::from_slice(keys).unwrap();
        // NaNs / -99 detect unwritten elements as well as incorrect values.
        let batched = GpuBuffer::from_slice(&vec![f32::NAN; nq * nk]).unwrap();
        let selected = GpuBuffer::from_slice(&vec![f32::NAN; nq * nk]).unwrap();
        let indices = GpuBuffer::from_slice(&vec![-99i32; nq * top_k]).unwrap();
        let batched_fn = kernels.get_function("cosine_similarity_batched").unwrap();
        let topk_fn = kernels.get_function("cosine_similarity_top_k").unwrap();
        // SAFETY: exact ABI, distinct adequately sized buffers, full warps, and
        // synchronization before reading/dropping buffers, stream, module, context.
        unsafe {
            launch!(batched_fn<<<(nk as u32, nq as u32, 1), block, 0, stream>>>(
                queries.as_device_ptr(), keys.as_device_ptr(), batched.as_device_ptr(),
                nq as i32, nk as i32, dim as i32
            ))
            .unwrap();
            launch!(topk_fn<<<nq as u32, block, 0, stream>>>(
                queries.as_device_ptr(), keys.as_device_ptr(), selected.as_device_ptr(),
                indices.as_device_ptr(), nq as i32, nk as i32, dim as i32, top_k as i32
            ))
            .unwrap();
        }
        stream.synchronize().unwrap();
        let batched = batched.to_vec().unwrap();
        let selected = selected.to_vec().unwrap();
        let indices = indices.to_vec().unwrap();
        let shape = format!("q={nq} keys={nk} dim={dim} K={top_k} block={block}");
        let mut max_error = 0.0f32;
        for (i, (&a, &b)) in batched.iter().zip(&selected).enumerate() {
            assert!(a.is_finite() && b.is_finite(), "{shape} pair={i}: {a}, {b}");
            max_error = max_error.max((a - b).abs());
            assert_eq!(a.to_bits(), b.to_bits(), "{shape} pair={i}: {a}, {b}");
        }
        for (q, row) in batched.chunks_exact(nk).enumerate() {
            let mut expected: Vec<usize> = (0..nk).collect();
            expected.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap().then(a.cmp(&b)));
            let count = top_k.min(nk).min(32);
            let mut expected: Vec<i32> = expected[..count].iter().map(|&k| k as i32).collect();
            expected.resize(top_k, -1);
            assert_eq!(
                &indices[q * top_k..(q + 1) * top_k],
                expected,
                "{shape} query={q}"
            );
        }
        eprintln!("{shape}: full matrix max_abs_error={max_error:e}; exact indices passed");
        (batched, indices)
    }

    fn assert_repeated_ties(&self, k: usize, block: u32) {
        let expected: Vec<i32> = (0..k.min(32) as i32)
            .chain(std::iter::repeat_n(-1, k.saturating_sub(32)))
            .collect();
        for _ in 0..3 {
            let (_, indices) = self.compare(k, block);
            for row in indices.chunks_exact(k) {
                assert_eq!(row, expected);
            }
        }
    }

    fn assert_ties_for_launch_shapes(&self) {
        for block in [32, 64, 256, 1024] {
            for k in [1, 7, 32, 35] {
                self.assert_repeated_ties(k, block);
            }
        }
    }

    fn assert_block_independent(&self, k: usize) {
        let reference = self.compare(k, 32);
        for block in [64, 128, 256, 1024] {
            let actual = self.compare(k, block);
            assert_eq!(reference, actual, "block-dependent pair arithmetic");
        }
    }

    fn assert_scalar_oracle(&self) {
        use myelin_accelerator::oracle::{
            COSINE_ABS_TOL, COSINE_REL_TOL, assert_f32, cosine_similarity_batched_oracle,
        };
        let oracle = cosine_similarity_batched_oracle(
            self.queries,
            self.keys,
            self.queries.len() / self.dim,
            self.keys.len() / self.dim,
            self.dim,
        );
        for block in [32, 256] {
            for k in [1, 7, 32] {
                let (scores, _) = self.compare(k, block);
                assert_f32(
                    &scores,
                    &oracle,
                    COSINE_ABS_TOL,
                    COSINE_REL_TOL,
                    47,
                    &format!("dense cosine oracle dim={} block={block}", self.dim),
                );
            }
        }
    }
}

fn signed_samples(seed: &mut u32, count: usize) -> Vec<f32> {
    (0..count)
        .map(|_| {
            *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((*seed >> 8) as f32 / 16_777_216.0) * 2.0 - 1.0
        })
        .collect()
}

fn perturbed_keys(base: &[f32]) -> Vec<f32> {
    let dim = base.len();
    let mut keys = base.repeat(67);
    for k in 0..67 {
        let i = k * dim + k % dim;
        keys[i] = f32::from_bits(keys[i].to_bits() + k as u32);
    }
    keys
}

#[test]
#[ignore] // requires GPU + driver >= 570
fn cosine_astra_ranking_reversal() {
    for block in [32, 256] {
        for k in [1, 2] {
            let (scores, indices) =
                CosineCase::new(&[0.0001, 0.0, 0.0], &[0.0001, 0.0, 0.0, 0.8, 0.6, 0.0], 3)
                    .compare(k, block);
            assert!((scores[0] - 0.5).abs() <= f32::EPSILON);
            assert!((scores[1] - 0.7999201).abs() <= f32::EPSILON);
            assert_eq!(indices[0], 1);
            eprintln!("Astra: scores={scores:?}, indices={indices:?}");
        }
    }
}

#[test]
#[ignore] // requires GPU + driver >= 570
fn cosine_zero_tiny_normalized_and_negative_multiple_queries() {
    // Every query/key cross product includes zero query, zero key, both zero,
    // tiny *positive normal* squared norms, and underflow/FTZ squared norms.
    let rows = [
        [0.0, 0.0, 0.0],
        [1.0e-10, 0.0, 0.0],
        [1.0e-20, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
    .concat();
    for k in [1, 4, 8] {
        let (scores, _) = CosineCase::new(&rows, &rows, 3).compare(k, 128);
        assert!(scores[..6].iter().all(|&v| v == 0.0));
        assert!(scores.iter().step_by(6).all(|&v| v == 0.0));
        assert!((scores[7] - 1.0e-12).abs() < 1.0e-18);
        assert_eq!(scores[14], 0.0); // squared norm flushed to zero
        assert_eq!(scores[21], 1.0);
        assert_eq!(scores[22], -1.0);
        assert_eq!(scores[23], 0.0);
    }
}

#[test]
#[ignore] // requires GPU + driver >= 570
fn cosine_equal_scores_select_lower_indices_deterministically() {
    // Duplicates span lanes/warps and repeated ownership in both score paths.
    for dim in [3, 33] {
        let mut key = vec![0.0f32; dim];
        key[..2].copy_from_slice(&[0.8, 0.6]);
        let keys = key.repeat(1031);
        let mut queries = vec![0.0f32; 3 * dim];
        queries[0] = 1.0;
        queries[dim] = -1.0;
        CosineCase::new(&queries, &keys, dim).assert_ties_for_launch_shapes();
    }
}

#[test]
#[ignore] // requires GPU + driver >= 570
fn cosine_adjacent_tiny_keys_preserve_exact_ranking() {
    // Neighbouring finite tiny keys produce scores separated by less than
    // eight f32 epsilons. Still require exact scores AND indices.
    let tiny = 0.0001f32;
    let larger = f32::from_bits(tiny.to_bits() + 2);
    let keys = [tiny, 0.0, 0.0, larger, 0.0, 0.0];
    for k in [1, 2] {
        let (scores, indices) = CosineCase::new(&[tiny, 0.0, 0.0], &keys, 3).compare(k, 256);
        let gap = scores[1] - scores[0];
        assert!(gap > 0.0 && gap < 8.0 * f32::EPSILON, "gap={gap}");
        assert_eq!(indices[0], 1);
    }
}

#[test]
#[ignore] // requires GPU + driver >= 570
fn cosine_dense_strided_reductions_preserve_batched_selection() {
    // Deterministic signed inputs exercise FMA cancellation and both reduction
    // levels. Pair equivalence is checked directly; the existing scalar oracle
    // additionally anchors the unchanged normalization equation.
    let mut seed = 47u32;
    for dim in [17, 31, 32, 33, 257, 1025] {
        let queries = signed_samples(&mut seed, 3 * dim);
        let keys = signed_samples(&mut seed, 67 * dim);
        CosineCase::new(&queries, &keys, dim).assert_scalar_oracle();
    }
}

#[test]
#[ignore] // requires GPU + driver >= 570
fn cosine_dense_near_ties_preserve_batched_ranking() {
    let mut seed = 47u32;
    for dim in [17, 31, 32, 33, 257, 1025] {
        let query = signed_samples(&mut seed, dim);
        let keys = perturbed_keys(&signed_samples(&mut seed, dim));
        let case = CosineCase::new(&query, &keys, dim);
        for k in [1, 7, 32] {
            case.assert_block_independent(k);
        }
    }
}

#[test]
#[ignore] // requires GPU + driver >= 570
fn cosine_underflowed_norm_does_not_discard_finite_negative_scores() {
    let (scores, indices) = CosineCase::new(
        &[1.0e-20, 0.0, 0.0],
        &[-1.0e18, 0.0, 0.0, -2.0e18, 0.0, 0.0],
        3,
    )
    .compare(2, 256);
    assert!(scores[0] < -2.0 && scores[1] < scores[0]);
    assert_eq!(indices, [0, 1]);
}
