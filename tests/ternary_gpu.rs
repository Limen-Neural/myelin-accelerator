// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

//! GPU goldens for packed ternary GEMV/GEMM (requires CUDA + sm_120 driver).

#![cfg(feature = "cuda")]

use myelin_accelerator::bitpacking::{
    DEFAULT_GROUP_SIZE, groups_per_row, pack_ternary, pack_ternary_matrix, packed_words_for_matrix,
    ternary_gemm_ref, ternary_gemv_ref, uniform_group_scales,
};
use myelin_accelerator::{GpuAccelerator, GpuBuffer};

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn ternary_pattern(n: usize) -> Vec<i8> {
    (0..n)
        .map(|i| match i % 3 {
            0 => 0i8,
            1 => 1i8,
            _ => -1i8,
        })
        .collect()
}

/// Non-uniform per-group scales so device scale strides cannot be faked by a constant.
fn varying_group_scales(m: usize, k: usize, group_size: usize) -> Vec<f32> {
    let gpr = groups_per_row(k, group_size);
    let mut scales = vec![0.0f32; m * gpr];
    for row in 0..m {
        for g in 0..gpr {
            scales[row * gpr + g] = 1.0 + g as f32 + 0.1 * row as f32;
        }
    }
    scales
}

#[test]
#[ignore] // requires GPU + driver ≥ 570
fn ternary_gemv_matches_host_ref() {
    let acc = GpuAccelerator::new();
    assert!(acc.is_ready(), "GPU not ready for ternary_gemv golden");

    let m = 64usize;
    let k = 256usize;
    let group_size = DEFAULT_GROUP_SIZE;
    let weights = ternary_pattern(m * k);
    let packed = pack_ternary_matrix(&weights, m, k);
    let scales = uniform_group_scales(m, k, group_size, 0.5);
    let x: Vec<f32> = (0..k).map(|i| (i % 7) as f32 * 0.1).collect();
    let expected = ternary_gemv_ref(&packed, &scales, &x, m, k, group_size, false);

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
        group_size as i32,
        false,
    )
    .expect("ternary_gemv launch");

    let got = d_y.to_vec().unwrap();
    let err = max_abs_diff(&got, &expected);
    assert!(
        err < 1e-4,
        "ternary_gemv max abs err {err} (got vs host ref)"
    );
}

#[test]
#[ignore] // requires GPU + driver ≥ 570
fn ternary_gemv_skip_zeros_matches_host_ref() {
    let acc = GpuAccelerator::new();
    assert!(acc.is_ready(), "GPU not ready");

    let m = 32usize;
    let k = 128usize;
    let group_size = 64usize;
    let weights = ternary_pattern(m * k);
    let packed = pack_ternary_matrix(&weights, m, k);
    let scales = uniform_group_scales(m, k, group_size, 1.0);
    let x: Vec<f32> = (0..k).map(|i| 1.0 + (i % 5) as f32).collect();
    let expected = ternary_gemv_ref(&packed, &scales, &x, m, k, group_size, true);

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
        group_size as i32,
        true,
    )
    .expect("ternary_gemv skip_zeros");

    let got = d_y.to_vec().unwrap();
    assert!(max_abs_diff(&got, &expected) < 1e-4);
}

#[test]
#[ignore] // requires GPU + driver ≥ 570
fn ternary_gemm_matches_host_ref() {
    let acc = GpuAccelerator::new();
    assert!(acc.is_ready(), "GPU not ready for ternary_gemm golden");

    let m = 16usize;
    let k = 64usize;
    let n = 8usize;
    let group_size = 32usize;
    let weights = ternary_pattern(m * k);
    let packed = pack_ternary_matrix(&weights, m, k);
    let scales = uniform_group_scales(m, k, group_size, 0.25);
    let b: Vec<f32> = (0..k * n).map(|i| ((i % 11) as f32 - 5.0) * 0.05).collect();
    let expected = ternary_gemm_ref(&packed, &scales, &b, m, k, n, group_size, false);

    let d_w = GpuBuffer::from_slice(&packed).unwrap();
    let d_s = GpuBuffer::from_slice(&scales).unwrap();
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
        group_size as i32,
        false,
    )
    .expect("ternary_gemm launch");

    let got = d_c.to_vec().unwrap();
    let err = max_abs_diff(&got, &expected);
    assert!(
        err < 1e-4,
        "ternary_gemm max abs err {err} (got vs host ref)"
    );
}

#[test]
#[ignore] // requires GPU + driver ≥ 570
fn ternary_gemm_skip_zeros_matches_host_ref() {
    let acc = GpuAccelerator::new();
    assert!(acc.is_ready(), "GPU not ready");

    let m = 12usize;
    let k = 48usize;
    let n = 4usize;
    let group_size = 16usize;
    let weights = ternary_pattern(m * k);
    let packed = pack_ternary_matrix(&weights, m, k);
    let scales = uniform_group_scales(m, k, group_size, 0.75);
    let b: Vec<f32> = (0..k * n).map(|i| ((i % 9) as f32 - 4.0) * 0.1).collect();
    let expected = ternary_gemm_ref(&packed, &scales, &b, m, k, n, group_size, true);

    let d_w = GpuBuffer::from_slice(&packed).unwrap();
    let d_s = GpuBuffer::from_slice(&scales).unwrap();
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
        group_size as i32,
        true,
    )
    .expect("ternary_gemm skip_zeros");

    let got = d_c.to_vec().unwrap();
    assert!(max_abs_diff(&got, &expected) < 1e-4);
}

#[test]
#[ignore] // requires GPU + driver ≥ 570
fn ternary_kernels_registered_in_module() {
    let acc = GpuAccelerator::new();
    assert!(acc.is_ready());
    let k = acc.kernels().expect("kernels");
    assert!(k.get_function("ternary_gemv").is_ok());
    assert!(k.get_function("ternary_gemm").is_ok());
}

#[test]
#[ignore] // requires GPU + driver ≥ 570
fn ternary_gemv_nonuniform_scales_and_k_not_multiple_of_16() {
    let acc = GpuAccelerator::new();
    assert!(acc.is_ready(), "GPU not ready");

    // K=20 is not a multiple of 16 → row-padded packing differs from flat pack.
    let m = 8usize;
    let k = 20usize;
    let group_size = 8usize; // also K % group_size != 0 (partial last group)
    assert_ne!(k % 16, 0);
    assert_ne!(k % group_size, 0);

    let weights = ternary_pattern(m * k);
    let packed = pack_ternary_matrix(&weights, m, k);
    let flat = pack_ternary(&weights);
    assert_eq!(packed.len(), packed_words_for_matrix(m, k));
    assert_ne!(
        packed.len(),
        flat.len(),
        "matrix packing must row-pad when K % 16 != 0"
    );

    let scales = varying_group_scales(m, k, group_size);
    let x: Vec<f32> = (0..k).map(|i| 0.25 + (i % 4) as f32 * 0.1).collect();
    let expected = ternary_gemv_ref(&packed, &scales, &x, m, k, group_size, false);

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
        group_size as i32,
        false,
    )
    .expect("ternary_gemv nonuniform/pad");

    let got = d_y.to_vec().unwrap();
    let err = max_abs_diff(&got, &expected);
    assert!(
        err < 1e-4,
        "nonuniform+pad gemv max abs err {err} (got vs host ref)"
    );
}

#[test]
#[ignore] // requires GPU + driver ≥ 570
fn ternary_gemm_nonuniform_scales_and_k_not_multiple_of_16() {
    let acc = GpuAccelerator::new();
    assert!(acc.is_ready(), "GPU not ready");

    let m = 6usize;
    let k = 20usize;
    let n = 3usize;
    let group_size = 12usize;
    assert_ne!(k % 16, 0);

    let weights = ternary_pattern(m * k);
    let packed = pack_ternary_matrix(&weights, m, k);
    let scales = varying_group_scales(m, k, group_size);
    let b: Vec<f32> = (0..k * n).map(|i| ((i % 7) as f32 - 3.0) * 0.05).collect();
    let expected = ternary_gemm_ref(&packed, &scales, &b, m, k, n, group_size, false);

    let d_w = GpuBuffer::from_slice(&packed).unwrap();
    let d_s = GpuBuffer::from_slice(&scales).unwrap();
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
        group_size as i32,
        false,
    )
    .expect("ternary_gemm nonuniform/pad");

    let got = d_c.to_vec().unwrap();
    assert!(max_abs_diff(&got, &expected) < 1e-4);
}

#[test]
#[ignore] // requires GPU + driver ≥ 570
fn ternary_gemv_empty_k_zeros_output() {
    let acc = GpuAccelerator::new();
    assert!(acc.is_ready());

    let m = 4usize;
    let packed = GpuBuffer::<u32>::alloc(0).unwrap();
    let scales = GpuBuffer::<f32>::alloc(0).unwrap();
    let x = GpuBuffer::<f32>::alloc(0).unwrap();
    // Poison y so a no-op early return would fail the test.
    let mut y = GpuBuffer::from_slice(&vec![7.0f32; m]).unwrap();

    acc.ternary_gemv(&packed, &scales, &x, &mut y, m as i32, 0, 1, false)
        .expect("empty-K gemv");

    let got = y.to_vec().unwrap();
    assert!(
        got.iter().all(|&v| v == 0.0),
        "empty K must zero y, got {got:?}"
    );
}
