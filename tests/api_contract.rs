// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Integration tests validating the public API contract.
//!
//! Stub-specific tests (context init, accelerator, buffer, error) are gated
//! to `not(feature = "cuda")` since the real GPU backend has different
//! runtime behaviour. Bitpacking tests run unconditionally.

// ── Bitpacking contract (always runs) ──────────────────────────────────────

#[test]
fn bitpacking_binary_roundtrip() {
    use myelin_accelerator::bitpacking::{pack_binary, unpack_binary};

    let values: Vec<bool> = (0..100).map(|i| i % 3 == 0).collect();
    let packed = pack_binary(&values);
    let unpacked = unpack_binary(&packed, Some(values.len()));
    assert_eq!(values, unpacked);
}

#[test]
fn bitpacking_ternary_roundtrip() {
    use myelin_accelerator::bitpacking::{pack_ternary, unpack_ternary};

    let values: Vec<i8> = (0..100)
        .map(|i| match i % 3 {
            0 => 0i8,
            1 => 1i8,
            _ => -1i8,
        })
        .collect();
    let packed = pack_ternary(&values);
    let unpacked = unpack_ternary(&packed, Some(values.len()));
    assert_eq!(values, unpacked);
}

#[test]
fn bitpacking_goz1_bytes_match_u32_le_prefix() {
    use myelin_accelerator::bitpacking::{
        pack_ternary, pack_ternary_bytes, packed_u32_as_le_bytes, ternary_payload_byte_len,
        unpack_ternary_bytes,
    };

    for n in [0usize, 1, 4, 5, 16, 17, 50] {
        let values: Vec<i8> = (0..n)
            .map(|i| match i % 3 {
                0 => 0i8,
                1 => 1i8,
                _ => -1i8,
            })
            .collect();
        let bytes = pack_ternary_bytes(&values);
        let le = packed_u32_as_le_bytes(&pack_ternary(&values));
        assert_eq!(bytes.len(), ternary_payload_byte_len(n));
        assert_eq!(&bytes[..], &le[..ternary_payload_byte_len(n)]);
        assert_eq!(unpack_ternary_bytes(&bytes, Some(n)), values);
    }
}

#[test]
fn bitpacking_group_scales_length() {
    use myelin_accelerator::bitpacking::{
        DEFAULT_GROUP_SIZE, groups_per_row, scale_count, uniform_group_scales,
    };

    assert_eq!(DEFAULT_GROUP_SIZE, 128);
    let m = 4usize;
    let k = 300usize;
    let scales = uniform_group_scales(m, k, DEFAULT_GROUP_SIZE, 1.0);
    assert_eq!(scales.len(), scale_count(m, k, DEFAULT_GROUP_SIZE));
    assert_eq!(scales.len(), m * groups_per_row(k, DEFAULT_GROUP_SIZE));
}

#[test]
fn bitpacking_ternary_gemv_ref_known() {
    use myelin_accelerator::bitpacking::{
        pack_ternary_matrix, ternary_gemv_ref, uniform_group_scales,
    };

    // 2×4 known goldens (see unit tests / docs/TERNARY.md)
    let w: Vec<i8> = vec![1, 0, -1, 1, 0, 1, 0, -1];
    let packed = pack_ternary_matrix(&w, 2, 4);
    let scales = uniform_group_scales(2, 4, 4, 2.0);
    let x = vec![1.0f32, 2.0, 3.0, 4.0];
    let y = ternary_gemv_ref(&packed, &scales, &x, 2, 4, 4, false);
    assert!((y[0] - 4.0).abs() < 1e-5);
    assert!((y[1] - (-4.0)).abs() < 1e-5);
}

#[test]
fn bitpacking_ternary_gemm_ref_matches_gemv_columns() {
    use myelin_accelerator::bitpacking::{
        pack_ternary_matrix, ternary_gemm_ref, ternary_gemv_ref, uniform_group_scales,
    };

    let w: Vec<i8> = vec![1, 0, -1, 1, 0, 1, 0, -1];
    let packed = pack_ternary_matrix(&w, 2, 4);
    let scales = uniform_group_scales(2, 4, 2, 1.5);
    let n = 3usize;
    let b: Vec<f32> = vec![1.0, 0.0, 2.0, 2.0, 1.0, 0.0, 3.0, 2.0, 1.0, 4.0, 3.0, 2.0];
    let c = ternary_gemm_ref(&packed, &scales, &b, 2, 4, n, 2, true);
    for col in 0..n {
        let x: Vec<f32> = (0..4).map(|kk| b[kk * n + col]).collect();
        let y = ternary_gemv_ref(&packed, &scales, &x, 2, 4, 2, false);
        assert!((c[col] - y[0]).abs() < 1e-5);
        assert!((c[n + col] - y[1]).abs() < 1e-5);
    }
}

// ── Stub-specific tests (CPU-only, no real GPU) ─────────────────────────────

#[cfg(not(feature = "cuda"))]
mod stub_contract {
    use myelin_accelerator::{GpuAccelerator, GpuBuffer, GpuContext, GpuError, KernelModule};

    #[test]
    fn public_types_are_exported() {
        let _: GpuContext;
        let _: KernelModule;
        let _: GpuAccelerator;
    }

    #[test]
    fn context_init_returns_error_without_gpu() {
        let result = GpuContext::init();
        assert!(result.is_err());
    }

    #[test]
    fn context_is_available_false_without_gpu() {
        assert!(!GpuContext::is_available());
    }

    #[test]
    fn kernel_load_returns_error_without_gpu() {
        assert!(matches!(KernelModule::load(), Err(GpuError::NoGpu)));
    }

    #[test]
    fn buffer_alloc_and_roundtrip() {
        let data = vec![42u32; 128];
        let buf = GpuBuffer::from_slice(&data).expect("from_slice should succeed in stub");
        assert_eq!(buf.len(), 128);
        let out = buf.to_vec().expect("to_vec should succeed in stub");
        assert_eq!(out, data);
    }

    #[test]
    fn buffer_upload_matches_length() {
        let mut buf = GpuBuffer::<i32>::alloc(8).expect("alloc should succeed in stub");
        let data = vec![1i32, 2, 3, 4, 5, 6, 7, 8];
        buf.upload(&data).expect("upload should succeed");
        assert_eq!(buf.to_vec().unwrap(), data);
    }

    #[test]
    fn buffer_upload_rejects_mismatch() {
        let mut buf = GpuBuffer::<i32>::alloc(8).expect("alloc should succeed in stub");
        let result = buf.upload(&[1, 2, 3]);
        assert!(result.is_err());
    }

    #[test]
    fn accelerator_construction() {
        let acc = GpuAccelerator::new();
        assert!(!acc.is_ready());
        assert!(acc.kernels().is_err());
        assert!(acc.synchronize().is_err());
    }

    #[test]
    fn accelerator_all_kernel_launches_fail_gracefully() {
        let acc = GpuAccelerator::new();

        let a = GpuBuffer::<u8>::alloc(10).unwrap();
        let b = GpuBuffer::<i32>::alloc(1).unwrap();
        let mut c = GpuBuffer::<u8>::alloc(10).unwrap();
        assert!(acc.satsolver_extract(&a, &b, &mut c, 10, 1).is_err());

        let stim = GpuBuffer::<f32>::alloc(10).unwrap();
        let mut spikes = GpuBuffer::<u32>::alloc(10).unwrap();
        assert!(acc.poisson_encode(&stim, &mut spikes, 42).is_err());

        let w = GpuBuffer::<u32>::alloc(1).unwrap();
        let s = GpuBuffer::<f32>::alloc(1).unwrap();
        let x = GpuBuffer::<f32>::alloc(1).unwrap();
        let mut y = GpuBuffer::<f32>::alloc(1).unwrap();
        assert!(
            acc.ternary_gemv(&w, &s, &x, &mut y, 1, 1, 1, false)
                .is_err()
        );
        let b = GpuBuffer::<f32>::alloc(1).unwrap();
        let mut c = GpuBuffer::<f32>::alloc(1).unwrap();
        assert!(
            acc.ternary_gemm(&w, &s, &b, &mut c, 1, 1, 1, 1, false)
                .is_err()
        );
    }

    #[test]
    fn accelerator_satsolver_aux_reduce_best_returns_no_gpu() {
        let acc = GpuAccelerator::new();
        let assignment = GpuBuffer::<u8>::alloc(10).unwrap();
        let mut sat_flags = GpuBuffer::<u8>::alloc(10).unwrap();
        let scores = GpuBuffer::<i32>::alloc(1).unwrap();
        let mut best_score = GpuBuffer::<i32>::alloc(1).unwrap();
        let mut best_walker = GpuBuffer::<i32>::alloc(1).unwrap();
        let clauses = GpuBuffer::<i32>::alloc(10).unwrap();
        assert!(
            acc.satsolver_aux_reduce_best(
                &assignment,
                &mut sat_flags,
                &scores,
                &mut best_score,
                &mut best_walker,
                &clauses,
                1,
                10,
                1,
                10,
            )
            .is_err()
        );
    }

    #[test]
    fn error_display_contains_variant_info() {
        let variants = [
            GpuError::NoGpu,
            GpuError::InitFailed("test".into()),
            GpuError::ModuleLoadFailed("test".into()),
            GpuError::KernelNotFound("test".into()),
            GpuError::MemoryError("test".into()),
            GpuError::LaunchFailed("test".into()),
            GpuError::CudaError("test".into()),
        ];

        for err in &variants {
            let msg = err.to_string();
            assert!(!msg.is_empty(), "error Display should not be empty");
            assert!(msg.len() > 5, "error Display too short: {msg}");
        }
    }

    #[test]
    fn error_implements_std_error() {
        fn check<T: std::error::Error>() {}
        check::<GpuError>();
    }
}
