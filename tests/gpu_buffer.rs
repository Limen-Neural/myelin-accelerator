// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Device-backed `GpuBuffer` allocation / upload contract regression tests
//! for LIM-1302 (requires CUDA + sm_120 driver).
//!
//! These mirror the CPU-stub contract tests in `src/gpu_stub.rs`, but exercise
//! real device buffers: initialised allocation, `from_slice`/`to_vec` round
//! trips, complete-replacement uploads, and length-mismatch rejection that
//! leaves device memory unchanged. Every test requires a working GPU — none of
//! them silently pass when the device is unavailable.

#![cfg(feature = "cuda")]

use cust::stream::{Stream, StreamFlags};
use myelin_accelerator::{GpuAccelerator, GpuBuffer, GpuError};

/// Require a real, ready GPU. A device test that cannot init the GPU must fail,
/// not silently pass. Holding the returned accelerator also keeps a CUDA
/// context current for the lifetime of the buffers created under it.
fn require_gpu() -> GpuAccelerator {
    GpuAccelerator::require_gpu().unwrap_or_else(|err| {
        panic!(
            "GPU required for buffer-contract tests: reason={} detail={}",
            err.fallback_reason().map(|r| r.code()).unwrap_or("unknown"),
            err
        )
    })
}

/// Representative allocation sizes: empty, singleton, and values around the
/// 16-element ternary word and 256-thread block boundaries.
const BOUNDARY_LENS: &[usize] = &[0, 1, 2, 15, 16, 17, 255, 256, 257, 4096];

#[test]
#[ignore] // requires GPU + driver ≥ 570
fn zero_prefix_on_uses_caller_stream_and_preserves_tail() {
    let _gpu = require_gpu();
    let stream = Stream::new(StreamFlags::DEFAULT, None).unwrap();
    let mut buffer = GpuBuffer::from_slice(&[7.0f32, 8.0, 9.0, 10.0]).unwrap();

    // SAFETY: buffer and stream stay alive until synchronization, and no
    // other operation touches this buffer while the memset is in flight.
    unsafe { buffer.zero_prefix_on(2, &stream) }.unwrap();
    stream.synchronize().unwrap();

    assert_eq!(buffer.to_vec().unwrap(), [0.0, 0.0, 9.0, 10.0]);
}

/// Assert that `err` is the length-mismatch `MemoryError` naming both lengths.
fn assert_length_mismatch(err: GpuError, buffer_len: usize, input_len: usize) {
    match err {
        GpuError::MemoryError(msg) => {
            assert!(msg.contains("length mismatch"), "msg: {msg}");
            assert!(
                msg.contains(&format!("buffer has {buffer_len} elements")),
                "msg: {msg}"
            );
            assert!(
                msg.contains(&format!("input has {input_len}")),
                "msg: {msg}"
            );
        }
        other => panic!("expected MemoryError, got {other}"),
    }
}

// ── Initialised allocation ──────────────────────────────────────────────────

#[test]
#[ignore] // requires GPU + driver ≥ 570
fn alloc_reads_back_default_immediately_u8_u32_i32_f32() {
    let _gpu = require_gpu();
    for &len in BOUNDARY_LENS {
        let b8 = GpuBuffer::<u8>::alloc(len).unwrap();
        assert_eq!(b8.len(), len);
        assert_eq!(b8.to_vec().unwrap(), vec![0u8; len], "u8 len {len}");

        let b32 = GpuBuffer::<u32>::alloc(len).unwrap();
        assert_eq!(b32.to_vec().unwrap(), vec![0u32; len], "u32 len {len}");

        let bi = GpuBuffer::<i32>::alloc(len).unwrap();
        assert_eq!(bi.to_vec().unwrap(), vec![0i32; len], "i32 len {len}");

        let bf = GpuBuffer::<f32>::alloc(len).unwrap();
        // f32::default() == 0.0 (positive zero); check the exact bit pattern.
        let got = bf.to_vec().unwrap();
        assert_eq!(got.len(), len, "f32 len {len}");
        assert!(
            got.iter().all(|v| v.to_bits() == 0.0f32.to_bits()),
            "f32 alloc not +0.0 at len {len}"
        );
    }
}

/// A `DeviceCopy` type whose `Default` value is deliberately nonzero. Reading it
/// back after `alloc` proves the buffer is initialised with `T::default()` and
/// not merely zeroed device memory.
#[derive(Clone, Copy, Debug, PartialEq, cust::DeviceCopy)]
#[repr(C)]
struct Sentinel {
    a: u32,
    b: i32,
}

impl Default for Sentinel {
    fn default() -> Self {
        // Nonzero in both fields; distinct so a partial zeroing would also fail.
        Sentinel {
            a: 0xDEAD_BEEF,
            b: -12_345,
        }
    }
}

#[test]
#[ignore] // requires GPU + driver ≥ 570
fn alloc_uses_default_not_zeroing_for_nonzero_default_type() {
    let _gpu = require_gpu();
    for &len in &[1usize, 4, 64, 257] {
        let buf = GpuBuffer::<Sentinel>::alloc(len).unwrap();
        assert_eq!(
            buf.to_vec().unwrap(),
            vec![Sentinel::default(); len],
            "nonzero-default alloc at len {len}"
        );
    }
}

// ── from_slice / to_vec round trips ──────────────────────────────────────────

#[test]
#[ignore] // requires GPU + driver ≥ 570
fn from_slice_to_vec_roundtrip_boundaries() {
    let _gpu = require_gpu();
    for &len in BOUNDARY_LENS {
        let input: Vec<u32> = (0..len as u32)
            .map(|i| i.wrapping_mul(2_654_435_761))
            .collect();
        let buf = GpuBuffer::from_slice(&input).unwrap();
        assert_eq!(buf.len(), len);
        assert_eq!(buf.to_vec().unwrap(), input, "roundtrip len {len}");
    }
}

/// Signed zero and NaN payloads survive an f32 round trip. Compare bit patterns:
/// `-0.0 == 0.0` and `NaN != NaN` under value equality would hide corruption.
#[test]
#[ignore] // requires GPU + driver ≥ 570
fn from_slice_preserves_f32_bit_patterns() {
    let _gpu = require_gpu();
    let input = [
        0.0f32,
        -0.0f32,
        f32::from_bits(0x7FC0_0001), // quiet NaN with payload
        f32::from_bits(0xFFA0_1234), // signaling-ish NaN, sign set, payload
        f32::INFINITY,
        f32::NEG_INFINITY,
        1.5,
        -2.25,
    ];
    let buf = GpuBuffer::from_slice(&input).unwrap();
    let got = buf.to_vec().unwrap();
    let got_bits: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
    let want_bits: Vec<u32> = input.iter().map(|v| v.to_bits()).collect();
    assert_eq!(got_bits, want_bits, "f32 bit patterns not preserved");
}

/// Oversized allocations return `MemoryError` with the same category as the CPU
/// stub (GH #48). Every case fails checked byte arithmetic before touching
/// VRAM, so no device memory is exhausted by this probe.
#[test]
#[ignore] // requires GPU + driver ≥ 570
fn alloc_overflow_returns_memory_error_without_allocating() {
    let _gpu = require_gpu();
    for len in [usize::MAX, isize::MAX as usize] {
        match GpuBuffer::<u64>::alloc(len) {
            Err(GpuError::MemoryError(msg)) => assert!(msg.contains("size overflow"), "msg: {msg}"),
            Err(other) => panic!("expected MemoryError for u64 len {len}, got {other}"),
            Ok(_) => panic!("oversized u64 alloc unexpectedly succeeded"),
        }
    }
    for len in [usize::MAX, isize::MAX as usize + 1] {
        match GpuBuffer::<u8>::alloc(len) {
            Err(GpuError::MemoryError(msg)) => assert!(msg.contains("size overflow"), "msg: {msg}"),
            Err(other) => panic!("expected MemoryError for u8 len {len}, got {other}"),
            Ok(_) => panic!("oversized u8 alloc unexpectedly succeeded"),
        }
    }
    match GpuBuffer::<u32>::alloc(usize::MAX) {
        Err(GpuError::MemoryError(_)) => {}
        Err(other) => panic!("expected MemoryError for u32::MAX elems, got {other}"),
        Ok(_) => panic!("oversized u32 alloc unexpectedly succeeded"),
    }
    // Zero and ordinary allocations still succeed alongside the overflow probes.
    assert!(GpuBuffer::<u64>::alloc(0).unwrap().is_empty());
    assert_eq!(
        GpuBuffer::<u64>::alloc(4).unwrap().to_vec().unwrap(),
        vec![0u64; 4]
    );
}

// ── Upload ───────────────────────────────────────────────────────────────────
#[test]
#[ignore] // requires GPU + driver ≥ 570
fn upload_equal_length_replaces_all() {
    let _gpu = require_gpu();
    let mut buf = GpuBuffer::from_slice(&[9i32; 8]).unwrap();
    let replacement: Vec<i32> = (0..8).map(|i| i * 10 - 15).collect();
    buf.upload(&replacement).unwrap();
    assert_eq!(buf.to_vec().unwrap(), replacement);
}

#[test]
#[ignore] // requires GPU + driver ≥ 570
fn upload_empty_into_empty_ok() {
    let _gpu = require_gpu();
    let mut buf = GpuBuffer::<i32>::alloc(0).unwrap();
    buf.upload(&[]).unwrap();
    assert!(buf.to_vec().unwrap().is_empty());
}

/// A too-short upload is rejected with the exact stub payload and leaves the
/// existing device contents unchanged (not a dependency panic).
#[test]
#[ignore] // requires GPU + driver ≥ 570
fn upload_too_short_rejected_and_non_mutating() {
    let _gpu = require_gpu();
    let original = vec![1i32, 2, 3, 4];
    let mut buf = GpuBuffer::from_slice(&original).unwrap();
    assert_length_mismatch(buf.upload(&[7, 8]).unwrap_err(), 4, 2);
    assert_eq!(buf.to_vec().unwrap(), original, "device contents mutated");
}

/// A too-long upload is rejected and non-mutating.
#[test]
#[ignore] // requires GPU + driver ≥ 570
fn upload_too_long_rejected_and_non_mutating() {
    let _gpu = require_gpu();
    let original = vec![1i32, 2, 3, 4];
    let mut buf = GpuBuffer::from_slice(&original).unwrap();
    assert_length_mismatch(buf.upload(&[7, 8, 9, 10, 11]).unwrap_err(), 4, 5);
    assert_eq!(buf.to_vec().unwrap(), original, "device contents mutated");
}

/// Empty/nonempty mismatches fail in both directions with lengths in the payload.
#[test]
#[ignore] // requires GPU + driver ≥ 570
fn upload_empty_nonempty_mismatches_fail() {
    let _gpu = require_gpu();

    let mut empty = GpuBuffer::<i32>::alloc(0).unwrap();
    assert_length_mismatch(empty.upload(&[1]).unwrap_err(), 0, 1);

    let mut nonempty = GpuBuffer::from_slice(&[5i32, 6, 7]).unwrap();
    assert_length_mismatch(nonempty.upload(&[]).unwrap_err(), 3, 0);
    // The rejected empty upload must not have disturbed the contents.
    assert_eq!(nonempty.to_vec().unwrap(), vec![5, 6, 7]);
}
