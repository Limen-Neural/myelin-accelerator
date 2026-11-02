// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

//! #46 / LIM-1464: run under scripts/sanitize_lifecycle.sh. Rust assertions
//! alone cannot detect CUDA errors silently discarded by cust destructors.
#![cfg(feature = "cuda")]

use cust::context::ContextHandle;
use cust::context::legacy::{Context, ContextFlags, ContextStack, CurrentContext};
use cust::device::Device;
use myelin_accelerator::{GpuAccelerator, GpuBuffer};

fn current() -> cust::sys::CUcontext {
    CurrentContext::get_current().unwrap().get_inner()
}

fn clear_current() {
    // SAFETY: CUDA is initialized; null unbinds the top context. Tests call
    // this only with a single binding, so the resulting stack is empty.
    unsafe {
        assert_eq!(
            cust::sys::cuCtxSetCurrent(std::ptr::null_mut()),
            cust::sys::CUresult::CUDA_SUCCESS
        );
    }
    assert!(current().is_null());
}

fn caller_context() -> Context {
    Context::create_and_push(ContextFlags::SCHED_AUTO, Device::get_device(0).unwrap()).unwrap()
}

#[test]
#[ignore] // requires GPU + driver ≥ 570
fn construct_and_immediate_drop() {
    drop(GpuAccelerator::require_gpu().unwrap());
}

#[test]
#[ignore] // requires GPU + driver ≥ 570
fn repeated_construct_drop() {
    for _ in 0..16 {
        drop(GpuAccelerator::require_gpu().unwrap());
    }
}

// Buffers are scoped below the accelerator so caller-owned allocations are
// released before its final context reference. No extra GpuContext is retained.
fn exercise_scratch(gpu: &GpuAccelerator, reject_foreign_context: bool) {
    let assignment = GpuBuffer::from_slice(&vec![1u8; 513]).unwrap();
    let mut flags = GpuBuffer::<u8>::alloc(513).unwrap();
    let scores = GpuBuffer::from_slice(&vec![7i32; 513]).unwrap();
    let mut best_score = GpuBuffer::<i32>::alloc(1).unwrap();
    let mut best_walker = GpuBuffer::<i32>::alloc(1).unwrap();
    let clauses = GpuBuffer::from_slice(&[1i32]).unwrap();
    for walkers in [1, 257, 513] {
        if reject_foreign_context {
            let accelerator_context = CurrentContext::get_current().unwrap();
            let caller = caller_context();
            let before = current();
            let result = gpu.satsolver_aux_reduce_best_async(
                &assignment,
                &mut flags,
                &scores,
                &mut best_score,
                &mut best_walker,
                &clauses,
                walkers,
                1,
                1,
                1,
            );
            assert!(
                result.is_err(),
                "scratch work must reject a foreign context"
            );
            assert_eq!(current(), before);
            drop(caller);
            CurrentContext::set_current(&accelerator_context).unwrap();
        }
        gpu.satsolver_aux_reduce_best(
            &assignment,
            &mut flags,
            &scores,
            &mut best_score,
            &mut best_walker,
            &clauses,
            walkers,
            1,
            1,
            1,
        )
        .unwrap();
        assert_eq!(best_score.to_vec().unwrap(), [7]);
        assert_eq!(best_walker.to_vec().unwrap(), [0]);
    }
}

#[test]
#[ignore] // requires GPU + driver ≥ 570
fn drop_after_scratch_allocation_and_growth() {
    let gpu = GpuAccelerator::require_gpu().unwrap();
    exercise_scratch(&gpu, false);
    drop(gpu);
}

#[test]
#[ignore] // requires GPU + driver ≥ 570
fn scratch_replacement_rejects_foreign_context() {
    let gpu = GpuAccelerator::require_gpu().unwrap();
    exercise_scratch(&gpu, true);
    drop(gpu);
}

#[test]
#[ignore] // requires GPU + driver ≥ 570
fn drop_restores_different_caller_context() {
    let gpu = GpuAccelerator::require_gpu().unwrap();
    exercise_scratch(&gpu, false);
    let caller = caller_context();
    let before = current();
    drop(gpu);
    assert_eq!(current(), before);
    drop(caller);
}

#[test]
#[ignore] // requires GPU + driver ≥ 570
fn drop_preserves_nested_caller_context_stack() {
    let gpu = GpuAccelerator::require_gpu().unwrap();
    clear_current();
    let outer = caller_context();
    let outer_raw = current();
    let inner = caller_context();
    let inner_raw = current();
    drop(gpu);
    assert_eq!(current(), inner_raw);
    ContextStack::pop().unwrap();
    assert_eq!(current(), outer_raw);
    drop(inner);
    drop(outer);
}

#[test]
#[ignore] // requires GPU + driver ≥ 570
fn drop_restores_no_current_context() {
    let gpu = GpuAccelerator::require_gpu().unwrap();
    exercise_scratch(&gpu, false);
    clear_current();
    drop(gpu);
    assert!(current().is_null());
}
