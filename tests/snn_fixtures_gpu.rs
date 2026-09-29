// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Multi-timestep CUDA differential tests for the raw `lif_step` and
//! `lif_step_weighted` symbols on pinned SNN fixtures (LIM-1462 / #44).
//!
//! Run with `cargo test --locked --features cuda -- --ignored --nocapture`.
//!
//! Both fixtures are **workload-only**: the GPU and the CPU oracle execute the
//! fixed myelin v0.2.0 LIF dynamics (decay 0.85, threshold 1.0, reset 0.0,
//! 2 refractory ticks) on real model shapes / weights / affine parameters.
//! This does not validate Spikenaut or NIR semantics; exact interoperability
//! is GH #43 / LIM-1461 (v0.3.0). See `docs/SNN_COMPATIBILITY.md`.
//!
//! Comparison policy: spikes and refractory counters exact; membrane
//! bit-exact. The device LIF update is a single `fma.rn.ftz.f32` (verified in
//! the sm_120 PTX) and the oracle uses `f32::mul_add`, both single-rounded
//! round-to-nearest. The only host/device divergence under `--use_fast_math`
//! is flush-to-zero, so every trace asserts that no subnormal occurs.
//!
//! GPU setup failures fail the test: there is no CPU fallback.

#![cfg(feature = "cuda")]

#[allow(dead_code)] // shared with tests/snn_fixtures.rs
mod snn_support;

use cust::launch;
use cust::stream::{Stream, StreamFlags};
use myelin_accelerator::{GpuBuffer, GpuContext, KernelModule};
use snn_support::{
    Coverage, StepState, TraceContext, UnweightedOracle, WORKLOAD_ONLY, WeightedOracle,
    assert_no_subnormals, check_step, spikenaut, synfire,
};

/// Written into `spikes_out` before every launch so a launch that skips a
/// neuron cannot pass by leaving the previous tick's value in place.
const SPIKE_POISON: u32 = 0xDEAD_BEEF;

struct Device {
    _ctx: GpuContext,
    kernels: KernelModule,
    stream: Stream,
}

fn device() -> Device {
    let ctx = GpuContext::init().unwrap_or_else(|e| panic!("GPU required (no CPU fallback): {e}"));
    let kernels = KernelModule::load().unwrap_or_else(|e| panic!("KernelModule::load: {e}"));
    let stream = Stream::new(StreamFlags::DEFAULT, None).expect("create CUDA stream");
    Device {
        _ctx: ctx,
        kernels,
        stream,
    }
}

fn report(ctx: &TraceContext<'_>, steps: usize, gpu: &Coverage, oracle: &Coverage) {
    eprintln!(
        "[snn-fixture gpu] {} steps={steps} result=match gpu_coverage={gpu:?} oracle_coverage={oracle:?}",
        ctx.describe()
    );
    assert_eq!(gpu.spikes, oracle.spikes, "spike totals must agree");
    assert_eq!(
        gpu.refractory_ticks, oracle.refractory_ticks,
        "refractory totals must agree"
    );
    assert!(gpu.spikes > 0, "no GPU spikes ({})", ctx.describe());
    assert!(
        gpu.refractory_ticks > 0,
        "no GPU refractory transition ({})",
        ctx.describe()
    );
}

/// Count coverage from GPU-read state (independent of the oracle's counter).
/// `ignored_input_ticks` needs the per-neuron current, so it is reported from
/// the oracle side only.
fn gpu_coverage(cov: &mut Coverage, prev_refract: &[u32], got: &StepState) {
    for i in 0..got.membrane.len() {
        cov.spikes += got.spikes[i] as usize;
        if prev_refract[i] > 0 {
            cov.refractory_ticks += 1;
        }
        if got.spikes[i] == 0 && got.membrane[i] != 0.0 {
            cov.subthreshold_nonzero += 1;
        }
        if got.membrane[i] < 0.0 {
            cov.negative_membrane += 1;
        }
    }
}

fn run_spikenaut_weighted(dev: &Device, block: u32) {
    let f = spikenaut();
    assert_eq!(f.compatibility, WORKLOAD_ONLY);
    let (n, k) = (f.n_neurons, f.n_inputs);
    let ctx = TraceContext {
        fixture_id: &f.fixture_id,
        compatibility: &f.compatibility,
        kernel: "lif_step_weighted",
        shape: f.shape(),
        stimulus: &f.stimulus_id,
        seed: Some(f.seed),
        block: Some(block),
    };
    eprintln!(
        "[snn-fixture gpu] start {} steps={} shared_bytes={}",
        ctx.describe(),
        f.steps,
        k * size_of::<f32>()
    );

    let func = dev
        .kernels
        .get_function("lif_step_weighted")
        .expect("lif_step_weighted registered");
    // Persistent device state for the whole run.
    let membrane = GpuBuffer::<f32>::alloc(n).expect("membrane");
    let weights = GpuBuffer::from_slice(&f.weights).expect("weights");
    let mut input = GpuBuffer::<f32>::alloc(k).expect("input");
    let refract = GpuBuffer::<u32>::alloc(n).expect("refract");
    let mut spikes_out = GpuBuffer::from_slice(&vec![SPIKE_POISON; n]).expect("spikes_out");

    let grid = (n as u32).div_ceil(block);
    let shared_bytes = (k * size_of::<f32>()) as u32;
    let mut oracle = WeightedOracle::new(&f.weights, n, k);
    let mut cov = Coverage::default();
    let mut prev_refract = vec![0u32; n];

    for (t, x) in f.inputs().iter().enumerate() {
        input.upload(x).expect("upload input");
        spikes_out
            .upload(&vec![SPIKE_POISON; n])
            .expect("poison spikes_out");
        // SAFETY: membrane/refract/spikes_out hold n elements, weights n×k,
        // input k; dynamic shared memory holds k floats as the kernel requires.
        // All buffers outlive the synchronize below.
        let stream = &dev.stream;
        unsafe {
            launch!(func<<<grid, block, shared_bytes, stream>>>(
                membrane.as_device_ptr(),
                weights.as_device_ptr(),
                input.as_device_ptr(),
                refract.as_device_ptr(),
                spikes_out.as_device_ptr(),
                n as i32,
                k as i32
            ))
            .unwrap_or_else(|e| panic!("launch t={t}: {e:?} ({})", ctx.describe()));
        }
        dev.stream
            .synchronize()
            .unwrap_or_else(|e| panic!("sync t={t}: {e:?} ({})", ctx.describe()));

        let got = StepState {
            membrane: membrane.to_vec().expect("read membrane"),
            refract: refract.to_vec().expect("read refract"),
            spikes: spikes_out.to_vec().expect("read spikes"),
        };
        let expected = oracle.step(x);
        assert_no_subnormals(&ctx, t, &expected.membrane, "oracle membrane");
        if let Err(msg) = check_step(&ctx, t, &got, &expected) {
            panic!("{msg}");
        }
        gpu_coverage(&mut cov, &prev_refract, &got);
        prev_refract = got.refract;
    }
    report(&ctx, f.steps, &cov, &oracle.coverage);
    assert!(cov.subthreshold_nonzero > 0, "membrane never integrated");
    assert!(cov.negative_membrane > 0, "negative rows never exercised");
}

#[test]
#[ignore] // requires GPU + driver ≥ 570
fn spikenaut_weighted_lif_matches_v0_2_0_oracle_every_timestep() {
    let dev = device();
    // 32: one warp covers all 16 neurons (single block).
    // 8: two blocks, and each thread stages two inputs into shared memory,
    //    exercising the kernel's blockDim-strided shared-memory fill.
    for block in [32u32, 8] {
        run_spikenaut_weighted(&dev, block);
    }
}

#[test]
#[ignore] // requires GPU + driver ≥ 570
fn synfire_lifneuron_affine_lif_matches_v0_2_0_oracle_every_timestep() {
    let f = synfire();
    assert_eq!(f.compatibility, WORKLOAD_ONLY);
    let dev = device();
    let func = dev
        .kernels
        .get_function("lif_step")
        .expect("lif_step registered");
    let block = 32u32;

    for stim in &f.stimuli {
        let ctx = TraceContext {
            fixture_id: &f.fixture_id,
            compatibility: &f.compatibility,
            kernel: "lif_step",
            shape: "n_neurons=1 affine=1x1".to_string(),
            stimulus: &stim.identity,
            seed: None,
            block: Some(block),
        };
        // Host Affine node only; NIR LIF parameters are deliberately unused.
        let currents = f.currents(stim);
        eprintln!(
            "[snn-fixture gpu] start {} steps={} affine_w={} affine_b={}",
            ctx.describe(),
            currents.len(),
            f.affine_weight,
            f.affine_bias
        );

        let membrane = GpuBuffer::<f32>::alloc(1).expect("membrane");
        let mut i_ext = GpuBuffer::<f32>::alloc(1).expect("i_ext");
        let refract = GpuBuffer::<u32>::alloc(1).expect("refract");
        let mut spikes_out = GpuBuffer::from_slice(&[SPIKE_POISON]).expect("spikes_out");
        let mut oracle = UnweightedOracle::new(1);
        let mut cov = Coverage::default();
        let mut prev_refract = vec![0u32];

        for (t, &current) in currents.iter().enumerate() {
            assert_no_subnormals(&ctx, t, &[current], "current");
            i_ext.upload(&[current]).expect("upload i_ext");
            spikes_out
                .upload(&[SPIKE_POISON])
                .expect("poison spikes_out");
            // SAFETY: every buffer holds exactly one element for n_neurons=1
            // and outlives the synchronize below; no shared memory is used.
            let stream = &dev.stream;
            unsafe {
                launch!(func<<<1u32, block, 0u32, stream>>>(
                    membrane.as_device_ptr(),
                    i_ext.as_device_ptr(),
                    refract.as_device_ptr(),
                    spikes_out.as_device_ptr(),
                    1i32
                ))
                .unwrap_or_else(|e| panic!("launch t={t}: {e:?} ({})", ctx.describe()));
            }
            dev.stream
                .synchronize()
                .unwrap_or_else(|e| panic!("sync t={t}: {e:?} ({})", ctx.describe()));

            let got = StepState {
                membrane: membrane.to_vec().expect("read membrane"),
                refract: refract.to_vec().expect("read refract"),
                spikes: spikes_out.to_vec().expect("read spikes"),
            };
            let expected = oracle.step(&[current]);
            assert_no_subnormals(&ctx, t, &expected.membrane, "oracle membrane");
            if let Err(msg) = check_step(&ctx, t, &got, &expected) {
                panic!("{msg}");
            }
            gpu_coverage(&mut cov, &prev_refract, &got);
            prev_refract = got.refract;
        }
        report(&ctx, currents.len(), &cov, &oracle.coverage);
    }
}
