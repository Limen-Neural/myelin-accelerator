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

/// Persistent per-run LIF state on the device (lives for the whole trace).
struct LifState {
    membrane: GpuBuffer<f32>,
    refract: GpuBuffer<u32>,
    spikes_out: GpuBuffer<u32>,
    n: usize,
}

impl LifState {
    fn new(n: usize) -> Self {
        Self {
            membrane: GpuBuffer::alloc(n).expect("alloc membrane"),
            refract: GpuBuffer::alloc(n).expect("alloc refract"),
            spikes_out: GpuBuffer::from_slice(&vec![SPIKE_POISON; n]).expect("alloc spikes_out"),
            n,
        }
    }

    fn poison_spikes(&mut self) {
        self.spikes_out
            .upload(&vec![SPIKE_POISON; self.n])
            .expect("poison spikes_out");
    }

    fn read(&self) -> StepState {
        StepState {
            membrane: self.membrane.to_vec().expect("read membrane"),
            refract: self.refract.to_vec().expect("read refract"),
            spikes: self.spikes_out.to_vec().expect("read spikes"),
        }
    }
}

/// Count coverage from GPU-read state (independent of the oracle's counter).
/// `ignored_input_ticks` needs the per-neuron current, so it is reported from
/// the oracle side only.
fn gpu_coverage(cov: &mut Coverage, prev_refract: &[u32], got: &StepState) {
    let neurons = prev_refract.iter().zip(&got.spikes).zip(&got.membrane);
    for ((&prev, &spike), &v) in neurons {
        cov.spikes += spike as usize;
        cov.refractory_ticks += usize::from(prev > 0);
        cov.subthreshold_nonzero += usize::from(spike == 0 && v != 0.0);
        cov.negative_membrane += usize::from(v < 0.0);
    }
}

/// Drive one trace: per tick, poison spikes, `launch(t)` (upload + launch),
/// synchronize, read membrane/refract/spikes back, advance the oracle with
/// `oracle(t)`, and compare. Returns GPU-side coverage.
fn drive(
    dev: &Device,
    ctx: &TraceContext<'_>,
    state: &mut LifState,
    steps: usize,
    mut launch_tick: impl FnMut(usize, &LifState),
    mut oracle_tick: impl FnMut(usize) -> StepState,
) -> Coverage {
    let mut cov = Coverage::default();
    let mut prev_refract = vec![0u32; state.n];
    for t in 0..steps {
        state.poison_spikes();
        launch_tick(t, state);
        dev.stream
            .synchronize()
            .unwrap_or_else(|e| panic!("sync t={t}: {e:?} ({})", ctx.describe()));
        let got = state.read();
        let expected = oracle_tick(t);
        assert_no_subnormals(ctx, t, &expected.membrane, "oracle membrane");
        if let Err(msg) = check_step(ctx, t, &got, &expected) {
            panic!("{msg}");
        }
        gpu_coverage(&mut cov, &prev_refract, &got);
        prev_refract = got.refract;
    }
    cov
}

fn report(ctx: &TraceContext<'_>, steps: usize, gpu: &Coverage, oracle: &Coverage) {
    eprintln!(
        "[snn-fixture gpu] {} steps={steps} result=match gpu_coverage={gpu:?} oracle_coverage={oracle:?}",
        ctx.describe()
    );
    assert_eq!(
        (gpu.spikes, gpu.refractory_ticks),
        (oracle.spikes, oracle.refractory_ticks),
        "GPU vs oracle (spikes, refractory ticks)"
    );
    assert!(
        gpu.spikes > 0 && gpu.refractory_ticks > 0,
        "need ≥1 spike and ≥1 refractory tick ({})",
        ctx.describe()
    );
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
    let shared_bytes = (k * size_of::<f32>()) as u32;
    eprintln!(
        "[snn-fixture gpu] start {} steps={} shared_bytes={shared_bytes}",
        ctx.describe(),
        f.steps
    );

    let func = dev
        .kernels
        .get_function("lif_step_weighted")
        .expect("lif_step_weighted registered");
    let weights = GpuBuffer::from_slice(&f.weights).expect("weights");
    let mut input = GpuBuffer::<f32>::alloc(k).expect("input");
    let mut state = LifState::new(n);
    let grid = (n as u32).div_ceil(block);
    let inputs = f.inputs();
    let mut oracle = WeightedOracle::new(&f.weights, n, k);

    let cov = drive(
        dev,
        &ctx,
        &mut state,
        f.steps,
        |t, s| {
            input.upload(&inputs[t]).expect("upload input");
            let stream = &dev.stream;
            // SAFETY: membrane/refract/spikes_out hold n elements, weights
            // n×k, input k; dynamic shared memory holds the k floats the
            // kernel stages. All buffers outlive the synchronize in `drive`.
            unsafe {
                launch!(func<<<grid, block, shared_bytes, stream>>>(
                    s.membrane.as_device_ptr(),
                    weights.as_device_ptr(),
                    input.as_device_ptr(),
                    s.refract.as_device_ptr(),
                    s.spikes_out.as_device_ptr(),
                    n as i32,
                    k as i32
                ))
                .unwrap_or_else(|e| panic!("launch t={t}: {e:?}"));
            }
        },
        |t| oracle.step(&inputs[t]),
    );
    report(&ctx, f.steps, &cov, &oracle.coverage);
    assert!(
        cov.subthreshold_nonzero > 0 && cov.negative_membrane > 0,
        "membrane never integrated / negative rows never exercised: {cov:?}"
    );
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
        for (t, &i) in currents.iter().enumerate() {
            assert_no_subnormals(&ctx, t, &[i], "current");
        }
        eprintln!(
            "[snn-fixture gpu] start {} steps={} affine_w={} affine_b={}",
            ctx.describe(),
            currents.len(),
            f.affine_weight,
            f.affine_bias
        );

        let mut i_ext = GpuBuffer::<f32>::alloc(1).expect("i_ext");
        let mut state = LifState::new(1);
        let mut oracle = UnweightedOracle::new(1);
        let cov = drive(
            &dev,
            &ctx,
            &mut state,
            currents.len(),
            |t, s| {
                i_ext.upload(&currents[t..=t]).expect("upload i_ext");
                let stream = &dev.stream;
                // SAFETY: every buffer holds exactly one element for
                // n_neurons=1 and outlives the synchronize in `drive`; no
                // shared memory is used.
                unsafe {
                    launch!(func<<<1u32, block, 0u32, stream>>>(
                        s.membrane.as_device_ptr(),
                        i_ext.as_device_ptr(),
                        s.refract.as_device_ptr(),
                        s.spikes_out.as_device_ptr(),
                        1i32
                    ))
                    .unwrap_or_else(|e| panic!("launch t={t}: {e:?}"));
                }
            },
            |t| oracle.step(&currents[t..=t]),
        );
        report(&ctx, currents.len(), &cov, &oracle.coverage);
    }
}
