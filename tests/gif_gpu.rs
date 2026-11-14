// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Public-API GPU goldens for GIF temporal stepping (requires CUDA + sm_120 driver).

#![cfg(feature = "cuda")]

use myelin_accelerator::GpuAccelerator;
use myelin_accelerator::gif::gif_step_weighted;

#[test]
#[ignore] // requires GPU + driver ≥ 570
fn gif_step_weighted_tick_matches_host_ref() {
    let mut acc = GpuAccelerator::new();
    assert!(acc.is_ready(), "GPU not ready for GIF golden");

    let n = 32usize;
    acc.ensure_temporal_state(n).expect("temporal state");

    let mut weights = vec![0.0f32; n * n];
    let mut input = vec![0.0f32; n];
    for i in 0..n {
        weights[i * n + i] = 1.0;
        input[i] = if i % 3 == 0 { 1.0 } else { 0.0 };
    }

    let mut membrane = vec![0.0f32; n];
    let mut adaptation = vec![0.0f32; n];
    let mut refractory = vec![0u32; n];
    let mut spikes = vec![0u32; n];
    gif_step_weighted(
        &mut membrane,
        &mut adaptation,
        &weights,
        &input,
        &vec![0.0f32; n],
        &mut refractory,
        &mut spikes,
        n,
        n,
    );

    acc.load_synapse_weights_named("gif-parity", &weights)
        .expect("weights");
    acc.upload_temporal_input_spikes(&input).expect("inputs");
    acc.gif_step_weighted_tick(n).expect("gif tick");

    let got_mem = acc.temporal_membrane_to_vec(n).expect("membrane");
    let got_adp = acc.temporal_adaptation_to_vec(n).expect("adaptation");
    let got_spk = acc.temporal_spikes_to_vec(n).expect("spikes");
    for i in 0..n {
        assert!(
            (got_mem[i] - membrane[i]).abs() < 1e-4,
            "membrane[{i}] gpu={} host={}",
            got_mem[i],
            membrane[i]
        );
        assert!(
            (got_adp[i] - adaptation[i]).abs() < 1e-4,
            "adaptation[{i}] gpu={} host={}",
            got_adp[i],
            adaptation[i]
        );
        assert_eq!(got_spk[i], spikes[i], "spikes[{i}]");
    }
}
