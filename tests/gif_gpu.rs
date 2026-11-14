// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Public-API GPU goldens for GIF temporal stepping (requires CUDA + sm_120 driver).

#![cfg(feature = "cuda")]

use myelin_accelerator::GpuAccelerator;
use myelin_accelerator::gif::{GifNeuronState, GifStepInputs, gif_step_weighted};

#[test]
#[ignore] // requires GPU + driver ≥ 570
fn gif_step_weighted_tick_matches_host_ref() {
    let mut acc = GpuAccelerator::new();
    assert!(acc.is_ready(), "GPU not ready for GIF golden");

    let n_neurons = 32usize;
    let n_inputs = 16usize;
    acc.ensure_temporal_state(n_neurons, n_inputs)
        .expect("temporal state");

    let mut weights = vec![0.0f32; n_neurons * n_inputs];
    let mut input = vec![0.0f32; n_inputs];
    for i in 0..n_inputs {
        input[i] = if i % 3 == 0 { 1.0 } else { 0.0 };
    }
    for i in 0..n_neurons {
        weights[i * n_inputs + (i % n_inputs)] = 1.0;
    }

    let mut membrane = vec![0.0f32; n_neurons];
    let mut adaptation = vec![0.0f32; n_neurons];
    let mut refractory = vec![0u32; n_neurons];
    let mut spikes = vec![0u32; n_neurons];
    let input_current: Vec<f32> = (0..n_neurons).map(|i| (i % 7) as f32 * 0.01).collect();
    let mut neurons = GifNeuronState {
        membrane: &mut membrane,
        adaptation: &mut adaptation,
        refractory: &mut refractory,
        spikes_out: &mut spikes,
    };
    let inputs = GifStepInputs {
        weights: &weights,
        input_spikes: &input,
        input_current: &input_current,
    };
    gif_step_weighted(&mut neurons, &inputs);

    acc.load_synapse_weights_named("gif-parity", &weights)
        .expect("weights");
    acc.upload_temporal_input_spikes(&input).expect("inputs");
    acc.upload_temporal_input_current(&input_current)
        .expect("input current");
    acc.gif_step_weighted_tick(n_neurons).expect("gif tick");

    let got_mem = acc.temporal_membrane_to_vec(n_neurons).expect("membrane");
    let got_adp = acc
        .temporal_adaptation_to_vec(n_neurons)
        .expect("adaptation");
    let got_spk = acc.temporal_spikes_to_vec(n_neurons).expect("spikes");
    for i in 0..n_neurons {
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
