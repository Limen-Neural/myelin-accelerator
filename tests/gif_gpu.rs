// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Public-API GPU goldens for GIF temporal stepping (requires CUDA + sm_120 driver).

#![cfg(feature = "cuda")]

use myelin_accelerator::GpuAccelerator;
use myelin_accelerator::gif::{
    GifNeuronState, GifStepInputs, TemporalDynamics, gif_step_weighted,
    gif_step_weighted_with_dynamics,
};

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
    for (i, value) in input.iter_mut().enumerate() {
        *value = if i % 3 == 0 { 1.0 } else { 0.0 };
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

#[test]
#[ignore] // requires GPU + driver ≥ 570
fn gif_step_weighted_tick_matches_custom_host_dynamics() {
    let mut acc = GpuAccelerator::new();
    assert!(
        acc.is_ready(),
        "GPU not ready for GIF custom-dynamics golden"
    );

    let dynamics = TemporalDynamics {
        lif_reset: 0.05,
        lif_refractory_ticks: 1,
        gif_leak: 0.5,
        gif_drive_scale: 0.9,
        gif_threshold_base: 0.4,
        gif_adaptation_scale: 0.1,
        gif_adaptation_decay: 0.8,
        gif_reset_ratio: 0.2,
        gif_adaptation_term: 0.03,
        ..TemporalDynamics::default()
    };
    let n_neurons = 16usize;
    let n_inputs = 8usize;
    acc.ensure_temporal_state_with_dynamics(n_neurons, n_inputs, dynamics)
        .expect("configured temporal state");
    assert_eq!(acc.temporal_dynamics(), dynamics);

    let weights: Vec<f32> = (0..n_neurons * n_inputs)
        .map(|i| if i % 9 == 0 { 0.75 } else { 0.05 })
        .collect();
    let input: Vec<f32> = (0..n_inputs)
        .map(|i| if i % 2 == 0 { 1.0 } else { 0.0 })
        .collect();
    let input_current: Vec<f32> = (0..n_neurons).map(|i| i as f32 * 0.005).collect();
    let mut membrane = vec![0.1f32; n_neurons];
    let mut adaptation = vec![0.02f32; n_neurons];
    let mut refractory = vec![0u32; n_neurons];
    let mut spikes = vec![0u32; n_neurons];
    gif_step_weighted_with_dynamics(
        &mut GifNeuronState {
            membrane: &mut membrane,
            adaptation: &mut adaptation,
            refractory: &mut refractory,
            spikes_out: &mut spikes,
        },
        &GifStepInputs {
            weights: &weights,
            input_spikes: &input,
            input_current: &input_current,
        },
        &dynamics,
    );

    // The resident GPU state starts zeroed, so upload the same initial state by
    // using a zeroed host reference instead of the nonzero one above.
    let mut zero_membrane = vec![0.0f32; n_neurons];
    let mut zero_adaptation = vec![0.0f32; n_neurons];
    let mut zero_refractory = vec![0u32; n_neurons];
    let mut zero_spikes = vec![0u32; n_neurons];
    gif_step_weighted_with_dynamics(
        &mut GifNeuronState {
            membrane: &mut zero_membrane,
            adaptation: &mut zero_adaptation,
            refractory: &mut zero_refractory,
            spikes_out: &mut zero_spikes,
        },
        &GifStepInputs {
            weights: &weights,
            input_spikes: &input,
            input_current: &input_current,
        },
        &dynamics,
    );

    acc.load_synapse_weights_named("gif-custom-parity", &weights)
        .expect("weights");
    acc.upload_temporal_input_spikes(&input).expect("inputs");
    acc.upload_temporal_input_current(&input_current)
        .expect("input current");
    acc.gif_step_weighted_tick(n_neurons).expect("gif tick");

    assert_eq!(
        acc.temporal_spikes_to_vec(n_neurons).expect("spikes"),
        zero_spikes
    );
    let got_mem = acc.temporal_membrane_to_vec(n_neurons).expect("membrane");
    let got_adp = acc
        .temporal_adaptation_to_vec(n_neurons)
        .expect("adaptation");
    for i in 0..n_neurons {
        assert!((got_mem[i] - zero_membrane[i]).abs() < 1e-4);
        assert!((got_adp[i] - zero_adaptation[i]).abs() < 1e-4);
    }
}
