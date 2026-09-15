// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Public-API GPU goldens for GIF (requires CUDA + sm_120 driver).
//!
//! Corinth SAAQ fixtures that poke resident membrane buffers live as
//! `#[ignore]` unit tests on `GpuAccelerator`.

#![cfg(feature = "cuda")]

use myelin_accelerator::gif::{GIF_ADAPTATION_SCALE, gif_step_weighted, saaq_find_best_walker};
use myelin_accelerator::{GpuAccelerator, SnapshotChannels};

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
        &mut refractory,
        &mut spikes,
        n,
        n,
    );
    let expected_walker = saaq_find_best_walker(&membrane, &adaptation, GIF_ADAPTATION_SCALE);

    acc.load_synapse_weights_named("gif-parity", &weights)
        .expect("weights");
    acc.upload_temporal_input_spikes(&input).expect("inputs");
    let walker = acc.gif_step_weighted_tick(n).expect("gif tick");
    assert_eq!(walker, expected_walker);

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

#[test]
#[ignore] // requires GPU + driver ≥ 570
fn project_snapshot_current_launches() {
    let mut acc = GpuAccelerator::new();
    assert!(acc.is_ready(), "GPU not ready for snapshot golden");
    let n = 64usize;
    acc.ensure_temporal_state(n).expect("temporal state");
    acc.project_snapshot_current(
        SnapshotChannels {
            gpu_temp_c: 75.0,
            gpu_power_w: 280.0,
            cpu_tctl_c: 62.0,
            cpu_package_power_w: 90.0,
        },
        n,
    )
    .expect("project_snapshot_current");
    let walker = acc.saaq_find_best_walker(n).expect("saaq after project");
    assert!((walker as usize) < n);
}
