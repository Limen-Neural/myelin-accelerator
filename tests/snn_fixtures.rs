// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

//! CPU-safe integrity and oracle-coverage checks for the pinned SNN fixtures
//! (LIM-1462 / #44). Runs in ordinary `cargo test`; fully offline.
//!
//! The CUDA differential run of the same workloads is in
//! `tests/snn_fixtures_gpu.rs`. Both fixtures are **workload-only**: the
//! reference below executes fixed myelin v0.2.0 LIF dynamics, not Spikenaut or
//! NIR dynamics.

#[allow(dead_code)] // shared with tests/snn_fixtures_gpu.rs
mod snn_support;

use myelin_accelerator::oracle::{LIF_DECAY, LIF_REFRACT_TICKS, LIF_THRESHOLD};
use snn_support::{
    StepState, TraceContext, UnweightedOracle, WORKLOAD_ONLY, WeightedOracle, assert_no_subnormals,
    check_step, decode_q88_word, spikenaut, synfire,
};

#[test]
fn q88_decoding_is_signed_twos_complement() {
    assert_eq!(decode_q88_word("0000"), 0);
    assert_eq!(decode_q88_word("0100"), 256); // 1.0
    assert_eq!(decode_q88_word("000E"), 14); // 0.0546875
    assert_eq!(decode_q88_word("019A"), 410); // 1.6015625
    assert_eq!(decode_q88_word("FF00"), -256); // -1.0
    assert_eq!(decode_q88_word("FFFF"), -1); // -1/256
    assert_eq!(decode_q88_word("8000"), -32768); // -128.0
    assert_eq!(decode_q88_word("7FFF"), 32767);
}

#[test]
fn spikenaut_fixture_integrity() {
    let f = spikenaut();
    assert_eq!(f.compatibility, WORKLOAD_ONLY);
    assert_eq!(f.kernel, "lif_step_weighted");
    assert_eq!(f.commit, "6965e12a6e46d29783e69f6b0fda3dda1058bd29");
    assert_eq!(
        f.sha256,
        "825969873444d215d09920e69592700a2cc8594d92e7b14c4059eef8919fe4f3"
    );
    assert_eq!(
        (f.n_neurons, f.n_inputs),
        (16, 16),
        "Spikenaut hidden layer is 16×16"
    );
    assert_eq!(f.codes.len(), f.expected_word_count);
    assert_eq!(f.codes.len(), f.n_neurons * f.n_inputs);
    assert_eq!(
        f.codes.iter().map(|&c| i64::from(c)).sum::<i64>(),
        f.expected_code_sum
    );
    assert_eq!(
        f.codes.iter().filter(|&&c| c != 0).count(),
        f.expected_nonzero
    );
    assert_eq!(f.codes.iter().copied().min(), Some(f.expected_min_code));
    assert_eq!(f.codes.iter().copied().max(), Some(f.expected_max_code));

    // Q8.8 → f32 is exact: every weight is an integer multiple of 1/256.
    for (idx, (&c, &w)) in f.codes.iter().zip(&f.weights).enumerate() {
        assert_eq!(w * 256.0, c as f32, "weight {idx} not exact");
    }
    // Neuron-major spot checks (file line = neuron*16 + input), matching
    // Spikenaut's own test "first hidden row, column 1 is 000E".
    let w = |n: usize, j: usize| f.weights[n * f.n_inputs + j];
    assert_eq!(w(0, 0), 308.0 / 256.0);
    assert_eq!(w(0, 1), 14.0 / 256.0);
    assert_eq!(w(0, 2), 409.0 / 256.0);
    assert_eq!(w(6, 0), -1.0);
    // Axons 5–15 are unused width in the shipped bank.
    for n in 0..f.n_neurons {
        for j in 5..f.n_inputs {
            assert_eq!(w(n, j), 0.0, "neuron {n} input {j}");
        }
    }
}

#[test]
fn spikenaut_stimulus_is_deterministic_binary_on_axons_0_to_4() {
    let f = spikenaut();
    let a = f.inputs();
    assert_eq!(a, f.inputs(), "same seed must replay the same inputs");
    assert_eq!(a.len(), f.steps);
    assert!(
        (64..=128).contains(&f.steps),
        "bounded release-qualification run"
    );
    let mut active = [0usize; 16];
    for row in &a {
        assert_eq!(row.len(), f.n_inputs);
        for (j, &x) in row.iter().enumerate() {
            assert!(x == 0.0 || x == 1.0, "input must be 0/1 spikes");
            active[j] += x as usize;
        }
    }
    assert!(
        active[..5].iter().all(|&c| c > 0),
        "axons 0-4 all fire: {active:?}"
    );
    assert!(
        active[5..].iter().all(|&c| c == 0),
        "axons 5-15 held at zero"
    );
}

#[test]
fn spikenaut_oracle_run_has_spike_refractory_and_membrane_coverage() {
    let f = spikenaut();
    let ctx = TraceContext {
        fixture_id: &f.fixture_id,
        compatibility: &f.compatibility,
        kernel: &f.kernel,
        shape: f.shape(),
        stimulus: &f.stimulus_id,
        seed: Some(f.seed),
        block: None,
    };
    let mut oracle = WeightedOracle::new(&f.weights, f.n_neurons, f.n_inputs);
    for (t, input) in f.inputs().iter().enumerate() {
        let s = oracle.step(input);
        assert_no_subnormals(&ctx, t, &s.membrane, "membrane");
        for (n, &r) in s.refract.iter().enumerate() {
            assert!(
                r <= LIF_REFRACT_TICKS,
                "refract {r} out of range at t={t} n={n}"
            );
            if s.spikes[n] == 1 {
                assert_eq!((s.membrane[n], r), (0.0, LIF_REFRACT_TICKS));
            } else {
                assert!(
                    s.membrane[n] < LIF_THRESHOLD,
                    "unspiked v ≥ threshold t={t} n={n}"
                );
            }
        }
    }
    let c = oracle.coverage;
    eprintln!(
        "[snn-fixture cpu] {} steps={} decay={LIF_DECAY} coverage={c:?}",
        ctx.describe(),
        f.steps
    );
    assert!(c.spikes > 0, "no spikes: {c:?}");
    assert!(c.refractory_ticks > 0, "no refractory ticks: {c:?}");
    assert!(
        c.ignored_input_ticks > 0,
        "no input arrived during refractory: {c:?}"
    );
    assert!(
        c.subthreshold_nonzero > 0,
        "membrane never integrated: {c:?}"
    );
    assert!(
        c.negative_membrane > 0,
        "negative-weight rows never exercised: {c:?}"
    );
}

#[test]
fn synfire_fixture_integrity() {
    let f = synfire();
    assert_eq!(f.compatibility, WORKLOAD_ONLY);
    assert_eq!(f.kernel, "lif_step");
    assert_eq!(f.fixture_id, "synfire:pabogdan/lifneuron:1.0.0");
    assert_eq!(
        f.model_nir_sha256,
        "0dd9143ef624892d4a6461f474d93653a337b3490a62e23ec9ec5d18fde9b7b6"
    );
    assert_eq!(f.topology, "Input(1) -> Affine(1) -> LIF(1) -> Output(1)");
    assert_eq!(f.affine_weight_shape, [1, 1]);
    // Decimal JSON values must round to the exact f32 bits read from model.nir.
    assert_eq!(f.affine_weight.to_bits(), f.affine_weight_bits);
    assert_eq!(f.affine_bias.to_bits(), f.affine_bias_bits);
    assert_eq!(f.nir_tau.to_bits(), f.nir_tau_bits);
    assert_eq!(f.nir_v_threshold.to_bits(), f.nir_v_threshold_bits);
    assert_eq!((f.nir_r, f.nir_v_leak), (1.0, 0.0));
    assert_eq!(f.nir_v_reset, None, "v_reset is absent from the artifact");
    // The NIR neuron differs from v0.2.0: record that the gap is real.
    assert_ne!(f.nir_v_threshold, LIF_THRESHOLD);

    let ids: Vec<&str> = f.stimuli.iter().map(|s| s.identity.as_str()).collect();
    assert_eq!(ids, ["nir-paper-d0", "myelin-ramp-256"]);
    let d0 = &f.stimuli[0].x;
    assert_eq!(d0.len(), 100);
    assert_eq!(d0.iter().sum::<f32>(), 34.0);
    assert_eq!(f.stimuli[1].x[0], 1.0 / 256.0);
    assert_eq!(f.stimuli[1].x[127], 0.5);
}

#[test]
fn synfire_oracle_runs_cover_spikes_and_refractory() {
    let f = synfire();
    for stim in &f.stimuli {
        let ctx = TraceContext {
            fixture_id: &f.fixture_id,
            compatibility: &f.compatibility,
            kernel: &f.kernel,
            shape: "n_neurons=1".to_string(),
            stimulus: &stim.identity,
            seed: None,
            block: None,
        };
        let currents = f.currents(stim);
        // w = 1, b = 0: the host Affine is the identity on these stimuli.
        assert_eq!(currents, stim.x);
        let mut oracle = UnweightedOracle::new(1);
        for (t, &i) in currents.iter().enumerate() {
            let s = oracle.step(&[i]);
            assert_no_subnormals(&ctx, t, &[i], "current");
            assert_no_subnormals(&ctx, t, &s.membrane, "membrane");
        }
        let c = oracle.coverage;
        eprintln!(
            "[snn-fixture cpu] {} steps={} coverage={c:?}",
            ctx.describe(),
            currents.len()
        );
        assert!(
            c.spikes > 0 && c.refractory_ticks > 0,
            "{}: {c:?}",
            stim.identity
        );
        match stim.identity.as_str() {
            // Consecutive d0 ones land inside the refractory window.
            "nir-paper-d0" => assert!(c.ignored_input_ticks > 0, "{c:?}"),
            // The ramp integrates sub-threshold before crossing.
            "myelin-ramp-256" => assert!(c.subthreshold_nonzero > 0, "{c:?}"),
            other => panic!("unexpected stimulus {other}"),
        }
    }
}

/// Guard for the GPU suite: the per-timestep comparator must reject a one-ulp membrane change,
/// so the bit-exact policy cannot silently degrade into a tolerance.
#[test]
fn check_step_rejects_one_ulp_membrane_drift() {
    let ctx = TraceContext {
        fixture_id: "guard",
        compatibility: WORKLOAD_ONLY,
        kernel: "lif_step",
        shape: "n_neurons=1".to_string(),
        stimulus: "guard",
        seed: None,
        block: None,
    };
    let expected = StepState {
        membrane: vec![0.925],
        refract: vec![0],
        spikes: vec![0],
    };
    let mut got = expected.clone();
    got.membrane[0] = f32::from_bits(got.membrane[0].to_bits() + 1);
    let err = check_step(&ctx, 3, &got, &expected).unwrap_err();
    for needle in [
        "timestep=3",
        "neuron=0",
        "fixture=guard",
        "label=workload-only",
    ] {
        assert!(err.contains(needle), "{err}");
    }
}
