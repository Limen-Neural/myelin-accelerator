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
    Coverage, SPIKENAUT_MEM_SHA256, StepState, TraceContext, UnweightedOracle, WORKLOAD_ONLY,
    WeightedOracle, assert_no_subnormals, check_step, decode_q88_word, membrane_bits_match,
    spikenaut, spikenaut_mem_sha256, synfire,
};

#[test]
fn q88_decoding_is_signed_twos_complement() {
    let cases = [
        ("0000", 0),
        ("0100", 256),    // 1.0
        ("000E", 14),     // 0.0546875
        ("019A", 410),    // 1.6015625
        ("FF00", -256),   // -1.0
        ("FFFF", -1),     // -1/256
        ("8000", -32768), // -128.0
        ("7FFF", 32767),
    ];
    for (word, code) in cases {
        assert_eq!(decode_q88_word(word), code, "word {word}");
    }
}

#[test]
#[should_panic(expected = "4 hex digits")]
fn q88_decoding_rejects_wrong_width() {
    let _ = decode_q88_word("100");
}

/// The embedded bytes, not just the card's recorded digest, must hash to the
/// pinned value; editing `fixture.json` alone cannot re-bless other weights.
#[test]
fn spikenaut_embedded_bytes_match_pinned_sha256() {
    let f = spikenaut();
    assert_eq!(spikenaut_mem_sha256(), SPIKENAUT_MEM_SHA256);
    assert_eq!(
        f.sha256, SPIKENAUT_MEM_SHA256,
        "fixture.json digest drifted"
    );
}

#[test]
fn spikenaut_fixture_integrity() {
    let f = spikenaut();
    assert_eq!(
        (
            f.compatibility.as_str(),
            f.kernel.as_str(),
            f.commit.as_str()
        ),
        (
            WORKLOAD_ONLY,
            "lif_step_weighted",
            "6965e12a6e46d29783e69f6b0fda3dda1058bd29"
        )
    );
    assert_eq!((f.n_neurons, f.n_inputs), (16, 16), "hidden layer is 16×16");
    let codes = &f.codes;
    let observed = (
        codes.len(),
        codes.iter().map(|&c| i64::from(c)).sum::<i64>(),
        codes.iter().filter(|&&c| c != 0).count(),
        codes.iter().copied().min(),
        codes.iter().copied().max(),
    );
    let recorded = (
        f.expected_word_count,
        f.expected_code_sum,
        f.expected_nonzero,
        Some(f.expected_min_code),
        Some(f.expected_max_code),
    );
    assert_eq!(observed, recorded, "(count, sum, nonzero, min, max)");
    assert_eq!(codes.len(), f.n_neurons * f.n_inputs);
}

#[test]
fn spikenaut_weights_are_exact_and_neuron_major() {
    let f = spikenaut();
    // Q8.8 → f32 is exact: every weight is an integer multiple of 1/256.
    for (idx, (&c, &w)) in f.codes.iter().zip(&f.weights).enumerate() {
        assert_eq!(w * 256.0, c as f32, "weight {idx} not exact");
    }
    // Neuron-major spot checks (file line = neuron*16 + input), matching
    // Spikenaut's own test "first hidden row, column 1 is 000E".
    let w = |n: usize, j: usize| f.weights[n * f.n_inputs + j];
    assert_eq!(
        [w(0, 0), w(0, 1), w(0, 2), w(6, 0)],
        [308.0 / 256.0, 14.0 / 256.0, 409.0 / 256.0, -1.0]
    );
    // Axons 5–15 are unused width in the shipped bank.
    let unused_nonzero = (0..f.n_neurons)
        .flat_map(|n| (5..f.n_inputs).map(move |j| (n, j)))
        .find(|&(n, j)| w(n, j) != 0.0);
    assert_eq!(
        unused_nonzero, None,
        "(neuron, input) with weight on axon ≥ 5"
    );
}

#[test]
fn spikenaut_stimulus_is_deterministic_binary_on_axons_0_to_4() {
    let f = spikenaut();
    let a = f.inputs();
    assert_eq!(a, f.inputs(), "same seed must replay the same inputs");
    assert!(
        a.len() == f.steps && (64..=128).contains(&f.steps),
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
    let fires = |j: usize| active[j] > 0;
    assert!((0..5).all(fires), "axons 0-4 all fire: {active:?}");
    assert!(!(5..16).any(fires), "axons 5-15 held at zero: {active:?}");
}

fn assert_tick_invariants(t: usize, s: &StepState) {
    for (n, &r) in s.refract.iter().enumerate() {
        assert!(
            r <= LIF_REFRACT_TICKS,
            "refract {r} out of range t={t} n={n}"
        );
        let ok = if s.spikes[n] == 1 {
            (s.membrane[n], r) == (0.0, LIF_REFRACT_TICKS)
        } else {
            s.membrane[n] < LIF_THRESHOLD
        };
        assert!(ok, "invalid post-tick state t={t} n={n}");
    }
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
        assert_tick_invariants(t, &s);
    }
    let c = oracle.coverage;
    eprintln!(
        "[snn-fixture cpu] {} steps={} decay={LIF_DECAY} coverage={c:?}",
        ctx.describe(),
        f.steps
    );
    let Coverage {
        spikes,
        refractory_ticks,
        ignored_input_ticks,
        subthreshold_nonzero,
        negative_membrane,
    } = c;
    let counters = [
        spikes,
        refractory_ticks,
        ignored_input_ticks,
        subthreshold_nonzero,
        negative_membrane,
    ];
    assert!(counters.iter().all(|&x| x > 0), "coverage gap: {c:?}");
}

#[test]
fn synfire_fixture_integrity() {
    let f = synfire();
    assert_eq!(
        (
            f.compatibility.as_str(),
            f.kernel.as_str(),
            f.fixture_id.as_str()
        ),
        (
            WORKLOAD_ONLY,
            "lif_step",
            "synfire:pabogdan/lifneuron:1.0.0"
        )
    );
    assert_eq!(
        f.model_nir_sha256,
        "0dd9143ef624892d4a6461f474d93653a337b3490a62e23ec9ec5d18fde9b7b6"
    );
    assert_eq!(
        (f.topology.as_str(), f.affine_weight_shape.as_slice()),
        (
            "Input(1) -> Affine(1) -> LIF(1) -> Output(1)",
            &[1usize, 1][..]
        )
    );
    // Decimal JSON values must round to the exact f32 bits read from model.nir.
    let decoded = [f.affine_weight, f.affine_bias, f.nir_tau, f.nir_v_threshold].map(f32::to_bits);
    let recorded = [
        f.affine_weight_bits,
        f.affine_bias_bits,
        f.nir_tau_bits,
        f.nir_v_threshold_bits,
    ];
    assert_eq!(decoded, recorded, "(w, b, tau, v_threshold) f32 bits");
    assert_eq!(
        (f.nir_r, f.nir_v_leak, f.nir_v_reset),
        (1.0, 0.0, None),
        "r, v_leak, v_reset (absent from the artifact)"
    );
    // The NIR neuron differs from v0.2.0: record that the gap is real.
    assert_ne!(f.nir_v_threshold, LIF_THRESHOLD);
}

#[test]
fn synfire_stimuli_are_pinned() {
    let f = synfire();
    let ids: Vec<&str> = f.stimuli.iter().map(|s| s.identity.as_str()).collect();
    assert_eq!(ids, ["nir-paper-d0", "myelin-ramp-256"]);
    let (d0, ramp) = (&f.stimuli[0].x, &f.stimuli[1].x);
    assert_eq!((d0.len(), d0.iter().sum::<f32>()), (100, 34.0));
    assert_eq!((ramp.len(), ramp[0], ramp[127]), (128, 1.0 / 256.0, 0.5));
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
        let specific = match stim.identity.as_str() {
            // Consecutive d0 ones land inside the refractory window.
            "nir-paper-d0" => c.ignored_input_ticks,
            // The ramp integrates sub-threshold before crossing.
            "myelin-ramp-256" => c.subthreshold_nonzero,
            other => panic!("unexpected stimulus {other}"),
        };
        assert!(
            c.spikes > 0 && c.refractory_ticks > 0 && specific > 0,
            "{}: {c:?}",
            stim.identity
        );
    }
}

fn guard_ctx() -> TraceContext<'static> {
    TraceContext {
        fixture_id: "guard",
        compatibility: WORKLOAD_ONLY,
        kernel: "lif_step",
        shape: "n_neurons=1".to_string(),
        stimulus: "guard",
        seed: None,
        block: None,
    }
}

fn one_neuron(v: f32) -> StepState {
    StepState {
        membrane: vec![v],
        refract: vec![0],
        spikes: vec![0],
    }
}

/// Guard for the GPU suite: the per-timestep comparator must reject a one-ulp
/// membrane change, so the bit-exact policy cannot degrade into a tolerance.
#[test]
fn check_step_rejects_one_ulp_membrane_drift() {
    let expected = one_neuron(0.925);
    let got = one_neuron(f32::from_bits(0.925f32.to_bits() + 1));
    let err = check_step(&guard_ctx(), 3, &got, &expected).unwrap_err();
    let needles = [
        "timestep=3",
        "neuron=0",
        "fixture=guard",
        "label=workload-only",
    ];
    let missing: Vec<_> = needles.iter().filter(|n| !err.contains(**n)).collect();
    assert!(missing.is_empty(), "missing {missing:?} in: {err}");
    assert!(check_step(&guard_ctx(), 3, &expected, &expected).is_ok());
}

/// NaN never matches, even an identical NaN bit pattern.
#[test]
fn check_step_rejects_identical_nan_membrane() {
    let nan = one_neuron(f32::NAN);
    assert!(!membrane_bits_match(f32::NAN, f32::NAN));
    let err = check_step(&guard_ctx(), 0, &nan, &nan).unwrap_err();
    assert!(err.contains("membrane mismatch"), "{err}");
}
