// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Exact external-model semantics through the public CUDA API (GH #43).

#![cfg(feature = "cuda")]
#![allow(clippy::chunks_exact_to_as_chunks, clippy::needless_range_loop)]

use myelin_accelerator::gif::TemporalDynamics;
use myelin_accelerator::{GpuAccelerator, GpuBuffer};

const SYNFIRE_CSV: &str = include_str!("fixtures/snn/synfire_lifneuron/lif_norse.csv");
const HIDDEN_Q88: &str = include_str!("fixtures/snn/spikenaut/parameters_weights.mem");
const THRESHOLDS_Q88: &str = include_str!("fixtures/snn/spikenaut/parameters.mem");
const DECAYS_Q88: &str = include_str!("fixtures/snn/spikenaut/parameters_decay.mem");
const OUTPUT_Q88: &str = include_str!("fixtures/snn/spikenaut/parameters_output_weights.mem");

fn decode_q88_file(text: &str) -> Vec<f32> {
    text.lines()
        .map(|line| {
            let raw = u16::from_str_radix(line.trim(), 16).expect("Q8.8 hex word");
            f32::from(raw as i16) / 256.0
        })
        .collect()
}

fn synfire_rows() -> Vec<(f32, f32, f32)> {
    SYNFIRE_CSV
        .lines()
        .map(|line| {
            let mut fields = line.split(',').map(|x| x.parse::<f32>().unwrap());
            (
                fields.next().unwrap(),
                fields.next().unwrap(),
                fields.next().unwrap(),
            )
        })
        .collect()
}

#[test]
#[ignore] // requires GPU + driver >= 570
fn synfire_nir_lif_exact_semantics_public_api_matches_pinned_norse_trace() {
    let acc = GpuAccelerator::require_gpu().unwrap();
    let rows = synfire_rows();
    assert_eq!(rows.len(), 1000);
    let dynamics = TemporalDynamics {
        lif_decay: 0.96, // 1 - dt/tau = 1 - 1e-4/0.0025
        lif_threshold: 0.1,
        lif_reset: 0.0,
        lif_refractory_ticks: 0,
        ..TemporalDynamics::default()
    };
    let mut membrane = GpuBuffer::from_slice(&[0.0f32]).unwrap();
    let mut refractory = GpuBuffer::from_slice(&[0u32]).unwrap();
    let mut spikes = GpuBuffer::<u32>::alloc(1).unwrap();

    for (t, &(input, expected_v, expected_z)) in rows.iter().enumerate() {
        // Host affine projection is w=1,b=0; the LIF input gain is dt/tau=0.04.
        let current = GpuBuffer::from_slice(&[input * 0.04]).unwrap();
        acc.lif_step_with_dynamics(
            &mut membrane,
            &current,
            &mut refractory,
            &mut spikes,
            dynamics,
        )
        .unwrap();
        let got_v = membrane.to_vec().unwrap()[0];
        let got_z = spikes.to_vec().unwrap()[0] as f32;
        assert_eq!(got_z, expected_z, "spike mismatch at upsampled tick {t}");
        assert!(
            (got_v - expected_v).abs() <= 1.0e-6,
            "voltage mismatch at upsampled tick {t}: gpu={got_v} norse={expected_v}"
        );
    }
}

struct SpikenautReference {
    weights: Vec<Vec<f32>>,
    decay: Vec<f32>,
    threshold: Vec<f32>,
    v: Vec<f32>,
    output_weights: Vec<f32>,
    output: [f32; 3],
}

impl SpikenautReference {
    fn new() -> Self {
        let hidden = decode_q88_file(HIDDEN_Q88);
        Self {
            weights: hidden.chunks_exact(16).map(<[_]>::to_vec).collect(),
            decay: decode_q88_file(DECAYS_Q88),
            threshold: decode_q88_file(THRESHOLDS_Q88),
            v: vec![0.0; 16],
            output_weights: decode_q88_file(OUTPUT_Q88),
            output: [0.0; 3],
        }
    }

    fn step(&mut self, stim: &[f32; 16]) -> Vec<u32> {
        let mut spikes = vec![0u32; 16];
        for n in 0..16 {
            // Port of tools/hamming_lif.py:_weighted_input at the pinned commit:
            // rounded multiply and rounded addition in input-index order.
            let mut input = 0.0f32;
            for c in 0..16 {
                let product = self.weights[n][c] * stim[c];
                input += product;
            }
            if n >= 12 {
                input += 0.05; // pinned exp-024 Dale inhibitory drive
            }
            let decayed = self.decay[n] * self.v[n];
            self.v[n] = decayed + input;
            if self.v[n] >= self.threshold[n] {
                spikes[n] = 1;
                self.v[n] = 0.0;
            }
        }
        for (n, &spike) in spikes.iter().enumerate() {
            if spike != 0 {
                for (o, out) in self.output.iter_mut().enumerate() {
                    *out += self.output_weights[n * 3 + o];
                }
            }
        }
        spikes
    }
}

fn spikenaut_stimulus(t: usize) -> [f32; 16] {
    let base = [0.8f32, 0.5, 0.3, 0.6, 0.45];
    let mut out = [0.0f32; 16];
    for (j, value) in base.into_iter().enumerate() {
        out[j] = if (t + j).is_multiple_of(4) {
            value * 0.5
        } else {
            value
        };
    }
    out
}

#[test]
#[ignore] // requires GPU + driver >= 570
fn spikenaut_bank_exact_hidden_and_readout_semantics_public_api() {
    let acc = GpuAccelerator::require_gpu().unwrap();
    let hidden = decode_q88_file(HIDDEN_Q88);
    let decay = decode_q88_file(DECAYS_Q88);
    let threshold = decode_q88_file(THRESHOLDS_Q88);
    let output_neuron_major = decode_q88_file(OUTPUT_Q88);
    assert_eq!(hidden.len(), 256);
    assert_eq!(decay.len(), 16);
    assert_eq!(threshold.len(), 16);
    assert_eq!(output_neuron_major.len(), 48);

    // Add a constant channel 16 carrying the pinned 0.05 inhibitory drive.
    let mut weights17 = Vec::with_capacity(16 * 17);
    for n in 0..16 {
        weights17.extend_from_slice(&hidden[n * 16..(n + 1) * 16]);
        weights17.push(if n >= 12 { 0.05 } else { 0.0 });
    }
    let weights_gpu = GpuBuffer::from_slice(&weights17).unwrap();
    let decay_gpu = GpuBuffer::from_slice(&decay).unwrap();
    let threshold_gpu = GpuBuffer::from_slice(&threshold).unwrap();
    let mut membrane = GpuBuffer::<f32>::alloc(16).unwrap();
    let mut refractory = GpuBuffer::<u32>::alloc(16).unwrap();
    let mut spikes = GpuBuffer::<u32>::alloc(16).unwrap();

    // Reuse the per-neuron linear recurrence as a public readout accumulator:
    // decay=1 and an unreachable threshold makes membrane += W_out @ spikes.
    let mut output_rows = vec![0.0f32; 3 * 16];
    for n in 0..16 {
        for o in 0..3 {
            output_rows[o * 16 + n] = output_neuron_major[n * 3 + o];
        }
    }
    let output_weights_gpu = GpuBuffer::from_slice(&output_rows).unwrap();
    let output_decay_gpu = GpuBuffer::from_slice(&[1.0f32; 3]).unwrap();
    let output_threshold_gpu = GpuBuffer::from_slice(&[f32::MAX; 3]).unwrap();
    let mut output_gpu = GpuBuffer::<f32>::alloc(3).unwrap();
    let mut output_refractory = GpuBuffer::<u32>::alloc(3).unwrap();
    let mut output_spikes = GpuBuffer::<u32>::alloc(3).unwrap();

    let mut reference = SpikenautReference::new();
    for t in 0..96 {
        let stim16 = spikenaut_stimulus(t);
        let mut stim17 = stim16.to_vec();
        stim17.push(1.0);
        let input_gpu = GpuBuffer::from_slice(&stim17).unwrap();
        acc.lif_step_weighted_per_neuron(
            &mut membrane,
            &weights_gpu,
            &input_gpu,
            &decay_gpu,
            &threshold_gpu,
            &mut refractory,
            &mut spikes,
            0.0,
            0,
        )
        .unwrap();
        let expected_spikes = reference.step(&stim16);
        let got_spikes = spikes.to_vec().unwrap();
        assert_eq!(
            got_spikes, expected_spikes,
            "hidden spikes differ at tick {t}"
        );
        let got_membrane = membrane.to_vec().unwrap();
        for n in 0..16 {
            assert!(
                (got_membrane[n] - reference.v[n]).abs() <= 2.0e-5,
                "membrane differs at tick {t} neuron {n}: gpu={} reference={}",
                got_membrane[n],
                reference.v[n]
            );
        }

        let spike_current: Vec<f32> = got_spikes.iter().map(|&x| x as f32).collect();
        let spike_current_gpu = GpuBuffer::from_slice(&spike_current).unwrap();
        acc.lif_step_weighted_per_neuron(
            &mut output_gpu,
            &output_weights_gpu,
            &spike_current_gpu,
            &output_decay_gpu,
            &output_threshold_gpu,
            &mut output_refractory,
            &mut output_spikes,
            0.0,
            0,
        )
        .unwrap();
    }

    let got_output = output_gpu.to_vec().unwrap();
    for o in 0..3 {
        assert!(
            (got_output[o] - reference.output[o]).abs() <= 1.0e-5,
            "readout output {o}: gpu={} reference={}",
            got_output[o],
            reference.output[o]
        );
    }
}
