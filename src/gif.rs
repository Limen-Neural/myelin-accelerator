// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

//! GIF host reference matching `cu/spiking_network.cu`.
//!
//! These constants and CPU kernels are the numerical reference for GPU
//! temporal-state parity tests.

/// Membrane leak per GIF tick (`GIF_LEAK` in `spiking_network.cu`).
pub const GIF_LEAK: f32 = 0.92;
/// Synaptic-drive scale (`GIF_DRIVE_SCALE`).
pub const GIF_DRIVE_SCALE: f32 = 0.75;
/// Base firing threshold (`GIF_THRESHOLD_BASE`).
pub const GIF_THRESHOLD_BASE: f32 = 0.65;
/// Adaptation contribution to threshold (`GIF_ADAPTATION_SCALE`).
pub const GIF_ADAPTATION_SCALE: f32 = 0.22;
/// Adaptation decay per tick (`GIF_ADAPTATION_DECAY`).
pub const GIF_ADAPTATION_DECAY: f32 = 0.94;
/// Soft-reset fraction of threshold (`GIF_RESET_RATIO`).
pub const GIF_RESET_RATIO: f32 = 0.35;
/// Adaptation subtracted from membrane (`GIF_ADAPTATION_TERM`).
pub const GIF_ADAPTATION_TERM: f32 = 0.05;
/// Reset potential after a spike or during refractory (`LIF_RESET`).
pub const LIF_RESET: f32 = 0.0;
/// Absolute refractory ticks (`LIF_REFRACT_TICK`).
pub const LIF_REFRACT_TICK: u32 = 2;
/// Threads per GIF block (matches the CUDA launch).
pub const GIF_BLOCK_SIZE: u32 = 256;

/// Grid.x for GIF launches. Rejects empty counts and counts that would wrap
/// `u32` when divided into blocks.
pub fn gif_grid(neuron_count: usize) -> Result<u32, String> {
    if neuron_count == 0 {
        return Err("temporal state requires neuron_count > 0".into());
    }
    let blocks = (neuron_count as u128).div_ceil(GIF_BLOCK_SIZE as u128);
    if blocks > u32::MAX as u128 {
        return Err(format!(
            "temporal grid exceeds u32 block limit (requested {neuron_count} neurons)"
        ));
    }
    Ok(blocks.max(1) as u32)
}

/// Per-neuron mutable GIF state.
pub struct GifNeuronState<'a> {
    pub membrane: &'a mut [f32],
    pub adaptation: &'a mut [f32],
    pub refractory: &'a mut [u32],
    pub spikes_out: &'a mut [u32],
}

/// Read-only GIF inputs for one step.
pub struct GifStepInputs<'a> {
    /// Row-major `n_neurons × n_inputs` synapse matrix.
    pub weights: &'a [f32],
    pub input_spikes: &'a [f32],
    pub input_current: &'a [f32],
}

/// Host GIF step matching `gif_step_weighted`.
///
/// `n_neurons` and `n_inputs` are derived from the buffer lengths. All buffers
/// are length-checked; panics only on programmer error in tests.
pub fn gif_step_weighted(neurons: &mut GifNeuronState<'_>, inputs: &GifStepInputs<'_>) {
    let n_neurons = neurons.membrane.len();
    let n_inputs = inputs.input_spikes.len();
    assert_eq!(neurons.adaptation.len(), n_neurons);
    assert_eq!(neurons.refractory.len(), n_neurons);
    assert_eq!(neurons.spikes_out.len(), n_neurons);
    assert_eq!(inputs.input_current.len(), n_neurons);
    assert_eq!(inputs.weights.len(), n_neurons.saturating_mul(n_inputs));

    for tid in 0..n_neurons {
        let mut v = neurons.membrane[tid];
        let mut a = neurons.adaptation[tid];
        let mut spike = 0u32;
        let r = neurons.refractory[tid];

        if r > 0 {
            v = LIF_RESET;
            neurons.refractory[tid] = r - 1;
            neurons.adaptation[tid] = a * GIF_ADAPTATION_DECAY;
        } else {
            a *= GIF_ADAPTATION_DECAY;
            let row = &inputs.weights[tid * n_inputs..tid * n_inputs + n_inputs];
            let mut drive = 0.0f32;
            for (weight, spike) in row.iter().zip(inputs.input_spikes.iter()) {
                drive = weight.mul_add(*spike, drive);
            }
            v = v * GIF_LEAK + drive * GIF_DRIVE_SCALE - a * GIF_ADAPTATION_TERM
                + inputs.input_current[tid];
            let threshold = GIF_THRESHOLD_BASE + a * GIF_ADAPTATION_SCALE;
            if v >= threshold {
                spike = 1;
                v -= threshold * GIF_RESET_RATIO;
                a += 1.0;
                neurons.refractory[tid] = LIF_REFRACT_TICK;
            }
            neurons.adaptation[tid] = a;
        }
        neurons.membrane[tid] = v;
        neurons.spikes_out[tid] = spike;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gif_step_fires_and_adapts() {
        let mut membrane = vec![0.8f32];
        let mut adaptation = vec![0.0f32];
        let weights = vec![1.0f32];
        let input = vec![1.0f32];
        let mut refract = vec![0u32];
        let mut spikes = vec![0u32];
        let mut neurons = GifNeuronState {
            membrane: &mut membrane,
            adaptation: &mut adaptation,
            refractory: &mut refract,
            spikes_out: &mut spikes,
        };
        let inputs = GifStepInputs {
            weights: &weights,
            input_spikes: &input,
            input_current: &[0.0],
        };
        gif_step_weighted(&mut neurons, &inputs);
        // drive = 1, v = 0.8*0.92 + 0.75 - 0 = 1.486; threshold = 0.65 → spike
        assert_eq!(spikes[0], 1);
        assert!(adaptation[0] > 0.9);
        assert_eq!(refract[0], LIF_REFRACT_TICK);
        let threshold = GIF_THRESHOLD_BASE; // a was 0 after decay from 0
        let v_pre = 0.8 * GIF_LEAK + GIF_DRIVE_SCALE;
        let expected_v = v_pre - threshold * GIF_RESET_RATIO;
        assert!((membrane[0] - expected_v).abs() < 1e-5);
    }

    #[test]
    fn gif_step_respects_refractory() {
        let mut membrane = vec![9.0f32];
        let mut adaptation = vec![2.0f32];
        let weights = vec![1.0f32];
        let input = vec![1.0f32];
        let mut refract = vec![1u32];
        let mut spikes = vec![99u32];
        let mut neurons = GifNeuronState {
            membrane: &mut membrane,
            adaptation: &mut adaptation,
            refractory: &mut refract,
            spikes_out: &mut spikes,
        };
        let inputs = GifStepInputs {
            weights: &weights,
            input_spikes: &input,
            input_current: &[0.0],
        };
        gif_step_weighted(&mut neurons, &inputs);
        assert_eq!(spikes[0], 0);
        assert_eq!(membrane[0], LIF_RESET);
        assert_eq!(refract[0], 0);
        assert!((adaptation[0] - 2.0 * GIF_ADAPTATION_DECAY).abs() < 1e-6);
    }

    #[test]
    fn gif_step_adds_input_current() {
        let mut membrane = vec![0.0f32];
        let mut adaptation = vec![0.0f32];
        let weights = vec![0.0f32];
        let input = vec![0.0f32];
        let current = vec![0.5f32];
        let mut refract = vec![0u32];
        let mut spikes = vec![0u32];
        let mut neurons = GifNeuronState {
            membrane: &mut membrane,
            adaptation: &mut adaptation,
            refractory: &mut refract,
            spikes_out: &mut spikes,
        };
        let inputs = GifStepInputs {
            weights: &weights,
            input_spikes: &input,
            input_current: &current,
        };
        gif_step_weighted(&mut neurons, &inputs);
        let expected = 0.0 * GIF_LEAK + 0.5;
        assert!((membrane[0] - expected).abs() < 1e-6);
        assert_eq!(spikes[0], 0);
    }

    #[test]
    fn gif_grid_rejects_zero_and_u32_wrap() {
        assert!(gif_grid(0).is_err());
        assert_eq!(gif_grid(1).unwrap(), 1);
        assert_eq!(gif_grid(GIF_BLOCK_SIZE as usize).unwrap(), 1);
        assert_eq!(gif_grid(GIF_BLOCK_SIZE as usize + 1).unwrap(), 2);
        let max = (u32::MAX as usize) * (GIF_BLOCK_SIZE as usize);
        assert_eq!(gif_grid(max).unwrap(), u32::MAX);
        assert!(gif_grid(max + 1).is_err());
        assert!(gif_grid(usize::MAX).is_err());
    }
}
