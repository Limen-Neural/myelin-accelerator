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

/// Host GIF step matching `gif_step_weighted`.
///
/// `weights` is row-major `n_neurons × n_inputs`. All buffers are
/// length-checked; panics only on programmer error in tests.
#[allow(clippy::too_many_arguments)]
pub fn gif_step_weighted(
    membrane: &mut [f32],
    adaptation: &mut [f32],
    weights: &[f32],
    input_spikes: &[f32],
    input_current: &[f32],
    refractory: &mut [u32],
    spikes_out: &mut [u32],
    n_neurons: usize,
    n_inputs: usize,
) {
    assert_eq!(membrane.len(), n_neurons);
    assert_eq!(adaptation.len(), n_neurons);
    assert_eq!(refractory.len(), n_neurons);
    assert_eq!(spikes_out.len(), n_neurons);
    assert_eq!(input_spikes.len(), n_inputs);
    assert_eq!(input_current.len(), n_neurons);
    assert_eq!(weights.len(), n_neurons.saturating_mul(n_inputs));

    for tid in 0..n_neurons {
        let mut v = membrane[tid];
        let mut a = adaptation[tid];
        let mut spike = 0u32;
        let r = refractory[tid];

        if r > 0 {
            v = LIF_RESET;
            refractory[tid] = r - 1;
            adaptation[tid] = a * GIF_ADAPTATION_DECAY;
        } else {
            a *= GIF_ADAPTATION_DECAY;
            let row = &weights[tid * n_inputs..tid * n_inputs + n_inputs];
            let mut drive = 0.0f32;
            for j in 0..n_inputs {
                drive = row[j].mul_add(input_spikes[j], drive);
            }
            v = v * GIF_LEAK + drive * GIF_DRIVE_SCALE - a * GIF_ADAPTATION_TERM
                + input_current[tid];
            let threshold = GIF_THRESHOLD_BASE + a * GIF_ADAPTATION_SCALE;
            if v >= threshold {
                spike = 1;
                v -= threshold * GIF_RESET_RATIO;
                a += 1.0;
                refractory[tid] = LIF_REFRACT_TICK;
            }
            adaptation[tid] = a;
        }
        membrane[tid] = v;
        spikes_out[tid] = spike;
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
        gif_step_weighted(
            &mut membrane,
            &mut adaptation,
            &weights,
            &input,
            &[0.0],
            &mut refract,
            &mut spikes,
            1,
            1,
        );
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
        gif_step_weighted(
            &mut membrane,
            &mut adaptation,
            &weights,
            &input,
            &[0.0],
            &mut refract,
            &mut spikes,
            1,
            1,
        );
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
        gif_step_weighted(
            &mut membrane,
            &mut adaptation,
            &weights,
            &input,
            &current,
            &mut refract,
            &mut spikes,
            1,
            1,
        );
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
