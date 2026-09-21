// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

//! GIF / SAAQ host reference matching `cu/spiking_network.cu`.
//!
//! These constants and CPU kernels are the numerical reference for
//! corinth-canal parity tests and for LIM-955 fusion work.

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
/// Threads per GIF/SAAQ block (matches the CUDA launch).
pub const GIF_BLOCK_SIZE: u32 = 256;
/// `saaq_reduce_partials_f16` is a single 32-thread warp.
pub const SAAQ_MAX_BLOCKS: u32 = 32;
/// Inactive-lane SAAQ score (`SAAQ_SCORE_SENTINEL` in `spiking_network.cu`).
pub const SAAQ_SCORE_SENTINEL: f32 = f32::NEG_INFINITY;

/// Grid.x for GIF/SAAQ launches. Rejects counts that would wrap `u32` or
/// exceed the pass-2 warp (`SAAQ_MAX_BLOCKS`).
pub fn gif_saaq_grid(neuron_count: usize) -> Result<u32, String> {
    let max_neurons = (SAAQ_MAX_BLOCKS as usize).saturating_mul(GIF_BLOCK_SIZE as usize);
    if neuron_count == 0 {
        return Err("temporal state requires neuron_count > 0".into());
    }
    if neuron_count > max_neurons {
        return Err(format!(
            "temporal grid exceeds SAAQ pass-2 cap of {SAAQ_MAX_BLOCKS} blocks \
             (max {max_neurons} neurons at {GIF_BLOCK_SIZE} threads; requested {neuron_count})"
        ));
    }
    Ok((neuron_count as u32).div_ceil(GIF_BLOCK_SIZE).max(1))
}

/// Four telemetry channels consumed by [`project_snapshot_current`].
///
/// Channel order matches the CUDA kernel: gpu temperature, gpu power,
/// cpu temperature, cpu power.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SnapshotChannels {
    /// GPU temperature in Celsius.
    pub gpu_temp_c: f32,
    /// GPU power in watts.
    pub gpu_power_w: f32,
    /// CPU Tctl temperature in Celsius.
    pub cpu_tctl_c: f32,
    /// CPU package power in watts.
    pub cpu_package_power_w: f32,
}

impl SnapshotChannels {
    /// Pack channels in CUDA kernel order.
    pub fn as_array(self) -> [f32; 4] {
        [
            self.gpu_temp_c,
            self.gpu_power_w,
            self.cpu_tctl_c,
            self.cpu_package_power_w,
        ]
    }
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

/// Host SAAQ argmax matching `saaq_find_best_walker` + `saaq_reduce_partials_f16`.
///
/// Score is `membrane[i] - adaptation_scale * adaptation[i]`. Ties keep the
/// lower walker index.
pub fn saaq_find_best_walker(membrane: &[f32], adaptation: &[f32], adaptation_scale: f32) -> u32 {
    assert_eq!(membrane.len(), adaptation.len());
    assert!(!membrane.is_empty());

    let mut best_score = SAAQ_SCORE_SENTINEL;
    let mut best_walker = 0u32;
    for (i, (&v, &a)) in membrane.iter().zip(adaptation.iter()).enumerate() {
        let score = v - adaptation_scale * a;
        if score > best_score || (score == best_score && (i as u32) < best_walker) {
            best_score = score;
            best_walker = i as u32;
        }
    }
    best_walker
}

/// Host snapshot projection matching `project_snapshot_current`.
pub fn project_snapshot_current(snapshot: SnapshotChannels, n_neurons: usize) -> Vec<f32> {
    let snap = snapshot.as_array();
    let temp_norm = ((snap[0] - 60.0) / 30.0).clamp(-1.0, 1.0);
    let gpu_power_norm = ((snap[1] - 220.0) / 220.0).clamp(-1.0, 1.0);
    let cpu_temp_norm = ((snap[2] - 60.0) / 30.0).clamp(-1.0, 1.0);
    let cpu_power_norm = ((snap[3] - 120.0) / 120.0).clamp(-1.0, 1.0);

    (0..n_neurons)
        .map(|tid| {
            let channel = match tid & 3 {
                0 => temp_norm,
                1 => gpu_power_norm,
                2 => cpu_temp_norm,
                _ => cpu_power_norm,
            };
            let phase_seed = (tid as u32)
                .wrapping_mul(1_664_525)
                .wrapping_add(1_013_904_223);
            let phase = ((phase_seed >> 8) & 1023) as f32 / 1023.0;
            let jitter = (phase - 0.5) * 0.25;
            (0.9 + channel * 0.45 + jitter).max(0.0)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Corinth `test_saaq_two_pass_reduction_selects_global_best_and_tie_breaks`
    /// fixture: 8×256 = 2048 neurons, winner at block 5 lane 9.
    #[test]
    fn corinth_saaq_fixture_selects_global_best() {
        const N: usize = 8 * 256;
        let mut membrane = vec![0.0f32; N];
        let adaptation = vec![0.0f32; N];
        membrane[17] = 2.0;
        membrane[5 * 256 + 9] = 4.5;
        assert_eq!(
            saaq_find_best_walker(&membrane, &adaptation, GIF_ADAPTATION_SCALE),
            (5 * 256 + 9) as u32
        );
    }

    /// Same corinth fixture: equal scores keep the lower walker index.
    #[test]
    fn corinth_saaq_fixture_tie_breaks_lower_index() {
        const N: usize = 8 * 256;
        let mut membrane = vec![0.0f32; N];
        let adaptation = vec![0.0f32; N];
        membrane[11] = 3.0;
        membrane[3 * 256 + 4] = 3.0;
        assert_eq!(
            saaq_find_best_walker(&membrane, &adaptation, GIF_ADAPTATION_SCALE),
            11
        );
    }

    #[test]
    fn saaq_penalises_adaptation() {
        let membrane = [1.0f32, 1.0];
        let adaptation = [0.0f32, 5.0];
        assert_eq!(
            saaq_find_best_walker(&membrane, &adaptation, GIF_ADAPTATION_SCALE),
            0
        );
    }

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
    fn snapshot_projection_is_non_negative_and_channel_periodic() {
        let snap = SnapshotChannels {
            gpu_temp_c: 90.0,
            gpu_power_w: 440.0,
            cpu_tctl_c: 90.0,
            cpu_package_power_w: 240.0,
        };
        let out = project_snapshot_current(snap, 8);
        assert_eq!(out.len(), 8);
        assert!(out.iter().all(|&v| v >= 0.0));
        // Saturated norms → every channel is 1.0, so variation is jitter only.
        let max = out.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let min = out.iter().copied().fold(f32::INFINITY, f32::min);
        assert!(max - min < 0.3);
    }

    #[test]
    fn gif_step_adds_projected_input_current() {
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
    fn gif_saaq_grid_rejects_zero_and_u32_wrap() {
        assert!(gif_saaq_grid(0).is_err());
        assert_eq!(gif_saaq_grid(1).unwrap(), 1);
        assert_eq!(gif_saaq_grid(GIF_BLOCK_SIZE as usize).unwrap(), 1);
        assert_eq!(gif_saaq_grid(GIF_BLOCK_SIZE as usize + 1).unwrap(), 2);
        let max = (SAAQ_MAX_BLOCKS as usize) * (GIF_BLOCK_SIZE as usize);
        assert_eq!(gif_saaq_grid(max).unwrap(), SAAQ_MAX_BLOCKS);
        assert!(gif_saaq_grid(max + 1).is_err());
        assert!(gif_saaq_grid(u32::MAX as usize + 1).is_err());
        assert!(gif_saaq_grid(usize::MAX).is_err());
    }
}
