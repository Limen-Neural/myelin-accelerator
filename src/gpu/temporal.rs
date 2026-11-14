// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::gif::{GIF_BLOCK_SIZE, gif_grid};
use crate::gpu::accelerator::GpuAccelerator;
use crate::gpu::error::{GpuError, GpuResult};
use crate::gpu::ffi;
use crate::gpu::memory::GpuBuffer;
use cust::launch;

const TEMPORAL_BLOCK_SIZE: u32 = GIF_BLOCK_SIZE;
const TEMPORAL_SHARED_MEM_BYTES: u32 = 0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SynapsePrecision {
    None,
    F32,
    F16,
}

pub(crate) struct TemporalState {
    neuron_count: usize,
    n_inputs: usize,
    membrane: GpuBuffer<f32>,
    refractory: GpuBuffer<u32>,
    spikes_out: GpuBuffer<u32>,
    input_current: GpuBuffer<f32>,
    input_spikes: GpuBuffer<f32>,
    adaptation: GpuBuffer<f32>,
    weights_f32: GpuBuffer<f32>,
    weights_f16: Option<GpuBuffer<u16>>,
    synapse_precision: SynapsePrecision,
    synapse_signature: Option<String>,
}

impl GpuAccelerator {
    fn temporal_grid(neuron_count: usize) -> GpuResult<u32> {
        gif_grid(neuron_count).map_err(GpuError::LaunchFailed)
    }

    pub fn ensure_temporal_state(&mut self, neuron_count: usize) -> GpuResult<()> {
        if !self.has_context() {
            return Err(GpuError::NoGpu);
        }
        if neuron_count == 0 {
            return Err(GpuError::LaunchFailed(
                "temporal state requires neuron_count > 0".into(),
            ));
        }
        let _ = Self::temporal_grid(neuron_count)?;

        let needs_realloc = self
            .temporal_state
            .as_ref()
            .is_none_or(|state| state.neuron_count != neuron_count);

        if needs_realloc {
            if self.temporal_state.is_some() {
                self.synchronize()?;
            }
            self.temporal_state = Some(Self::build_temporal_state(neuron_count)?);
        }

        Ok(())
    }

    /// Download the current GIF spike vector.
    ///
    /// `neuron_count` must equal the resident temporal size.
    pub fn temporal_spikes_to_vec(&self, neuron_count: usize) -> GpuResult<Vec<u32>> {
        if !self.has_context() {
            return Err(GpuError::NoGpu);
        }
        let state = self
            .temporal_state
            .as_ref()
            .ok_or_else(|| GpuError::MemoryError("temporal state not initialised".into()))?;
        Self::require_state_neuron_count(state, neuron_count)?;
        state.spikes_out.to_vec()
    }

    /// Download the current GIF membrane vector.
    ///
    /// `neuron_count` must equal the resident temporal size.
    pub fn temporal_membrane_to_vec(&self, neuron_count: usize) -> GpuResult<Vec<f32>> {
        if !self.has_context() {
            return Err(GpuError::NoGpu);
        }
        let state = self
            .temporal_state
            .as_ref()
            .ok_or_else(|| GpuError::MemoryError("temporal state not initialised".into()))?;
        Self::require_state_neuron_count(state, neuron_count)?;
        state.membrane.to_vec()
    }

    /// Download the current GIF adaptation vector.
    ///
    /// `neuron_count` must equal the resident temporal size.
    pub fn temporal_adaptation_to_vec(&self, neuron_count: usize) -> GpuResult<Vec<f32>> {
        if !self.has_context() {
            return Err(GpuError::NoGpu);
        }
        let state = self
            .temporal_state
            .as_ref()
            .ok_or_else(|| GpuError::MemoryError("temporal state not initialised".into()))?;
        Self::require_state_neuron_count(state, neuron_count)?;
        state.adaptation.to_vec()
    }

    /// Upload a per-neuron (or per-input) vector into the resident `input_spikes` buffer.
    pub fn upload_temporal_input_spikes(&mut self, input_spikes: &[f32]) -> GpuResult<()> {
        if !self.has_context() {
            return Err(GpuError::NoGpu);
        }
        let state = self
            .temporal_state
            .as_mut()
            .ok_or_else(|| GpuError::MemoryError("temporal state not initialised".into()))?;
        if input_spikes.len() != state.n_inputs {
            return Err(GpuError::MemoryError(format!(
                "temporal input_spikes length mismatch: expected {}, got {}",
                state.n_inputs,
                input_spikes.len()
            )));
        }
        state
            .input_spikes
            .upload(input_spikes)
            .map_err(|e| GpuError::MemoryError(format!("input_spikes upload failed: {e}")))
    }

    fn temporal_state_for_synapse(&mut self) -> GpuResult<&mut TemporalState> {
        if !self.has_context() {
            return Err(GpuError::NoGpu);
        }
        self.temporal_state
            .as_mut()
            .ok_or_else(|| GpuError::MemoryError("temporal state not initialised".into()))
    }

    fn validate_synapse_weights(
        state: &TemporalState,
        weights_len: usize,
        label: &str,
    ) -> GpuResult<usize> {
        let expected = state.neuron_count * state.n_inputs;
        if weights_len != expected {
            return Err(GpuError::MemoryError(format!(
                "{label} length mismatch: expected {} ({}x{}), got {}",
                expected, state.neuron_count, state.n_inputs, weights_len
            )));
        }
        Ok(expected)
    }

    /// Load an f32 synapse matrix, always uploading the supplied contents.
    pub fn load_synapse_weights(&mut self, weights: &[f32]) -> GpuResult<()> {
        self.load_synapse_weights_impl(None, weights)
    }

    /// Load an f32 synapse matrix, skipping the upload only when `signature`
    /// matches the resident signature. Callers must change `signature` whenever
    /// the matrix contents change.
    pub fn load_synapse_weights_named(
        &mut self,
        signature: &str,
        weights: &[f32],
    ) -> GpuResult<()> {
        self.load_synapse_weights_impl(Some(signature), weights)
    }

    fn load_synapse_weights_impl(
        &mut self,
        signature: Option<&str>,
        weights: &[f32],
    ) -> GpuResult<()> {
        let state = self.temporal_state_for_synapse()?;
        Self::validate_synapse_weights(state, weights.len(), "weights")?;
        if let Some(signature) = signature
            && state.synapse_precision == SynapsePrecision::F32
            && state.synapse_signature.as_deref() == Some(signature)
        {
            return Ok(());
        }
        state
            .weights_f32
            .upload(weights)
            .map_err(|e| GpuError::MemoryError(format!("synapse weights upload failed: {e}")))?;
        state.synapse_precision = SynapsePrecision::F32;
        state.synapse_signature = signature.map(str::to_owned);
        Ok(())
    }

    /// Load an IEEE f16 synapse matrix from host `u16` bits.
    pub fn load_synapse_weights_f16_registered(
        &mut self,
        signature: &str,
        weights: &[u16],
    ) -> GpuResult<()> {
        let state = self.temporal_state_for_synapse()?;
        let expected = Self::validate_synapse_weights(state, weights.len(), "f16 weights")?;

        if state.synapse_precision == SynapsePrecision::F16
            && state.synapse_signature.as_deref() == Some(signature)
        {
            return Ok(());
        }

        let f16_weights = match state.weights_f16.as_mut() {
            Some(weights_f16) => weights_f16,
            _ => {
                state.weights_f16 = Some(GpuBuffer::<u16>::alloc(expected)?);
                state
                    .weights_f16
                    .as_mut()
                    .expect("weights_f16 was just inserted")
            }
        };
        f16_weights.upload(weights).map_err(|e| {
            GpuError::MemoryError(format!("registered f16 synapse upload failed: {e}"))
        })?;
        state.synapse_precision = SynapsePrecision::F16;
        state.synapse_signature = Some(signature.to_owned());
        Ok(())
    }

    /// One GIF-weighted tick over the resident temporal state.
    ///
    /// Requires synapse weights to be loaded. Synchronizes before returning so
    /// callers can immediately download spikes, membrane, or adaptation.
    pub fn gif_step_weighted_tick(&mut self, neuron_count: usize) -> GpuResult<()> {
        self.ensure_temporal_state(neuron_count)?;

        let precision = self
            .temporal_state
            .as_ref()
            .ok_or_else(|| GpuError::MemoryError("temporal state not initialised".into()))?
            .synapse_precision;
        if precision == SynapsePrecision::None {
            return Err(GpuError::MemoryError(
                "synapse weights must be loaded before gif_step_weighted_tick".into(),
            ));
        }

        let n_inputs = self
            .temporal_state
            .as_ref()
            .expect("temporal state checked above")
            .n_inputs;
        let grid = Self::temporal_grid(neuron_count)?;
        let shared_bytes = (n_inputs * 4) as u32;

        {
            let stream = self.cuda_stream().ok_or(GpuError::NoGpu)?;
            match precision {
                SynapsePrecision::F32 => {
                    let gif_step = self.kernels()?.get_function("gif_step_weighted")?;
                    let state = self
                        .temporal_state
                        .as_ref()
                        .expect("temporal state checked above");
                    unsafe {
                        launch!(gif_step<<<grid, TEMPORAL_BLOCK_SIZE, shared_bytes, stream>>>(
                            state.membrane.as_device_ptr(),
                            state.adaptation.as_device_ptr(),
                            state.weights_f32.as_device_ptr(),
                            state.input_spikes.as_device_ptr(),
                            state.input_current.as_device_ptr(),
                            state.refractory.as_device_ptr(),
                            state.spikes_out.as_device_ptr(),
                            neuron_count as i32,
                            n_inputs as i32
                        ))
                        .map_err(|e| {
                            Self::ptx_launch_error(
                                "gif_step_weighted",
                                grid,
                                TEMPORAL_BLOCK_SIZE,
                                shared_bytes,
                                Some(neuron_count),
                                e,
                            )
                        })?;
                    }
                }
                SynapsePrecision::F16 => {
                    let state = self
                        .temporal_state
                        .as_ref()
                        .expect("temporal state checked above");
                    let weights_f16 = state.weights_f16.as_ref().ok_or_else(|| {
                        GpuError::MemoryError("f16 synapse buffer not initialised".into())
                    })?;
                    ffi::launch_gif_step_weighted_f16(ffi::GifF16Launch {
                        stream,
                        grid_x: grid,
                        block_x: TEMPORAL_BLOCK_SIZE,
                        shared_bytes,
                        buffers: ffi::GifF16Buffers {
                            membrane: state.membrane.as_device_ptr(),
                            adaptation: state.adaptation.as_device_ptr(),
                            weights: weights_f16.as_device_ptr(),
                            input_spikes: state.input_spikes.as_device_ptr(),
                            input_current: state.input_current.as_device_ptr(),
                            refractory: state.refractory.as_device_ptr(),
                            spikes_out: state.spikes_out.as_device_ptr(),
                        },
                        n_neurons: neuron_count as i32,
                        n_inputs: n_inputs as i32,
                    })?;
                }
                SynapsePrecision::None => unreachable!("validated above"),
            }
        }

        self.synchronize()
    }

    /// Zero resident GIF state (membrane, adaptation, spikes, inputs).
    pub fn reset_temporal_state(&mut self) -> GpuResult<()> {
        if !self.has_context() {
            return Err(GpuError::NoGpu);
        }
        if self.temporal_state.is_none() {
            return Ok(());
        }

        let neuron_count = self
            .temporal_state
            .as_ref()
            .expect("checked above")
            .neuron_count;
        if self.has_modules() {
            let reset_membrane = self.kernels()?.get_function("reset_membrane")?;
            let grid = Self::temporal_grid(neuron_count)?;
            {
                let stream = self.cuda_stream().ok_or(GpuError::NoGpu)?;
                let state = self.temporal_state.as_ref().expect("checked above");
                unsafe {
                    launch!(reset_membrane<<<grid, TEMPORAL_BLOCK_SIZE, TEMPORAL_SHARED_MEM_BYTES, stream>>>(
                        state.membrane.as_device_ptr(),
                        neuron_count as i32,
                        0.0f32
                    ))
                    .map_err(|e| Self::ptx_launch_error(
                        "reset_membrane",
                        grid,
                        TEMPORAL_BLOCK_SIZE,
                        TEMPORAL_SHARED_MEM_BYTES,
                        Some(neuron_count),
                        e,
                    ))?;
                }
            }
            self.synchronize()?;
        } else {
            let state = self.temporal_state.as_mut().expect("checked above");
            state.membrane.zero_prefix(neuron_count)?;
        }

        let state = self.temporal_state.as_mut().expect("checked above");
        state.refractory.zero_prefix(state.neuron_count)?;
        state.spikes_out.zero_prefix(state.neuron_count)?;
        state.input_current.zero_prefix(state.neuron_count)?;
        state.input_spikes.zero_prefix(state.n_inputs)?;
        state.adaptation.zero_prefix(state.neuron_count)?;

        Ok(())
    }

    /// Currently loaded synapse signature, if any.
    pub fn synapse_signature(&self) -> Option<&str> {
        self.temporal_state
            .as_ref()
            .and_then(|state| state.synapse_signature.as_deref())
    }

    fn build_temporal_state(neuron_count: usize) -> GpuResult<TemporalState> {
        let n_inputs = neuron_count;
        let weight_size = neuron_count * n_inputs;
        Ok(TemporalState {
            neuron_count,
            n_inputs,
            membrane: GpuBuffer::<f32>::from_slice(&vec![0.0f32; neuron_count])?,
            refractory: GpuBuffer::<u32>::from_slice(&vec![0u32; neuron_count])?,
            spikes_out: GpuBuffer::<u32>::from_slice(&vec![0u32; neuron_count])?,
            input_current: GpuBuffer::<f32>::from_slice(&vec![0.0f32; neuron_count])?,
            input_spikes: GpuBuffer::<f32>::from_slice(&vec![0.0f32; n_inputs])?,
            adaptation: GpuBuffer::<f32>::from_slice(&vec![0.0f32; neuron_count])?,
            weights_f32: {
                let mut weights = GpuBuffer::<f32>::alloc(weight_size)?;
                weights.zero_prefix(weight_size)?;
                weights
            },
            weights_f16: None,
            synapse_precision: SynapsePrecision::None,
            synapse_signature: None,
        })
    }

    fn require_state_neuron_count(state: &TemporalState, neuron_count: usize) -> GpuResult<()> {
        if state.neuron_count != neuron_count {
            return Err(GpuError::MemoryError(format!(
                "temporal neuron_count mismatch: state has {}, requested {neuron_count}",
                state.neuron_count
            )));
        }
        Ok(())
    }
}
