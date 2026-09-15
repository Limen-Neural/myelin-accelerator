// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

// ════════════════════════════════════════════════════════════════════
//  gpu/ffi.rs — C ABI shim wrappers for Blackwell-critical kernels
//
//  Most kernels still launch through fatbin/PTX in kernel.rs. The two
//  latency-critical Blackwell paths below launch through a linked CUDA
//  shim so Rust can pass raw driver handles into a runtime `<<<>>>` call.
// ════════════════════════════════════════════════════════════════════

use crate::gpu::error::{GpuError, GpuResult};
use crate::launch_hook::{LaunchFailure, LaunchType, report_launch_failure};
use cust::memory::{DeviceCopy, DevicePointer};
use cust::stream::Stream;
use std::ffi::c_void;

unsafe extern "C" {
    fn myelin_launch_gif_step_weighted_f16(
        stream: *mut c_void,
        grid_x: u32,
        block_x: u32,
        shared_bytes: u32,
        membrane: *mut c_void,
        adaptation: *mut c_void,
        weights: *mut c_void,
        input_spikes: *mut c_void,
        refractory: *mut c_void,
        spikes_out: *mut c_void,
        n_neurons: i32,
        n_inputs: i32,
    ) -> i32;

    fn myelin_launch_saaq_find_best_walker(
        stream: *mut c_void,
        grid_x: u32,
        block_x: u32,
        shared_bytes: u32,
        membrane: *mut c_void,
        adaptation: *mut c_void,
        partial_scores: *mut c_void,
        partial_walkers: *mut c_void,
        best_walker_out: *mut c_void,
        n_neurons: i32,
        adaptation_scale: f32,
    ) -> i32;
}

#[allow(clippy::too_many_arguments)]
pub fn launch_gif_step_weighted_f16(
    stream: &Stream,
    grid_x: u32,
    block_x: u32,
    shared_bytes: u32,
    membrane: DevicePointer<f32>,
    adaptation: DevicePointer<f32>,
    weights: DevicePointer<u16>,
    input_spikes: DevicePointer<f32>,
    refractory: DevicePointer<u32>,
    spikes_out: DevicePointer<u32>,
    n_neurons: i32,
    n_inputs: i32,
) -> GpuResult<()> {
    let code = unsafe {
        myelin_launch_gif_step_weighted_f16(
            stream.as_inner().cast::<c_void>(),
            grid_x,
            block_x,
            shared_bytes,
            device_ptr_to_void(membrane),
            device_ptr_to_void(adaptation),
            device_ptr_to_void(weights),
            device_ptr_to_void(input_spikes),
            device_ptr_to_void(refractory),
            device_ptr_to_void(spikes_out),
            n_neurons,
            n_inputs,
        )
    };

    if code == 0 {
        Ok(())
    } else {
        let gpu_error = GpuError::LaunchFailed(format!(
            "myelin_launch_gif_step_weighted_f16 failed with CUDA runtime error code {code}"
        ));
        report_c_abi_failure(
            "gif_step_weighted_f16",
            grid_x,
            block_x,
            shared_bytes,
            Some(n_neurons as usize),
            &gpu_error,
        );
        Err(gpu_error)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn launch_saaq_find_best_walker(
    stream: &Stream,
    grid_x: u32,
    block_x: u32,
    shared_bytes: u32,
    membrane: DevicePointer<f32>,
    adaptation: DevicePointer<f32>,
    partial_scores: DevicePointer<f32>,
    partial_walkers: DevicePointer<u32>,
    best_walker_out: DevicePointer<u32>,
    n_neurons: i32,
    adaptation_scale: f32,
) -> GpuResult<()> {
    let code = unsafe {
        myelin_launch_saaq_find_best_walker(
            stream.as_inner().cast::<c_void>(),
            grid_x,
            block_x,
            shared_bytes,
            device_ptr_to_void(membrane),
            device_ptr_to_void(adaptation),
            device_ptr_to_void(partial_scores),
            device_ptr_to_void(partial_walkers),
            device_ptr_to_void(best_walker_out),
            n_neurons,
            adaptation_scale,
        )
    };

    if code == 0 {
        Ok(())
    } else {
        let gpu_error = GpuError::LaunchFailed(format!(
            "myelin_launch_saaq_find_best_walker failed with CUDA runtime error code {code}"
        ));
        report_c_abi_failure(
            "saaq_find_best_walker",
            grid_x,
            block_x,
            shared_bytes,
            Some(n_neurons as usize),
            &gpu_error,
        );
        Err(gpu_error)
    }
}

fn report_c_abi_failure(
    kernel_name: &str,
    grid_x: u32,
    block_x: u32,
    shared_mem: u32,
    neuron_count: Option<usize>,
    error: &GpuError,
) {
    report_launch_failure(LaunchFailure {
        kernel_name: kernel_name.to_string(),
        launch_type: LaunchType::CAbiShim,
        grid: (grid_x, 1, 1),
        block: (block_x, 1, 1),
        shared_mem,
        neuron_count,
        error: error.to_string(),
        jit_error_log: None,
        jit_info_log: None,
    });
}

fn device_ptr_to_void<T: DeviceCopy>(ptr: DevicePointer<T>) -> *mut c_void {
    ptr.as_raw() as usize as *mut c_void
}
