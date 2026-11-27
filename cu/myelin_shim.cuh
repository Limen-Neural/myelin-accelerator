// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

#ifndef MYELIN_SHIM_CUH
#define MYELIN_SHIM_CUH

#ifdef __cplusplus
extern "C" {
#endif

int myelin_launch_gif_step_weighted_f16(
    void* stream,
    unsigned int grid_x,
    unsigned int block_x,
    unsigned int shared_bytes,
    unsigned int shared_mem_limit,
    void* membrane,
    void* adaptation,
    void* weights,
    void* input_spikes,
    void* input_current,
    void* refractory,
    void* spikes_out,
    int n_neurons,
    int n_inputs,
    float lif_reset,
    unsigned int lif_refractory_ticks,
    float gif_leak,
    float gif_drive_scale,
    float gif_threshold_base,
    float gif_adaptation_scale,
    float gif_adaptation_decay,
    float gif_reset_ratio,
    float gif_adaptation_term);

#ifdef __cplusplus
}
#endif

#endif  // MYELIN_SHIM_CUH
