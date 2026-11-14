// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

#include "myelin_shim.cuh"

#include "spiking_network.cu"

extern "C" int myelin_launch_gif_step_weighted_f16(
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
    int n_inputs)
{
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    float* membrane_ptr = reinterpret_cast<float*>(membrane);
    float* adaptation_ptr = reinterpret_cast<float*>(adaptation);
    const half* weights_ptr = reinterpret_cast<const half*>(weights);
    const float* input_spikes_ptr = reinterpret_cast<const float*>(input_spikes);
    const float* input_current_ptr = reinterpret_cast<const float*>(input_current);
    unsigned int* refractory_ptr = reinterpret_cast<unsigned int*>(refractory);
    unsigned int* spikes_out_ptr = reinterpret_cast<unsigned int*>(spikes_out);

    if (shared_bytes > 48u * 1024u) {
        cudaError_t attr_status = cudaFuncSetAttribute(
            gif_step_weighted_f16,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shared_mem_limit));
        if (attr_status != cudaSuccess) {
            return static_cast<int>(attr_status);
        }
    }

    gif_step_weighted_f16<<<grid_x, block_x, shared_bytes, cuda_stream>>>(
        membrane_ptr,
        adaptation_ptr,
        weights_ptr,
        input_spikes_ptr,
        input_current_ptr,
        refractory_ptr,
        spikes_out_ptr,
        n_neurons,
        n_inputs);

    return static_cast<int>(cudaGetLastError());
}
