# Limen-Neural/myelin-accelerator Wiki

> This directory is machine-managed by cubic. Edit wiki content through [cubic wiki settings](https://www.cubic.dev/wiki/Limen-Neural/myelin-accelerator) and custom instructions.

Wiki version: 2
Source commit: 7ed09a8a587f53604fad2aed98516ada865f52d5
Source branch: main
Generated: 2026-07-30T06:31:53.772Z

## Contents

### Overview

- [Home / Repository Overview](01-sec-overview/01-page-home.md)
- [Hardware & VRAM Requirements](01-sec-overview/02-page-hardware.md)
- [Consumer Integration Guide](01-sec-overview/03-page-integration.md)
- [Changelog & Optimizations](01-sec-overview/04-page-release-notes.md)

### System Architecture

- [Architecture Overview](02-sec-architecture/01-page-arch-overview.md)
- [Rust-CUDA FFI Boundary](02-sec-architecture/02-page-rust-cuda-ffi.md)

### Core Features: CUDA Kernels

- [Spiking Network Inference](03-sec-core-kernels/01-page-kernel-snn.md)
- [SAT Solver Kernel](03-sec-core-kernels/02-page-kernel-sat.md)
- [Vector Similarity & MoE Routing](03-sec-core-kernels/03-page-kernel-routing.md)

### Core Features: Host Utilities

- [Ternary & Binary Bitpacking](04-sec-core-host/01-page-bitpacking.md)
- [CPU-Safe Stub API](04-sec-core-host/02-page-cpu-stub.md)

### Data Management & GPU Memory

- [GPU Accelerator & Context](05-sec-data-flow/01-page-gpu-context.md)
- [GPU Memory Management](05-sec-data-flow/02-page-gpu-memory.md)
- [PTX Module Loading](05-sec-data-flow/03-page-kernel-loading.md)

### Backend Systems

- [Error Handling (GpuError)](06-sec-backend-systems/01-page-error-handling.md)
- [NVTX Profiling Instrumentation](06-sec-backend-systems/02-page-nvtx-profiling.md)

### Build & Infrastructure

- [Cargo Build Features](07-sec-infrastructure/01-page-cargo-features.md)
- [CMake & NVCC Integration](07-sec-infrastructure/02-page-cmake-nvcc.md)
- [PTX Versioning & Blackwell Support](07-sec-infrastructure/03-page-ptx-versioning.md)

### Testing & Validation

- [Testing Strategy & CI](08-sec-testing/01-page-testing-strategy.md)
- [Local GPU Quality Gate](08-sec-testing/02-page-gpu-quality-gate.md)
- [GPU Microbenchmarks](08-sec-testing/03-page-microbenchmarks.md)
- [CTest & CLion Workflow](08-sec-testing/04-page-ctest-clion.md)

### Developer Tools & Workflow

- [AI Agent Boundaries & Guidelines](09-sec-dev-tools/01-page-ai-agents.md)
- [Commit Signing Requirements](09-sec-dev-tools/02-page-commit-signing.md)
