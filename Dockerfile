# Copyright 2026 Raul Mc
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Reproducible CUDA 13.3.1 toolkit build for myelin-accelerator.
# Matches ShipOfTheseus local default (/usr/local/cuda → cuda-13.3).
#
# Build (no GPU required for compile-only):
#   docker build -t myelin-accelerator:cuda13.3 .
#
# Optional GPU runtime (needs nvidia-container-toolkit + sm_120 host):
#   docker run --rm --gpus all myelin-accelerator:cuda13.3 \
#     cargo test --locked --features cuda -- --ignored

FROM nvidia/cuda:13.3.1-devel-ubuntu24.04@sha256:03c372fd9c65fe7739279f8c65473b315dc61efaaffab03e1e65bc7be7aee61e

ENV DEBIAN_FRONTEND=noninteractive \
    CARGO_TERM_COLOR=always \
    RUSTFLAGS="-D warnings" \
    MYELIN_CUDA_ARCH=sm_120 \
    CUDA_NVCC=/usr/local/cuda/bin/nvcc \
    PATH="/root/.cargo/bin:${PATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl ca-certificates build-essential pkg-config libssl-dev git \
    && rm -rf /var/lib/apt/lists/* \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain stable --profile minimal \
    && rustup component add rustfmt clippy

WORKDIR /src
COPY . .

# Default: compile the real GPU path (PTX embed). Runtime tests need --gpus all.
RUN cargo build --locked --features cuda \
    && cargo clippy --locked --features cuda -- -D warnings

CMD ["cargo", "test", "--locked", "--features", "cuda"]
