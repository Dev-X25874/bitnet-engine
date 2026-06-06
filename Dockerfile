# Dockerfile
# Reproducible build environment for BitNet Engine.
# Base: CUDA 12.4 + Ubuntu 22.04 + Rust stable.

FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential pkg-config libssl-dev git python3 python3-pip \
 && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

# Pre-cache crate registry
WORKDIR /build
COPY Cargo.toml Cargo.lock ./
RUN mkdir -p src && echo "fn main(){}" > src/main.rs \
 && cargo fetch \
 && rm -rf src

# Build the actual project
COPY . .
ARG CUDA_ARCH=sm_80
RUN CUDA_ARCH=${CUDA_ARCH} cargo build --release --features cuda

ENTRYPOINT ["/build/target/release/bitnet-cli"]
CMD ["--help"]
