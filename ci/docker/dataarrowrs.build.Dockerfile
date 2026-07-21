# syntax=docker/dockerfile:1.3-labs

# Test image for the experimental arrow-rs Parquet reader. Layers a Rust
# toolchain + maturin onto the standard "datal" (arrow-latest) data image and
# builds the native `ray_data_arrow_rs` crate into it, so the `arrow_rs`-tagged
# data tests can import the extension. The crate is abi3 (py39+), so a single
# wheel works across Python versions. Only the dedicated
# ":database: data: arrow-rs reader tests" step uses this image; the main data
# images stay Rust-free.

ARG DOCKER_IMAGE_BASE_BUILD=cr.ray.io/rayproject/datalbuild-py3.12
FROM $DOCKER_IMAGE_BASE_BUILD

# Repo-relative crate dir; its files are supplied via the wanda `srcs` list.
ARG CRATE_DIR=python/ray/data/_internal/datasource_v2/native/ray_data_arrow_rs

COPY $CRATE_DIR /home/ray/ray_data_arrow_rs

SHELL ["/bin/bash", "-ice"]

RUN <<EOF
#!/bin/bash

set -ex

# Official Rust toolchain via rustup (the canonical Rust installer).
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
  | sh -s -- -y --default-toolchain stable --profile minimal
source "$HOME/.cargo/env"

uv pip install --system maturin

# Build the extension wheel and install it system-wide. --no-deps: the crate
# has no Python deps; pyarrow/ray already live in the base image.
cd /home/ray/ray_data_arrow_rs
maturin build --release --out /home/ray/arrow_rs_wheels
uv pip install --system --no-deps /home/ray/arrow_rs_wheels/*.whl

# Fail the image build if the extension can't be imported.
python -c "import ray_data_arrow_rs; print('ray_data_arrow_rs import OK')"

EOF
