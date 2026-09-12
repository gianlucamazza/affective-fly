FROM ubuntu:24.04

# System packages: git for VCS, build-essential for Brian2 C++ code generation,
# curl/ca-certificates to fetch the uv installer.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv system-wide so it is on PATH for every user and shell.
# uv manages the pinned Python 3.11 toolchain (see .python-version) at sync time.
ENV UV_INSTALL_DIR=/usr/local/bin
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && uv --version
