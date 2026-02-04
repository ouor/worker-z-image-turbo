# Base image with CUDA 12.1 (Must match PyTorch/CUDA version requirements)
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Install Python 3.10 and necessary build tools
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN ln -sf /usr/bin/python3.10 /usr/local/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/local/bin/python3

# Copy requirements first
COPY requirements.txt /requirements.txt

# Install dependencies with CMAKE_ARGS for CUDA support
# We set SD_CUDA=ON for stable-diffusion.cpp
RUN CMAKE_ARGS="-DSD_CUDA=ON" pip install -r /requirements.txt

# Copy necessary files
COPY download_weights.py schemas.py handler.py /

# Download models during build to bake them into the image
RUN python /download_weights.py

# Run the handler
CMD ["python", "-u", "/handler.py"]
