# Base image: PyTorch 2.0.1 + CUDA 11.7 (ổn định cho Detectron2)
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Ho_Chi_Minh

# Cài các package cần thiết
RUN apt-get update && apt-get install -y \
    tzdata \
    git \
    python3-dev \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Tạo workspace
WORKDIR /workspace

# Copy requirements and constraints
COPY requirements.txt constraints.txt ./

# CRITICAL: Pin NumPy to 1.x BEFORE installing anything else
# PyTorch 2.0.1 and detectron2 require NumPy 1.x (ABI version 0x1000009)
# Force reinstall to override any NumPy 2.x from base image
RUN pip install --no-cache-dir --force-reinstall -c constraints.txt "numpy<2.0,>=1.21.0"

# Cài torch + torchvision + detectron2 + các lib khác
# Detectron2 official wheels link cho torch==2.0.1 + cu117
# Use constraints to prevent NumPy upgrade
RUN pip install --no-cache-dir -c constraints.txt \
    torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu117

# Install packages (use constraints to ensure NumPy stays at 1.x)
RUN pip install --no-cache-dir -c constraints.txt \
    'detectron2@git+https://github.com/facebookresearch/detectron2.git' \
    git+https://github.com/lucasb-eyer/pydensecrf.git \
    opencv-python \
    timm \
    matplotlib \
    tqdm \
    scikit-image \
    scipy \
    scikit-learn \
    moviepy \
    colored

# Install requirements.txt while respecting NumPy constraint
RUN pip install --no-cache-dir -c constraints.txt -r requirements.txt

# Default command
CMD ["/bin/bash"]
