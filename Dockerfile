FROM runpod/base:1.0.2-cuda1281-ubuntu2404

# Set environment variables
# This ensures Python output is immediately visible in logs
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    libglib2.0-0 \
    build-essential \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    ca-certificates \
    tzdata \
    ffmpeg \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*


# Install UV and update PATH
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Create virtual environment
RUN uv venv /opt/venv --python 3.10
ENV PATH="/opt/venv/bin:$PATH"

# Install Python packages (psutil, packaging)
RUN uv pip install psutil packaging

# Copy dependency files first (Docker cache optimization)
COPY requirements.txt .

# Install packages in virtual environment with UV
RUN uv pip install -r requirements.txt

# Install pip in virtual environment (needed for build)
RUN uv pip install pip

# Install PyTorch, torchvision, torchaudio with CUDA 12.1
RUN uv pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# Install xformers
RUN uv pip install -U xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu121

RUN uv pip install soundfile
RUN uv pip install librosa

RUN uv pip install misaki[en] ninja psutil packaging 

# Install flash_attn
RUN uv pip install flash-attn==2.7.4.post1 --no-build-isolation

RUN uv pip install hf_transfer

# Copy application code
COPY . .

RUN huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./weights/Wan2.1-I2V-14B-480P
RUN huggingface-cli download TencentGameMate/chinese-wav2vec2-base --local-dir ./weights/chinese-wav2vec2-base
RUN huggingface-cli download TencentGameMate/chinese-wav2vec2-base model.safetensors --revision refs/pr/1 --local-dir ./weights/chinese-wav2vec2-base
RUN huggingface-cli download MeiGen-AI/InfiniteTalk --local-dir ./weights/InfiniteTalk