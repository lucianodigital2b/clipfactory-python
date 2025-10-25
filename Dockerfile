# Use NVIDIA CUDA base image with Python 3.11
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-pip \
    python3.11-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgtk-3-0 \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge

# Install PyTorch with CUDA support (optimized for size)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip cache purge

# Install additional ML packages for GPU acceleration
RUN pip install --no-cache-dir ultralytics mediapipe librosa soundfile && \
    pip cache purge

# Copy application code
COPY . .

# Clean up unnecessary files and caches to reduce image size
RUN apt-get autoremove -y && \
    apt-get autoclean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    find /usr/local -name '*.pyc' -delete && \
    find /usr/local -name '__pycache__' -delete

# Create non-root user for security
RUN useradd -m -u 1000 clipfactory && \
    chown -R clipfactory:clipfactory /app
USER clipfactory

# Create necessary directories with proper permissions
RUN mkdir -p /app/temp /app/uploads /app/downloads && \
    chown -R clipfactory:clipfactory /app

# Download YOLO model if not present (as clipfactory user)
USER clipfactory
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" || echo "YOLO model will be downloaded on first run"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/status/health || exit 1

# Add health endpoint to app (will be added separately)
# Run the application
CMD ["python", "app.py"]