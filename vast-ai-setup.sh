#!/bin/bash

# Vast.ai Setup Script for ClipFactory Python
echo "🚀 Setting up ClipFactory Python on Vast.ai..."

# Update system packages
apt-get update

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    echo "📦 Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    rm get-docker.sh
fi

# Install Docker Compose if not present
if ! command -v docker-compose &> /dev/null; then
    echo "📦 Installing Docker Compose..."
    curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
fi

# Clone the repository (if not already present)
if [ ! -d "/workspace/clipfactory-python" ]; then
    echo "📥 Cloning ClipFactory repository..."
    cd /workspace
    git clone https://github.com/yourusername/clipfactory-python.git
    cd clipfactory-python
else
    echo "📂 Repository already exists, updating..."
    cd /workspace/clipfactory-python
    git pull
fi

# Set environment variables (these should be set in Vast.ai template)
echo "🔧 Setting up environment variables..."
cat > .env << EOF
# R2 Storage Configuration
R2_ENDPOINT_URL=${R2_ENDPOINT_URL}
R2_ACCESS_KEY_ID=${R2_ACCESS_KEY_ID}
R2_SECRET_ACCESS_KEY=${R2_SECRET_ACCESS_KEY}
R2_BUCKET_NAME=${R2_BUCKET_NAME}

# API Keys
OPENAI_API_KEY=${OPENAI_API_KEY}
GROQ_API_KEY=${GROQ_API_KEY}

# Application Configuration
CLIPPER_MAX_WORKERS=4
FFMPEG_PATH=ffmpeg
FLASK_ENV=production
FLASK_DEBUG=false

# GPU Configuration
CUDA_VISIBLE_DEVICES=0
NVIDIA_VISIBLE_DEVICES=all
NVIDIA_DRIVER_CAPABILITIES=compute,utility
EOF

# Build and start the application
echo "🏗️ Building Docker image..."
docker-compose build

echo "🚀 Starting ClipFactory service..."
docker-compose up -d

# Wait for service to be ready
echo "⏳ Waiting for service to be ready..."
sleep 30

# Check if service is running
if curl -f http://localhost:8000/status/health > /dev/null 2>&1; then
    echo "✅ ClipFactory is running successfully!"
    echo "🌐 Service available at: http://localhost:8000"
    echo "📊 Health check: http://localhost:8000/status/health"
    echo "🎬 Process endpoint: http://localhost:8000/process"
else
    echo "❌ Service failed to start. Checking logs..."
    docker-compose logs
fi

echo "🎉 Setup complete!"