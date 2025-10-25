# ClipFactory Python - Docker Deployment Guide

## 🚀 Vast.ai GPU Deployment

This guide covers deploying ClipFactory Python on Vast.ai's GPU cloud platform for scalable AI video processing.

## 📋 Prerequisites

- Vast.ai account with credits
- Docker Hub account (optional, for custom images)
- Required API keys:
  - Cloudflare R2 storage credentials
  - OpenAI API key (optional)
  - Groq API key (optional)

## 🛠️ Deployment Options

### Option 1: Using Vast.ai Template (Recommended)

1. **Upload Template**:
   - Go to Vast.ai Templates section
   - Upload `vast-ai-template.json`
   - Configure environment variables

2. **Set Environment Variables**:
   ```bash
   R2_ENDPOINT_URL=https://your-account-id.r2.cloudflarestorage.com
   R2_ACCESS_KEY_ID=your_access_key
   R2_SECRET_ACCESS_KEY=your_secret_key
   R2_BUCKET_NAME=your_bucket_name
   OPENAI_API_KEY=your_openai_key
   GROQ_API_KEY=your_groq_key
   ```

3. **Launch Instance**:
   - Select template
   - Choose GPU (minimum 8GB VRAM recommended)
   - Launch instance

### Option 2: Manual Docker Deployment

1. **Build and Push Image** (if customizing):
   ```bash
   docker build -t your-username/clipfactory-python .
   docker push your-username/clipfactory-python
   ```

2. **Launch on Vast.ai**:
   ```bash
   vastai create instance \
     --image your-username/clipfactory-python \
     --disk 50 \
     --gpu-ram 8 \
     --env R2_ENDPOINT_URL=your_endpoint \
     --env R2_ACCESS_KEY_ID=your_key \
     --env R2_SECRET_ACCESS_KEY=your_secret \
     --env R2_BUCKET_NAME=your_bucket \
     --ports 8000:8000
   ```

### Option 3: Local Testing with Docker Compose

```bash
# Set environment variables in .env file
cp .env.example .env
# Edit .env with your credentials

# Build and run
docker-compose up --build
```

## 🔧 Configuration

### GPU Requirements
- **Minimum**: 8GB VRAM (RTX 3070, RTX 4060 Ti, etc.)
- **Recommended**: 16GB+ VRAM (RTX 4080, RTX 4090, A100, etc.)
- **CUDA**: 12.1 or compatible

### Memory Requirements
- **RAM**: 16GB minimum, 32GB recommended
- **Disk**: 50GB minimum, 100GB recommended
- **Shared Memory**: 2GB (automatically configured)

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `R2_ENDPOINT_URL` | Yes | Cloudflare R2 storage endpoint |
| `R2_ACCESS_KEY_ID` | Yes | R2 access key ID |
| `R2_SECRET_ACCESS_KEY` | Yes | R2 secret access key |
| `R2_BUCKET_NAME` | Yes | R2 bucket name |
| `OPENAI_API_KEY` | No | OpenAI API for transcription |
| `GROQ_API_KEY` | No | Groq API for fast transcription |
| `CLIPPER_MAX_WORKERS` | No | Max worker threads (default: 4) |
| `FLASK_ENV` | No | Flask environment (default: production) |

## 📊 Monitoring and Health Checks

### Health Check Endpoint
```bash
curl http://your-instance-ip:8000/status/health
```

### Docker Health Check
The container includes automatic health checks:
- **Interval**: 30 seconds
- **Timeout**: 10 seconds
- **Retries**: 3
- **Start Period**: 40 seconds

### Logs
```bash
# View container logs
docker logs clipfactory-container

# Follow logs in real-time
docker logs -f clipfactory-container
```

## 🔄 API Usage

### Process Video
```bash
curl -X POST http://your-instance-ip:8000/process \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://example.com/video.mp4",
    "transcription_method": "groq"
  }'
```

### Check Job Status
```bash
curl http://your-instance-ip:8000/status/JOB_ID
```

## 🚨 Troubleshooting

### Common Issues

1. **GPU Not Detected**:
   ```bash
   # Check GPU availability
   nvidia-smi
   
   # Verify CUDA in container
   docker exec -it container_name nvidia-smi
   ```

2. **Out of Memory**:
   - Reduce `CLIPPER_MAX_WORKERS`
   - Choose instance with more VRAM
   - Process shorter videos

3. **Slow Processing**:
   - Ensure GPU acceleration is working
   - Check network bandwidth for video downloads
   - Monitor CPU/GPU utilization

4. **Storage Issues**:
   - Verify R2 credentials
   - Check bucket permissions
   - Monitor disk space

### Performance Optimization

1. **GPU Selection**:
   - RTX 4090: Best price/performance
   - A100: Maximum performance
   - RTX 3090: Good budget option

2. **Instance Configuration**:
   - Use NVMe SSD storage
   - Select high-bandwidth regions
   - Enable persistent storage for models

3. **Batch Processing**:
   - Process multiple videos simultaneously
   - Use queue management for high loads
   - Implement auto-scaling based on demand

## 📈 Scaling

### Horizontal Scaling
- Deploy multiple instances
- Use load balancer (nginx, HAProxy)
- Implement job queue (Redis, RabbitMQ)

### Vertical Scaling
- Increase GPU memory
- Add more CPU cores
- Expand storage capacity

## 💰 Cost Optimization

### Vast.ai Tips
- Use spot instances for batch processing
- Monitor usage and stop unused instances
- Choose optimal GPU for your workload
- Use interruptible instances for development

### Resource Management
- Implement auto-shutdown for idle instances
- Use efficient video formats
- Cache frequently used models
- Optimize batch sizes

## 🔒 Security

### Best Practices
- Use non-root user in container
- Secure API endpoints
- Rotate API keys regularly
- Monitor access logs
- Use HTTPS in production

### Network Security
- Restrict port access
- Use VPN for sensitive data
- Implement rate limiting
- Monitor for unusual activity

## 📞 Support

For issues or questions:
1. Check container logs
2. Verify environment variables
3. Test health endpoint
4. Review Vast.ai instance status
5. Contact support with detailed error information

## 🔄 Updates

To update the application:
1. Pull latest code
2. Rebuild Docker image
3. Stop current container
4. Start new container with updated image
5. Verify health check passes

---

**Note**: This deployment is optimized for Vast.ai's GPU cloud platform. For other cloud providers, adjust the configuration accordingly.