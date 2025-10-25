from flask import Flask, request, jsonify
from dotenv import load_dotenv
from services.video_processor import start_video_processing
from services.job_manager import job_manager
import os, time, json, logging
from datetime import datetime

# Configure logging to show debug messages
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

load_dotenv()
app = Flask(__name__)

@app.route("/process", methods=["POST"])
def process_video():
    """Start async video processing and return job ID immediately"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*60}")
    print(f"[{timestamp}] 🎬 NEW ASYNC VIDEO PROCESSING REQUEST")
    print(f"{'='*60}")

    data = request.get_json()
    video_url = data.get("video_url")
    video_id = data.get("video_id")
    clip_duration = data.get("clip_duration", "60")
    webhook_url = data.get("webhook_url")
    platform = data.get("platform", "YouTube")
    style = data.get("style", "default")
    transcription_method = data.get("transcription_method", "faster-whisper")

    print(f"[{timestamp}] 📝 Request Parameters - URL: {video_url}, Duration: {clip_duration}s, Platform: {platform}, Style: {style}, Transcription: {transcription_method}, Webhook: {webhook_url}")

    # Validation
    if not video_url:
        return jsonify({"error": "video_url is required"}), 400

    if transcription_method not in ["groq", "faster-whisper"]:
        return jsonify({"error": "transcription_method must be 'groq' or 'faster-whisper'"}), 400

    # Start async processing
    job_data = {
        "video_url": video_url,
        "video_id": video_id,  # Pass video_id through to the processing pipeline
        "clip_duration": clip_duration,
        "webhook_url": webhook_url,
        "platform": platform,
        "style": style,
        "transcription_method": transcription_method
    }
    
    job_id = start_video_processing(job_data)
    
    print(f"[{timestamp}] ✅ Job {job_id} started - processing in background", flush=True)
    print(f"[{timestamp}] 🔄 Background thread initiated for job {job_id}", flush=True)
    print(f"{'='*60}\n", flush=True)

    return jsonify({
        "status": "accepted",
        "job_id": job_id,
        "message": "Video processing started. Use /status/{job_id} to check progress.",
        "status_url": f"/status/{job_id}"
    }), 202

@app.route("/status/<job_id>", methods=["GET"])
def get_job_status(job_id):
    """Get the status of a processing job"""
    job = job_manager.get_job(job_id)
    
    if not job:
        return jsonify({"error": "Job not found"}), 404
    
    return jsonify({
        "job_id": job_id,
        "status": job["status"].value,
        "progress": job.get("progress", 0),
        "message": job.get("message", ""),
        "result": job.get("result"),
        "error": job.get("error"),
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at")
    })

@app.route("/status/health", methods=["GET"])
def health_check():
    """Health check endpoint for Docker"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "clipfactory-python"
    })


if __name__ == "__main__":
    print("🚀 Starting ClipFactory Python Server...")
    print("📡 Endpoint: http://0.0.0.0:8000/process")
    print("-" * 50)
    app.run(host="0.0.0.0", port=8000, debug=False)
