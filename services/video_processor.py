import os
import threading
import tempfile
import time
from datetime import datetime
from services.downloader import download_video
from services.transcriber import transcribe_video
from services.clipper import generate_clips
from services.uploader import upload_to_r2
from services.progress import ProgressReporter
from services.title_generator import generate_titles_batch
from services.viral_detector import extract_viral_moments
from services.job_manager import job_manager, JobStatus

def log_step(step_name, start_time=None, end_time=None):
    """Logging utility function"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if start_time is None and end_time is None:
        print(f"[{timestamp}] 🚀 Starting: {step_name}")
        return time.time()
    elif start_time and end_time:
        duration = end_time - start_time
        print(f"[{timestamp}] ✅ Completed: {step_name} (took {duration:.2f}s)")
        return duration
    else:
        print(f"[{timestamp}] 📝 {step_name}")

def process_video_async(job_id: str, job_data: dict):
    """Process video asynchronously in background thread"""
    print(f"🎬 Background thread started for job {job_id}", flush=True)
    
    # Create progress reporter class
    class JobProgressReporter:
        def __init__(self, job_id, webhook_url, video_id=None):
            self.job_id = job_id
            self.webhook_url = webhook_url
            self.video_id = video_id
            self.original_reporter = ProgressReporter(webhook_url) if webhook_url else None
        
        def update(self, progress, message, **kwargs):
            job_manager.update_progress(self.job_id, progress, message)
            if self.original_reporter:
                # Always pass video_id if available, and merge any additional kwargs
                update_kwargs = {"video_id": self.video_id} if self.video_id else {}
                update_kwargs.update(kwargs)
                self.original_reporter.update(progress, message, **update_kwargs)
    
    # Initialize progress reporter to None to avoid unbound variable issues
    progress = None
    
    try:
        video_url = job_data["video_url"]
        video_id = job_data.get("video_id")  # Get video_id from job_data
        # Ensure video_id is a string for path operations
        if video_id is not None:
            video_id = str(video_id)
        clip_duration = job_data.get("clip_duration", "60")
        webhook_url = job_data.get("webhook_url")
        platform = job_data.get("platform", "YouTube")
        style = job_data.get("style", "default")
        transcription_method = job_data.get("transcription_method", "faster-whisper")

        print(f"🔄 Processing job {job_id} with URL: {video_url}", flush=True)
        
        # Create progress reporter
        progress = JobProgressReporter(job_id, webhook_url, video_id)
        
        # Update job status to processing
        job_manager.update_job_status(job_id, JobStatus.PROCESSING)
        progress.update(0, "Starting video processing...")
        
        print(f"📊 Job {job_id} status updated to processing", flush=True)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Step 1: Download video
            progress.update(10, "Downloading video...")
            print(f"📥 Downloading video for job {job_id}", flush=True)
            local_path = download_video(video_url, tmpdir)
            
            # Step 2: Transcribe video
            progress.update(30, f"Transcribing video ({transcription_method})...")
            print(f"🎤 Transcribing video for job {job_id}", flush=True)
            transcript = transcribe_video(local_path, method=transcription_method)
            print(f"📝 Transcript generated with {len(transcript)} segments for job {job_id}", flush=True)
            
            # Step 3: Generate clips
            progress.update(60, "Generating clips...")
            print(f"✂️ Generating clips for job {job_id}", flush=True)
            
            # Extract video ID from URL for folder organization (use provided video_id if available)
            if not video_id:
                video_id = video_url.split('/')[-1].split('?')[0].split('=')[-1] if 'youtube.com' in video_url or 'youtu.be' in video_url else job_id
            
            # Ensure video_id is always a string for path operations
            video_id = str(video_id)
            
            # Create video-specific output directory
            video_output_dir = os.path.join(tmpdir, video_id)
            os.makedirs(video_output_dir, exist_ok=True)
            
            print(f"🔍 DEBUG: About to call generate_clips with:", flush=True)
            print(f"  - video_path: {local_path}", flush=True)
            print(f"  - clip_duration: {clip_duration}", flush=True)
            print(f"  - transcript segments: {len(transcript)}", flush=True)
            print(f"  - output_dir: {video_output_dir}", flush=True)
            print(f"  - platform: {platform}", flush=True)
            print(f"  - source_id: {video_id}", flush=True)
            
            try:
                clips = generate_clips(
                    video_path=local_path, 
                    clip_duration=clip_duration, 
                    transcript=transcript, 
                    output_dir=video_output_dir,
                    platform=platform,
                    source_id=video_id
                )
                print(f"🎬 Generated {len(clips)} clips for job {job_id}", flush=True)
                print(f"🔍 DEBUG: generate_clips returned: {clips}", flush=True)
            except Exception as e:
                print(f"❌ ERROR: Exception in generate_clips: {type(e).__name__}: {str(e)}", flush=True)
                import traceback
                print(f"❌ ERROR: Traceback: {traceback.format_exc()}", flush=True)
                clips = []
            
            # Step 4: Upload clips
            progress.update(80, "Uploading clips...")
            print(f"☁️ Uploading clips for job {job_id}", flush=True)
            uploaded_clips = []
            
            for clip in clips:
                video_url_uploaded = upload_to_r2(clip["video"], video_id)
                srt_url = upload_to_r2(clip["srt"], video_id)
                
                uploaded_clips.append({
                    "index": clip["index"],
                    "clip_path": video_url_uploaded,  # Laravel expects clip_path
                    "start": clip["start_time"],      # Laravel expects start (numeric)
                    "end": clip["end_time"],          # Laravel expects end (numeric)
                    "transcript": clip["transcript_text"],  # Laravel expects transcript
                    "subtitles_path": srt_url,        # Laravel expects subtitles_path
                    "video": video_url_uploaded,      # Keep for backward compatibility
                    "subtitles": srt_url             # Keep for backward compatibility
                })
            
            # Step 5: Generate titles
            progress.update(90, "Generating titles...")
            print(f"📝 Generating titles for job {job_id}", flush=True)
            clips_data = [{"index": c["index"], "transcript": c["transcript"]} for c in uploaded_clips]
            titles = generate_titles_batch(clips_data, platform, style)
            
            # Assign titles to clips
            for c in uploaded_clips:
                match = next((t for t in titles if t["index"] == c["index"]), None)
                c["title"] = match["title"] if match else "Untitled"
                # Add optional Laravel fields with default values
                c["description"] = None  # nullable string
                c["virality_score"] = None  # nullable numeric between 0,1
                c["thumbnail_path"] = None  # nullable string
            
            progress.update(100, "Processing complete!", clips=uploaded_clips, total_clips=len(uploaded_clips))
            print(f"✅ Job {job_id} completed successfully with {len(uploaded_clips)} clips", flush=True)
            
            # Set job result
            result = {
                "status": "success",
                "clips": uploaded_clips,
                "total_clips": len(uploaded_clips)
            }
            job_manager.set_job_result(job_id, result)
            
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Job {job_id} failed: {error_msg}", flush=True)
        
        # Send failure webhook with error_message
        if progress is not None:
            progress.update(0, "Processing failed", error_message=error_msg)
        
        job_manager.set_job_error(job_id, error_msg)

def start_video_processing(job_data: dict) -> str:
    """Start video processing in background thread and return job ID"""
    job_id = job_manager.create_job(job_data)
    
    print(f"🚀 Creating job {job_id}", flush=True)
    
    # Start processing in background thread
    thread = threading.Thread(
        target=process_video_async,
        args=(job_id, job_data),
        daemon=True
    )
    thread.start()
    
    print(f"✅ Job {job_id} thread started", flush=True)
    
    return job_id