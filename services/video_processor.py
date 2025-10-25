import os
import threading
import tempfile
import time
import re
from datetime import datetime
from services.downloader import download_video
from services.transcriber import transcribe_video
from services.clipper import generate_clips
from services.frame_focus import focus_on_speaker
from services.uploader import upload_to_r2
from services.progress import ProgressReporter

from services.viral_detector import extract_viral_moments
from services.job_manager import job_manager, JobStatus

def generate_title_from_transcript(transcript_text):
    """Generate a meaningful title from transcript text"""
    if not transcript_text or not transcript_text.strip():
        return "Untitled Clip"
    
    # Clean up the transcript text
    text = transcript_text.strip()
    
    # Remove common filler words and clean up
    text = re.sub(r'\b(um|uh|like|you know|so|well|actually|basically)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove leading commas and spaces that might be left after filler word removal
    text = re.sub(r'^[,\s]+', '', text)
    text = re.sub(r'[,\s]+$', '', text)
    
    # Split into sentences and find the most meaningful one
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    if not sentences:
        # If no good sentences, use first few words
        words = text.split()[:6]
        if words:
            title = ' '.join(words)
            if not title.endswith(('!', '?', '.')):
                title += '...'
            return title.capitalize()
        return "Untitled Clip"
    
    # Use the first meaningful sentence
    title = sentences[0].strip()
    
    # Capitalize first letter
    if title:
        title = title[0].upper() + title[1:] if len(title) > 1 else title.upper()
    
    # Ensure it's not too long (max 60 characters for social media)
    if len(title) > 60:
        title = title[:57] + "..."
    
    # Add punctuation if missing
    if title and not title.endswith(('!', '?', '.')):
        # Add appropriate punctuation based on content
        if any(word in title.lower() for word in ['what', 'how', 'why', 'when', 'where', 'who']):
            title += '?'
        elif any(word in title.lower() for word in ['amazing', 'incredible', 'wow', 'shocking', 'unbelievable']):
            title += '!'
        else:
            title += '.'
    
    return title

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
        aspect_ratio = job_data.get("aspect_ratio", "9:16")  # Default to 16:9
        transcription = job_data.get("subtitles_path")  # Pre-existing transcription from R2

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
            
            # Step 2: Handle transcription
            if transcription:
                # Use pre-existing transcription from R2
                progress.update(30, "Using provided transcription...")
                print(f"📝 Using pre-existing transcription for job {job_id}", flush=True)
                transcript = transcription
                print(f"📝 Loaded transcript with {len(transcript)} segments for job {job_id}", flush=True)
            else:
                # Transcribe video using specified method
                progress.update(30, f"Transcribing video ({transcription_method})...")
                print(f"🎤 Transcribing video for job {job_id}", flush=True)
                transcript = transcribe_video(local_path, method=transcription_method, video_id=video_id, progress=progress)
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
            print(f"  - aspect_ratio: {aspect_ratio}", flush=True)
            
            try:
                clips = generate_clips(
                    video_path=local_path, 
                    clip_duration=clip_duration, 
                    transcript=transcript, 
                    output_dir=video_output_dir,
                    platform=platform,
                    source_id=video_id,
                    aspect_ratio=aspect_ratio
                )
                print(f"🎬 Generated {len(clips)} clips for job {job_id}", flush=True)
                print(f"🔍 DEBUG: generate_clips returned: {clips}", flush=True)
                
                # Step 3.5: Apply smart camera focus to clips
                progress.update(70, "Applying smart camera focus...")
                print(f"🎯 Applying smart camera focus to clips for job {job_id}", flush=True)
                
                focused_clips = []
                for i, clip in enumerate(clips):
                    try:
                        # Create focused version filename
                        clip_filename = os.path.basename(clip["video"])
                        focused_filename = f"focused_{clip_filename}"
                        focused_path = os.path.join(video_output_dir, focused_filename)
                        
                        # Apply smart camera focus (before aspect ratio conversion)
                        print(f"🎯 Processing clip {i+1}/{len(clips)}: {clip_filename}", flush=True)
                        
                        # Use the original video for focus detection, not the aspect-ratio converted clip
                        focus_on_speaker(local_path, focused_path, fps_batch=5, 
                                       start_time=clip["start_time"], 
                                       duration=clip["end_time"] - clip["start_time"])
                        
                        # Delete original clip to save space (we only keep the focused version)
                        try:
                            os.remove(clip["video"])
                            print(f"🗑️ Deleted original clip to save space: {clip['video']}", flush=True)
                        except Exception as delete_error:
                            print(f"⚠️ Warning: Could not delete original clip {clip['video']}: {delete_error}", flush=True)
                        
                        # Update clip path to focused version
                        focused_clip = clip.copy()
                        focused_clip["video"] = focused_path
                        focused_clips.append(focused_clip)
                        
                        print(f"✅ Smart focus applied to clip {i+1}/{len(clips)}", flush=True)
                        
                    except Exception as focus_error:
                        print(f"⚠️ Warning: Smart focus failed for clip {i+1}, using original: {focus_error}", flush=True)
                        # Use original clip if focus fails
                        focused_clips.append(clip)
                
                clips = focused_clips
                print(f"🎯 Smart camera focus completed for {len(clips)} clips", flush=True)
                
                # Step 3.6: Extract viral moments and assign titles
                progress.update(75, "Analyzing viral moments and generating titles...")
                print(f"🔥 Extracting viral moments and generating titles for job {job_id}", flush=True)
                
                try:
                    viral_moments = extract_viral_moments(transcript)
                    print(f"🔥 Found {len(viral_moments)} viral moments", flush=True)
                    
                    # Assign titles from viral moments to clips based on time overlap
                    for clip in clips:
                        clip_start = clip["start_time"]
                        clip_end = clip["end_time"]
                        
                        print(f"🔍 DEBUG: Processing clip {clip['index']}: {clip_start}s-{clip_end}s", flush=True)
                        
                        # Find the viral moment that best overlaps with this clip
                        best_match = None
                        best_overlap = 0
                        
                        for moment in viral_moments:
                            moment_start = moment.get("start_time", 0)
                            moment_end = moment.get("end_time", 0)
                            
                            # Calculate overlap between clip and viral moment
                            overlap_start = max(clip_start, moment_start)
                            overlap_end = min(clip_end, moment_end)
                            overlap_duration = max(0, overlap_end - overlap_start)
                            
                            print(f"  🔍 Checking moment {moment_start}s-{moment_end}s: overlap = {overlap_duration}s", flush=True)
                            
                            if overlap_duration > best_overlap:
                                best_overlap = overlap_duration
                                best_match = moment
                        
                        # Assign title and virality score from best matching viral moment
                        if best_match and best_overlap > 0.5:  # At least 0.5 seconds overlap
                            clip["title"] = best_match.get("title", "Untitled")
                            clip["virality_score"] = best_match.get("virality_score", 0.0)
                            print(f"📝 Assigned title to clip {clip['index']}: '{clip['title']}' (virality: {clip['virality_score']}, overlap: {best_overlap}s)", flush=True)
                        else:
                            # Fallback: Generate title from clip's transcript content
                            clip_transcript = clip.get("transcript_text", "").strip()
                            if clip_transcript:
                                # Create a meaningful title from the transcript
                                title = generate_title_from_transcript(clip_transcript)
                                clip["title"] = title
                                clip["virality_score"] = 0.3  # Default moderate score for transcript-based titles
                                print(f"📝 Generated title from transcript for clip {clip['index']}: '{clip['title']}' (virality: {clip['virality_score']})", flush=True)
                            else:
                                # Last resort: Use clip index
                                clip["title"] = f"Clip {clip['index']}"
                                clip["virality_score"] = 0.1
                                print(f"📝 Using fallback title for clip {clip['index']}: '{clip['title']}' (virality: {clip['virality_score']})", flush=True)
                            
                except Exception as viral_error:
                    print(f"⚠️ Warning: Viral moment extraction failed: {viral_error}", flush=True)
                    # Fallback: Generate titles from transcript content for all clips
                    for clip in clips:
                        clip_transcript = clip.get("transcript_text", "").strip()
                        if clip_transcript:
                            title = generate_title_from_transcript(clip_transcript)
                            clip["title"] = title
                            clip["virality_score"] = 0.3
                            print(f"📝 Generated fallback title from transcript for clip {clip['index']}: '{clip['title']}'", flush=True)
                        else:
                            clip["title"] = f"Clip {clip['index']}"
                            clip["virality_score"] = 0.1
                            print(f"📝 Using index-based title for clip {clip['index']}: '{clip['title']}'", flush=True)
                
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
            
            # Step 5: Complete processing
            progress.update(90, "Finalizing clips...")
            print(f"✅ Processing complete for job {job_id}", flush=True)
            
            # Clips already have titles and virality scores from viral detector
            for c in uploaded_clips:
                # Ensure title and virality score are included in the uploaded clip data
                c["title"] = next((clip["title"] for clip in clips if clip["index"] == c["index"]), "Untitled")
                c["virality_score"] = next((clip["virality_score"] for clip in clips if clip["index"] == c["index"]), 0.0)
                c["description"] = None  # nullable string
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