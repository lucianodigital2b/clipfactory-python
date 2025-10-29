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

# Configuration constants
VIRAL_MOMENT_OVERLAP_THRESHOLD = 0.3  # 30% overlap required
FACE_DETECTION_CONFIDENCE = 0.6
FACE_DETECTION_FPS_BATCH = 5

def generate_title_from_transcript(transcript_text):
    """Generate a meaningful title from transcript text"""
    if not transcript_text or not transcript_text.strip():
        return "Untitled Clip"
    
    text = transcript_text.strip()
    text = re.sub(r'\b(um|uh|like|you know|so|well|actually|basically)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'^[,\s]+', '', text)
    text = re.sub(r'[,\s]+$', '', text)
    
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    if not sentences:
        words = text.split()[:6]
        if words:
            title = ' '.join(words)
            if not title.endswith(('!', '?', '.')):
                title += '...'
            return title.capitalize()
        return "Untitled Clip"
    
    title = sentences[0].strip()
    
    if title:
        title = title[0].upper() + title[1:] if len(title) > 1 else title.upper()
    
    if len(title) > 60:
        title = title[:57] + "..."
    
    if title and not title.endswith(('!', '?', '.')):
        if any(word in title.lower() for word in ['what', 'how', 'why', 'when', 'where', 'who']):
            title += '?'
        elif any(word in title.lower() for word in ['amazing', 'incredible', 'wow', 'shocking', 'unbelievable']):
            title += '!'
        else:
            title += '.'
    
    return title

def match_viral_moment_to_clip(clip, viral_moments):
    """
    Match a viral moment to a clip based on percentage overlap.
    Returns (viral_moment, overlap_percentage) or (None, 0)
    """
    if not viral_moments:
        return None, 0
    
    clip_start = clip["start_time"]
    clip_end = clip["end_time"]
    clip_duration = clip_end - clip_start
    
    best_match = None
    best_overlap_pct = 0
    
    for moment in viral_moments:
        moment_start = moment.get("start_time", 0)
        moment_end = moment.get("end_time", 0)
        
        # Calculate overlap
        overlap_start = max(clip_start, moment_start)
        overlap_end = min(clip_end, moment_end)
        overlap_duration = max(0, overlap_end - overlap_start)
        
        # Calculate percentage of clip covered by this viral moment
        overlap_percentage = overlap_duration / clip_duration if clip_duration > 0 else 0
        
        if overlap_percentage > best_overlap_pct:
            best_overlap_pct = overlap_percentage
            best_match = moment
    
    # Require threshold overlap to consider it a match
    if best_match and best_overlap_pct >= VIRAL_MOMENT_OVERLAP_THRESHOLD:
        return best_match, best_overlap_pct
    
    return None, 0

def apply_smart_focus_and_aspect_ratio(clips, original_video_path, video_output_dir, 
                                       aspect_ratio, enable_face_focus=True, resolution="360p"):
    """
    Apply smart camera focus and aspect ratio conversion in one integrated step.
    This ensures face detection happens on the original video before cropping.
    
    Args:
        clips: List of clip dictionaries from generate_clips
        original_video_path: Path to the original full video
        video_output_dir: Directory to save processed clips
        aspect_ratio: Target aspect ratio (e.g., "9:16")
        enable_face_focus: Whether to enable face detection
        resolution: Target resolution (e.g., "360p", "720p", "1080p")
    
    Returns:
        List of processed clips with face focus and aspect ratio applied
    """
    processed_clips = []
    clips_to_delete = []  # Track originals for safer deletion
    
    for i, clip in enumerate(clips):
        try:
            clip_filename = os.path.basename(clip["video"])
            processed_filename = f"processed_{clip_filename}"
            processed_path = os.path.join(video_output_dir, processed_filename)
            
            print(f"🎯 Processing clip {i+1}/{len(clips)}: {clip_filename}", flush=True)
            
            face_detected = False
            
            # Apply face focus if enabled and aspect ratio is vertical (9:16)
            if enable_face_focus and aspect_ratio == "9:16":
                try:
                    print(f"  👤 Detecting faces and applying smart focus...", flush=True)
                    
                    # Apply face focus on the ORIGINAL video segment
                    # This detects faces and crops to 9:16 centered on the speaker
                    face_detected = focus_on_speaker(
                        video_path=original_video_path,  # Original full video
                        output_path=processed_path,
                        start_time=clip["start_time"],
                        duration=clip["end_time"] - clip["start_time"],
                        target_aspect_ratio=aspect_ratio,  # Crop to 9:16 around face
                        min_confidence=FACE_DETECTION_CONFIDENCE,
                        fps_batch=FACE_DETECTION_FPS_BATCH,
                        resolution=resolution  # Pass resolution parameter
                    )
                    
                    if face_detected and os.path.exists(processed_path) and os.path.getsize(processed_path) > 0:
                        print(f"  ✅ Face focus applied successfully", flush=True)
                        clips_to_delete.append(clip["video"])
                    else:
                        print(f"  ℹ️ No faces detected with sufficient confidence", flush=True)
                        face_detected = False
                        
                except Exception as e:
                    print(f"  ⚠️ Face focus failed: {e}", flush=True)
                    face_detected = False
            
            # If face focus wasn't applied or failed, apply standard aspect ratio conversion
            if not face_detected:
                if aspect_ratio and aspect_ratio != "original":
                    try:
                        from services.aspect_ratio_converter import convert_aspect_ratio
                        
                        print(f"  📐 Applying standard {aspect_ratio} crop (centered)...", flush=True)
                        convert_aspect_ratio(
                            input_path=clip["video"],
                            output_path=processed_path,
                            target_ratio=aspect_ratio,
                            resolution=resolution  # Pass resolution parameter
                        )
                        
                        if os.path.exists(processed_path) and os.path.getsize(processed_path) > 0:
                            clips_to_delete.append(clip["video"])
                        else:
                            # If conversion failed, keep original
                            processed_path = clip["video"]
                            
                    except Exception as e:
                        print(f"  ⚠️ Aspect ratio conversion failed: {e}, using original", flush=True)
                        processed_path = clip["video"]
                else:
                    # No aspect ratio conversion needed
                    processed_path = clip["video"]
            
            # Update clip with processed video
            processed_clip = clip.copy()
            processed_clip["video"] = processed_path
            processed_clip["has_face_focus"] = face_detected
            processed_clips.append(processed_clip)
            
            print(f"  ✅ Clip {i+1}/{len(clips)} processed successfully", flush=True)
            
        except Exception as e:
            print(f"⚠️ Warning: Processing failed for clip {i+1}: {e}", flush=True)
            # Use original clip if processing fails
            clip["has_face_focus"] = False
            processed_clips.append(clip)
    
    # Safely delete original clips only after all processing is complete
    for original_clip_path in clips_to_delete:
        try:
            if os.path.exists(original_clip_path):
                os.remove(original_clip_path)
                print(f"🗑️ Deleted original clip: {os.path.basename(original_clip_path)}", flush=True)
        except Exception as e:
            print(f"⚠️ Warning: Could not delete {original_clip_path}: {e}", flush=True)
    
    print(f"✅ All clips processed: {sum(1 for c in processed_clips if c.get('has_face_focus'))} with face focus, "
          f"{len(processed_clips) - sum(1 for c in processed_clips if c.get('has_face_focus'))} with standard crop", flush=True)
    
    return processed_clips

def assign_titles_and_scores(clips, viral_moments):
    """
    Assign titles and virality scores to clips based on viral moment matching.
    Modifies clips in-place.
    """
    for clip in clips:
        # Match viral moment
        viral_moment, overlap_pct = match_viral_moment_to_clip(clip, viral_moments)
        
        if viral_moment:
            clip["title"] = viral_moment.get("title", "Untitled")
            clip["virality_score"] = viral_moment.get("virality_score", 0.5)
            clip["has_viral_match"] = True
            print(f"🔥 Clip {clip['index']}: '{clip['title']}' (virality: {clip['virality_score']:.2f}, overlap: {overlap_pct:.0%})", flush=True)
        else:
            # Fallback to transcript-based title
            clip_transcript = clip.get("transcript_text", "").strip()
            if clip_transcript:
                clip["title"] = generate_title_from_transcript(clip_transcript)
                clip["virality_score"] = 0.3
                clip["has_viral_match"] = False
                print(f"📝 Clip {clip['index']}: '{clip['title']}' (generated from transcript)", flush=True)
            else:
                clip["title"] = f"Clip {clip['index']}"
                clip["virality_score"] = 0.1
                clip["has_viral_match"] = False
                print(f"📝 Clip {clip['index']}: Using fallback title", flush=True)

def process_video_async(job_id: str, job_data: dict):
    """Process video asynchronously in background thread"""
    print(f"🎬 Background thread started for job {job_id}", flush=True)
    
    class JobProgressReporter:
        def __init__(self, job_id, webhook_url, video_id=None):
            self.job_id = job_id
            self.webhook_url = webhook_url
            self.video_id = video_id
            self.original_reporter = ProgressReporter(webhook_url) if webhook_url else None
        
        def update(self, progress, message, **kwargs):
            job_manager.update_progress(self.job_id, progress, message)
            if self.original_reporter:
                update_kwargs = {"video_id": self.video_id} if self.video_id else {}
                update_kwargs.update(kwargs)
                self.original_reporter.update(progress, message, **update_kwargs)
    
    progress = None
    
    try:
        video_url = job_data["video_url"]
        video_id = job_data.get("video_id")
        if video_id is not None:
            video_id = str(video_id)
        
        clip_duration = job_data.get("clip_duration", "60")
        webhook_url = job_data.get("webhook_url")
        platform = job_data.get("platform", "YouTube")
        transcription_method = job_data.get("transcription_method", "faster-whisper")
        aspect_ratio = job_data.get("aspect_ratio", "9:16")
        enable_face_focus = job_data.get("enable_face_focus", True)
        resolution = job_data.get("resolution", "360p")  # Default to 360p
        
        # Support both field names for transcription
        transcription = job_data.get("transcription") or job_data.get("subtitles_path")

        print(f"🔄 Processing job {job_id}", flush=True)
        print(f"  - URL: {video_url}", flush=True)
        print(f"  - Aspect ratio: {aspect_ratio}", flush=True)
        print(f"  - Face focus: {enable_face_focus}", flush=True)
        print(f"  - Resolution: {resolution}", flush=True)
        
        progress = JobProgressReporter(job_id, webhook_url, video_id)
        job_manager.update_job_status(job_id, JobStatus.PROCESSING)
        progress.update(0, "Starting video processing...")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Step 1: Download video (0-15%)
            progress.update(5, "Downloading video...")
            print(f"📥 Downloading video", flush=True)
            local_path = download_video(video_url, tmpdir)
            progress.update(15, "Download complete")
            
            # Step 2: Handle transcription (15-40%)
            if transcription:
                progress.update(20, "Using provided transcription...")
                print(f"📝 Using pre-existing transcription", flush=True)
                transcript = transcription
            else:
                progress.update(20, f"Transcribing video ({transcription_method})...")
                print(f"🎤 Transcribing video", flush=True)
                transcript = transcribe_video(local_path, method=transcription_method, 
                                            video_id=video_id, progress=progress)
            
            print(f"📝 Transcript ready: {len(transcript)} segments", flush=True)
            progress.update(40, "Transcription complete")
            
            # Step 3: Extract viral moments (40-50%)
            progress.update(45, "Analyzing viral moments...")
            print(f"🔥 Extracting viral moments", flush=True)
            
            viral_moments = []
            try:
                viral_moments = extract_viral_moments(transcript)
                print(f"🔥 Found {len(viral_moments)} viral moments", flush=True)
            except Exception as e:
                print(f"⚠️ Viral moment extraction failed: {e}", flush=True)
            
            progress.update(50, f"Found {len(viral_moments)} viral moments")
            
            # Step 4: Generate clips WITHOUT aspect ratio (50-60%)
            progress.update(55, "Generating clips...")
            print(f"✂️ Generating clips", flush=True)
            
            if not video_id:
                video_id = video_url.split('/')[-1].split('?')[0].split('=')[-1] if 'youtube.com' in video_url or 'youtu.be' in video_url else job_id
            video_id = str(video_id)
            
            video_output_dir = os.path.join(tmpdir, video_id)
            os.makedirs(video_output_dir, exist_ok=True)
            
            clips = generate_clips(
                video_path=local_path, 
                clip_duration=clip_duration, 
                transcript=transcript, 
                output_dir=video_output_dir,
                platform=platform,
                source_id=video_id,
                aspect_ratio=None  # Don't apply aspect ratio yet - will be done with face focus
            )
            print(f"✂️ Generated {len(clips)} raw clips", flush=True)
            progress.update(60, f"Generated {len(clips)} clips")
            
            # Step 5: Apply face focus + aspect ratio (60-75%)
            progress.update(65, "Applying smart focus and aspect ratio...")
            print(f"🎯 Applying smart focus and aspect ratio conversion", flush=True)
            
            clips = apply_smart_focus_and_aspect_ratio(
                clips=clips,
                original_video_path=local_path,
                video_output_dir=video_output_dir,
                aspect_ratio=aspect_ratio,
                enable_face_focus=enable_face_focus,
                resolution=resolution  # Pass resolution parameter
            )
            
            progress.update(75, "Focus and aspect ratio applied")
            
            # Step 6: Assign titles and scores (75-80%)
            progress.update(77, "Assigning titles and scores...")
            print(f"📝 Assigning titles and virality scores", flush=True)
            assign_titles_and_scores(clips, viral_moments)
            progress.update(80, "Titles assigned")
            
            # Step 7: Upload clips (80-95%)
            progress.update(82, "Uploading clips...")
            print(f"☁️ Uploading {len(clips)} clips", flush=True)
            
            uploaded_clips = []
            for i, clip in enumerate(clips):
                video_url_uploaded = upload_to_r2(clip["video"], video_id)
                srt_url = upload_to_r2(clip["srt"], video_id)
                
                uploaded_clips.append({
                    "index": clip["index"],
                    "clip_path": video_url_uploaded,
                    "start": clip["start_time"],
                    "end": clip["end_time"],
                    "transcript": clip["transcript_text"],
                    "subtitles_path": srt_url,
                    "title": clip.get("title", "Untitled"),
                    "virality_score": clip.get("virality_score", 0.0),
                    "has_viral_match": clip.get("has_viral_match", False),
                    "has_face_focus": clip.get("has_face_focus", False),
                    "description": None,
                    "thumbnail_path": None
                })
                
                # Update progress for each upload
                upload_progress = 82 + int((i + 1) / len(clips) * 13)  # 82-95%
                progress.update(upload_progress, f"Uploaded {i+1}/{len(clips)} clips")
            
            # Step 8: Complete (95-100%)
            progress.update(95, "Finalizing...")
            print(f"✅ Processing complete", flush=True)
            
            # Calculate statistics
            face_focus_count = sum(1 for c in uploaded_clips if c.get("has_face_focus"))
            viral_match_count = sum(1 for c in uploaded_clips if c.get("has_viral_match"))
            
            result = {
                "status": "success",
                "clips": uploaded_clips,
                "total_clips": len(uploaded_clips),
                "clips_with_face_focus": face_focus_count,
                "clips_with_viral_match": viral_match_count,
                "viral_moments_found": len(viral_moments)
            }
            
            progress.update(100, "Processing complete!", clips=uploaded_clips, total_clips=len(uploaded_clips))
            print(f"✅ Job {job_id} completed: {len(uploaded_clips)} clips ({face_focus_count} with face focus, {viral_match_count} with viral match)", flush=True)
            
            job_manager.set_job_result(job_id, result)
            
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Job {job_id} failed: {error_msg}", flush=True)
        
        import traceback
        print(f"❌ Traceback: {traceback.format_exc()}", flush=True)
        
        if progress is not None:
            progress.update(0, "Processing failed", error_message=error_msg)
        
        job_manager.set_job_error(job_id, error_msg)

def start_video_processing(job_data: dict) -> str:
    """Start video processing in background thread and return job ID"""
    job_id = job_manager.create_job(job_data)
    
    print(f"🚀 Creating job {job_id}", flush=True)
    
    thread = threading.Thread(
        target=process_video_async,
        args=(job_id, job_data),
        daemon=True
    )
    thread.start()
    
    print(f"✅ Job {job_id} thread started", flush=True)
    
    return job_id