import os
import tempfile
import json
from faster_whisper import WhisperModel
from groq import Groq

def transcribe_with_groq(video_path):
    """
    Transcribe video using Groq API
    Requires GROQ_API_KEY environment variable
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is required for Groq transcription")
    
    client = Groq(api_key=api_key)
    
    # Open the audio file
    with open(video_path, "rb") as file:
        # Create transcription using Groq API
        transcription = client.audio.transcriptions.create(
            file=(video_path, file.read()),
            model="whisper-large-v3",  # Groq's Whisper model
            response_format="verbose_json",  # Get timestamps
            temperature=0.0
        )
    
    # Convert Groq response to our standard format
    transcript = []
    if hasattr(transcription, 'segments') and transcription.segments:
        for seg in transcription.segments:
            transcript.append({
                "text": seg.text.strip(),
                "start": seg.start,
                "end": seg.end
            })
    else:
        # Fallback if segments not available - create single segment
        transcript.append({
            "text": transcription.text.strip(),
            "start": 0.0,
            "end": 0.0  # We don't have timing info in this case
        })
    
    return transcript

def transcribe_with_faster_whisper(video_path):
    """
    Transcribe video using faster-whisper (local processing)
    """
    model = WhisperModel("base")
    segments, _ = model.transcribe(video_path, word_timestamps=True)
    transcript = []
    for seg in segments:
        transcript.append({
            "text": seg.text.strip(),
            "start": seg.start,
            "end": seg.end
        })
    return transcript

def transcribe_video(video_path, method="faster-whisper", video_id=None, progress=None):
    """
    Transcribe video using specified method and optionally upload to R2
    
    Args:
        video_path (str): Path to the video file
        method (str): Transcription method - "groq" or "faster-whisper" (default)
        video_id (str): Video ID for R2 upload path (optional)
        progress: Progress reporter for webhook updates (optional)
    
    Returns:
        list: List of transcript segments with text, start, and end times
    
    Raises:
        ValueError: If invalid method specified or missing API key for Groq
        Exception: If transcription fails
    """
    # Perform transcription
    if method == "groq":
        transcript = transcribe_with_groq(video_path)
    elif method == "faster-whisper":
        transcript = transcribe_with_faster_whisper(video_path)
    else:
        raise ValueError(f"Invalid transcription method: {method}. Use 'groq' or 'faster-whisper'")
    
    # Upload transcription to R2 if video_id is provided
    if video_id and transcript:
        try:
            # Import here to avoid circular imports
            from services.uploader import upload_to_r2
            
            # Create temporary file with transcription JSON
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
                json.dump(transcript, temp_file, indent=2)
                temp_transcript_path = temp_file.name
            
            try:
                # Upload transcription to R2
                transcript_url = upload_to_r2(temp_transcript_path, video_id)
                print(f"📤 Transcription uploaded to R2: {transcript_url}", flush=True)
                
                # Send webhook update with transcription URL
                if progress:
                    progress.update(35, "Transcription uploaded to R2", transcription_url=transcript_url)
                    
            finally:
                # Clean up temporary file
                os.unlink(temp_transcript_path)
                
        except Exception as upload_error:
            print(f"⚠️ Warning: Failed to upload transcription to R2: {upload_error}", flush=True)
    
    return transcript
