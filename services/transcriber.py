import os
import tempfile
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

def transcribe_video(video_path, method="faster-whisper"):
    """
    Transcribe video using specified method
    
    Args:
        video_path (str): Path to the video file
        method (str): Transcription method - "groq" or "faster-whisper" (default)
    
    Returns:
        list: List of transcript segments with text, start, and end times
    
    Raises:
        ValueError: If invalid method specified or missing API key for Groq
        Exception: If transcription fails
    """
    if method == "groq":
        return transcribe_with_groq(video_path)
    elif method == "faster-whisper":
        return transcribe_with_faster_whisper(video_path)
    else:
        raise ValueError(f"Invalid transcription method: {method}. Use 'groq' or 'faster-whisper'")
