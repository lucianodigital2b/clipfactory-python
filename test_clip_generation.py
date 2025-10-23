#!/usr/bin/env python3
"""
Test script to debug clip generation issues
"""
import os
import tempfile
from dotenv import load_dotenv
from services.downloader import download_video
from services.transcriber import transcribe_video
from services.clipper import generate_clips
from utils.ffmpeg_utils import get_video_duration

# Load environment variables
load_dotenv()

def test_clip_generation():
    """Test the entire clip generation pipeline"""
    print("🧪 Starting clip generation test...")
    
    # Test video URL (short video for testing)
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Rick Roll - short video
    
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"📁 Using temp directory: {tmpdir}")
        
        try:
            # Step 1: Download video
            print("📥 Downloading video...")
            local_path = download_video(test_url, tmpdir)
            print(f"✅ Video downloaded to: {local_path}")
            
            # Step 2: Check video duration
            print("⏱️ Getting video duration...")
            duration = get_video_duration(local_path)
            print(f"✅ Video duration: {duration} seconds")
            
            # Step 3: Transcribe video (use faster-whisper for testing)
            print("🎤 Transcribing video...")
            transcript = transcribe_video(local_path, method="faster-whisper")
            print(f"✅ Transcript generated with {len(transcript)} segments")
            
            if transcript:
                print(f"📝 First segment: {transcript[0]}")
                print(f"📝 Last segment: {transcript[-1]}")
            
            # Step 4: Generate clips
            print("✂️ Generating clips...")
            output_dir = os.path.join(tmpdir, "clips")
            clips = generate_clips(
                video_path=local_path,
                clip_duration="30",  # 30 second clips for testing
                transcript=transcript,
                output_dir=output_dir,
                platform="YouTube",
                source_id="test_video"
            )
            
            print(f"🎬 Generated {len(clips)} clips")
            
            if clips:
                print("✅ Clip generation successful!")
                for i, clip in enumerate(clips):
                    print(f"  Clip {i+1}: {clip['video']} ({clip['start_time']}s - {clip['end_time']}s)")
            else:
                print("❌ No clips were generated!")
                
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_clip_generation()