#!/usr/bin/env python3
"""
Test script for the generate_clips function in clipper.py
This script creates a sample transcript and tests the clip generation functionality.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.clipper import generate_clips

def create_sample_transcript():
    """
    Create a sample whisper transcript of 2 minutes of conversation
    Format: [{"start": float, "end": float, "text": str}, ...]
    """
    transcript = [
        {"start": 0.0, "end": 3.5, "text": "Hello everyone, welcome to today's presentation about artificial intelligence."},
        {"start": 3.5, "end": 7.2, "text": "We're going to explore how AI is transforming various industries."},
        {"start": 7.2, "end": 11.8, "text": "First, let's talk about machine learning and its applications in healthcare."},
        {"start": 11.8, "end": 15.4, "text": "Machine learning algorithms can analyze medical images with incredible accuracy."},
        {"start": 15.4, "end": 19.1, "text": "They help doctors detect diseases earlier than traditional methods."},
        {"start": 19.1, "end": 23.7, "text": "Moving on to the financial sector, AI is revolutionizing fraud detection."},
        {"start": 23.7, "end": 27.3, "text": "Banks use sophisticated algorithms to identify suspicious transactions."},
        {"start": 27.3, "end": 31.0, "text": "This has significantly reduced financial crimes and improved security."},
        {"start": 31.0, "end": 35.6, "text": "In the automotive industry, self-driving cars are becoming a reality."},
        {"start": 35.6, "end": 39.2, "text": "These vehicles use computer vision and sensor fusion technologies."},
        {"start": 39.2, "end": 43.8, "text": "They can navigate complex traffic situations better than human drivers."},
        {"start": 43.8, "end": 47.5, "text": "Natural language processing is another exciting field in AI."},
        {"start": 47.5, "end": 51.1, "text": "Chatbots and virtual assistants are becoming more conversational."},
        {"start": 51.1, "end": 55.7, "text": "They can understand context and provide more helpful responses."},
        {"start": 55.7, "end": 59.4, "text": "Let's discuss the ethical implications of artificial intelligence."},
        {"start": 59.4, "end": 63.0, "text": "We need to ensure AI systems are fair and unbiased."},
        {"start": 63.0, "end": 67.6, "text": "Transparency in AI decision-making is crucial for public trust."},
        {"start": 67.6, "end": 71.2, "text": "Data privacy is another important consideration in AI development."},
        {"start": 71.2, "end": 75.8, "text": "Companies must protect user information while training AI models."},
        {"start": 75.8, "end": 79.5, "text": "The future of AI looks promising with continued research and innovation."},
        {"start": 79.5, "end": 83.1, "text": "We're seeing breakthroughs in quantum computing and neural networks."},
        {"start": 83.1, "end": 87.7, "text": "These advances will enable even more powerful AI applications."},
        {"start": 87.7, "end": 91.4, "text": "Education is also being transformed by artificial intelligence."},
        {"start": 91.4, "end": 95.0, "text": "Personalized learning platforms adapt to individual student needs."},
        {"start": 95.0, "end": 99.6, "text": "This helps students learn more effectively at their own pace."},
        {"start": 99.6, "end": 103.2, "text": "AI is also making significant contributions to climate science."},
        {"start": 103.2, "end": 107.8, "text": "It helps predict weather patterns and analyze environmental data."},
        {"start": 107.8, "end": 111.5, "text": "This information is crucial for addressing climate change challenges."},
        {"start": 111.5, "end": 115.1, "text": "In conclusion, AI is reshaping our world in remarkable ways."},
        {"start": 115.1, "end": 120.0, "text": "Thank you for your attention, and I hope you found this presentation informative."}
    ]
    return transcript

def create_test_video():
    """
    Create a simple test video using FFmpeg (2 minutes long)
    """
    video_path = "test_video.mp4"
    
    # Check if FFmpeg is available
    ffmpeg_path = os.getenv("FFMPEG_PATH", "ffmpeg")
    
    # Create a 2-minute test video with color bars and audio tone
    cmd = [
        ffmpeg_path,
        "-f", "lavfi",
        "-i", "testsrc2=duration=120:size=1280x720:rate=30",
        "-f", "lavfi", 
        "-i", "sine=frequency=1000:duration=120",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-t", "120",
        "-y",
        video_path
    ]
    
    print(f"🎬 Creating test video: {video_path}")
    print(f"🔧 Running command: {' '.join(cmd)}")
    
    import subprocess
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Failed to create test video: {result.stderr}")
        return None
    
    if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
        print(f"✅ Test video created successfully: {video_path}")
        return video_path
    else:
        print(f"❌ Test video creation failed - file not found or empty")
        return None

def test_generate_clips():
    """
    Test the generate_clips function
    """
    print("🧪 Starting generate_clips test...")
    
    # Create sample transcript
    print("📝 Creating sample transcript...")
    transcript = create_sample_transcript()
    print(f"✅ Created transcript with {len(transcript)} segments (total duration: {transcript[-1]['end']:.1f}s)")
    
    # Create test video
    print("🎬 Creating test video...")
    video_path = create_test_video()
    if not video_path:
        print("❌ Cannot proceed without test video")
        return False
    
    # Create temporary output directory
    output_dir = "test_clips_output"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        print(f"🎯 Testing generate_clips function...")
        print(f"   Video: {video_path}")
        print(f"   Clip duration: 30s")
        print(f"   Output directory: {output_dir}")
        
        # Test the function
        clips = generate_clips(
            video_path=video_path,
            clip_duration="30s",  # 30 second clips
            transcript=transcript,
            output_dir=output_dir,
            cache_file=os.path.join(output_dir, "cache.json"),
            platform="test",
            source_id="test_video"
        )
        
        print(f"🎉 generate_clips completed successfully!")
        print(f"📊 Generated {len(clips)} clips")
        
        # Verify results
        if len(clips) == 0:
            print("❌ No clips were generated!")
            return False
        
        print("\n📋 Clip details:")
        for i, clip in enumerate(clips):
            print(f"  Clip {i+1}:")
            print(f"    Index: {clip['index']}")
            print(f"    Video: {clip['video']}")
            print(f"    SRT: {clip['srt']}")
            print(f"    Start: {clip['start_time']}s")
            print(f"    End: {clip['end_time']}s")
            print(f"    Transcript length: {len(clip['transcript_text'])} chars")
            
            # Check if files exist
            video_exists = os.path.exists(clip['video'])
            srt_exists = os.path.exists(clip['srt'])
            video_size = os.path.getsize(clip['video']) if video_exists else 0
            
            print(f"    Video file exists: {video_exists} ({video_size} bytes)")
            print(f"    SRT file exists: {srt_exists}")
            
            if not video_exists or video_size == 0:
                print(f"    ❌ Video file issue!")
                return False
            if not srt_exists:
                print(f"    ❌ SRT file missing!")
                return False
        
        print(f"\n✅ All {len(clips)} clips generated successfully!")
        print(f"📁 Output directory: {os.path.abspath(output_dir)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        print(f"📋 Traceback: {traceback.format_exc()}")
        return False
    
    finally:
        # Cleanup
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
                print(f"🧹 Cleaned up test video: {video_path}")
            except:
                pass

if __name__ == "__main__":
    print("🚀 Starting clipper.py test suite")
    print("=" * 50)
    
    success = test_generate_clips()
    
    print("=" * 50)
    if success:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("❌ Tests failed!")
        sys.exit(1)