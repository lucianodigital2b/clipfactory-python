#!/usr/bin/env python3
"""
Test script for speaker detection using focus_on_speaker in frame_focus.py
Uses the provided sample video in the tests directory to verify detection.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from services.frame_focus import focus_on_speaker


def test_speaker_detection():
    """Run speaker detection on the sample video and validate output"""
    print("🧪 Starting speaker detection test...")

    tests_dir = Path(__file__).resolve().parent/'sample-files'
    video_file = tests_dir / "sample-video.mp4"

    if not video_file.exists():
        print(f"❌ Sample video not found: {video_file}")
        return False

    # Output path for the focused clip
    output_path = tests_dir / "focused_test_output.mp4"
    if output_path.exists():
        try:
            output_path.unlink()
        except Exception:
            pass

    # Use a short segment to keep the test fast
    start_time = 0  # start at 1:00 to skip any intro
    duration = 60    # process 15 seconds

    print("🎬 Running focus_on_speaker...")
    print(f"   Input: {video_file}")
    print(f"   Output: {output_path}")
    print(f"   Clip: start={start_time}s, duration={duration}s")

    try:
        result = focus_on_speaker(
            video_path=str(video_file),
            output_path=str(output_path),
            start_time=start_time,
            duration=duration,
            target_aspect_ratio="16:9",
            min_confidence=0.5,
            fps_batch=10,
            resolution="2160p"  # Test with 360p resolution
        )

        print(f"✅ focus_on_speaker returned: {result}")

        # Validate output
        if not result:
            print("❌ Speaker detection indicated failure (no face focus)")
            return False

        if not output_path.exists() or output_path.stat().st_size == 0:
            print("❌ Output video not created or empty")
            return False

        print(f"📁 Output file created: {output_path} ({output_path.stat().st_size} bytes)")
        print("✅ Speaker detection test passed!")
        return True

    except Exception as e:
        print(f"❌ Exception occurred during speaker detection: {e}")
        import traceback
        print(traceback.format_exc())
        return False


if __name__ == "__main__":
    print("🚀 Starting frame_focus.py speaker detection test")
    print("=" * 50)
    success = test_speaker_detection()
    print("=" * 50)
    if success:
        print("🎉 Test passed!")
        sys.exit(0)
    else:
        print("❌ Test failed!")
        sys.exit(1)