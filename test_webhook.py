#!/usr/bin/env python3
"""
Test script for webhook functionality with Laravel-compatible payload format
"""

import requests
import urllib3
from services.progress import ProgressReporter

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def test_webhook_url():
    """Test basic webhook connectivity"""
    webhook_url = "https://clipfactory.test/api/clip/update"
    
    print("Testing webhook URL connectivity...")
    try:
        response = requests.get(webhook_url, verify=False, timeout=5)
        print(f"Webhook URL response: {response.status_code}")
        return True
    except Exception as e:
        print(f"Webhook URL test failed: {e}")
        return False

def test_progress_updates():
    """Test progress updates with Laravel-compatible format"""
    webhook_url = "https://clipfactory.test/api/clip/update"
    
    print("Testing progress updates...")
    progress_reporter = ProgressReporter(webhook_url)
    
    # Test processing status
    progress_reporter.update(25, "Starting video processing", video_id=22)
    
    # Test mid-progress
    progress_reporter.update(50, "Generating clips", video_id=22)
    
    # Test near completion
    progress_reporter.update(90, "Uploading clips", video_id=22)

def test_final_completion_webhook():
    """Test final completion webhook with full Laravel-compatible payload"""
    webhook_url = "https://clipfactory.test/api/clip/update"
    
    print("Testing final completion webhook...")
    
    # Sample clips data matching Laravel validator requirements
    sample_clips = [
        {
            "index": 1,
            "clip_path": "https://r2.example.com/video_22/clip_1.mp4",
            "start": 0.0,
            "end": 30.0,
            "transcript": "This is the first clip transcript text",
            "title": "Amazing Moment #1",
            "description": None,
            "virality_score": None,
            "thumbnail_path": None,
            "subtitles_path": "https://r2.example.com/video_22/clip_1.srt"
        },
        {
            "index": 2,
            "clip_path": "https://r2.example.com/video_22/clip_2.mp4",
            "start": 30.0,
            "end": 60.0,
            "transcript": "This is the second clip transcript text",
            "title": "Incredible Scene #2",
            "description": None,
            "virality_score": None,
            "thumbnail_path": None,
            "subtitles_path": "https://r2.example.com/video_22/clip_2.srt"
        }
    ]
    
    progress_reporter = ProgressReporter(webhook_url)
    progress_reporter.update(
        100, 
        "Video processing completed successfully",
        video_id=22,
        clips=sample_clips
    )

def test_error_webhook():
    """Test error webhook with Laravel-compatible format"""
    webhook_url = "https://clipfactory.test/api/clip/update"
    
    print("Testing error webhook...")
    progress_reporter = ProgressReporter(webhook_url)
    progress_reporter.update(
        0, 
        "Processing failed",
        video_id=22,
        error_message="Sample error: Video download failed"
    )

if __name__ == "__main__":
    print("Starting Laravel-compatible webhook tests...")
    print("=" * 50)
    
    # Test 1: Basic connectivity
    test_webhook_url()
    print()
    
    # Test 2: Progress updates
    test_progress_updates()
    print()
    
    # Test 3: Final completion
    test_final_completion_webhook()
    print()
    
    # Test 4: Error handling
    test_error_webhook()
    print()
    
    print("All webhook tests completed!")