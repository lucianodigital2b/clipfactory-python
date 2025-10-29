#!/usr/bin/env python3
"""
Test script for transcriber.py functions
Tests both Groq and faster-whisper transcription methods
"""

import os
import sys
import json
import tempfile
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from services.transcriber import transcribe_with_faster_whisper, transcribe_with_groq, transcribe_video

def test_faster_whisper_transcription():
    """Test faster-whisper transcription method"""
    print("\n" + "="*60)
    print("🧪 Testing faster-whisper transcription")
    print("="*60)
    
    sample_video = os.path.join(project_root, "sample-files", "sample-video-2.mp4")
    
    if not os.path.exists(sample_video):
        print(f"❌ Sample video not found: {sample_video}")
        return False
    
    print(f"📁 Using sample video: {sample_video}")
    print(f"📊 File size: {os.path.getsize(sample_video)} bytes")
    
    try:
        print("🎤 Starting faster-whisper transcription...")
        start_time = datetime.now()
        
        transcript = transcribe_with_faster_whisper(sample_video)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"⏱️ Transcription completed in {duration:.2f} seconds")
        print(f"📝 Generated {len(transcript)} transcript segments")
        
        # Display transcript segments
        print("\n📄 Transcript segments:")
        for i, segment in enumerate(transcript[:5]):  # Show first 5 segments
            print(f"  {i+1}. [{segment['start']:.2f}s - {segment['end']:.2f}s]: {segment['text']}")
        
        if len(transcript) > 5:
            print(f"  ... and {len(transcript) - 5} more segments")
        
        # Validate transcript structure
        for segment in transcript:
            if not all(key in segment for key in ['text', 'start', 'end']):
                print("❌ Invalid segment structure")
                return False
            if not isinstance(segment['start'], (int, float)) or not isinstance(segment['end'], (int, float)):
                print("❌ Invalid timestamp types")
                return False
        
        print("✅ faster-whisper transcription test passed!")
        return True
        
    except Exception as e:
        print(f"❌ faster-whisper transcription failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_groq_transcription():
    """Test Groq API transcription method"""
    print("\n" + "="*60)
    print("🧪 Testing Groq API transcription")
    print("="*60)
    
    # Check if Groq API key is available
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("⚠️ GROQ_API_KEY not found in environment variables")
        print("💡 Skipping Groq test - set GROQ_API_KEY to test this method")
        return True  # Return True since this is expected behavior
    
    sample_video = os.path.join(project_root, "sample-files", "sample-video.mp4")
    
    if not os.path.exists(sample_video):
        print(f"❌ Sample video not found: {sample_video}")
        return False
    
    print(f"📁 Using sample video: {sample_video}")
    print(f"🔑 Groq API key found (length: {len(api_key)} chars)")
    
    try:
        print("🎤 Starting Groq API transcription...")
        start_time = datetime.now()
        
        transcript = transcribe_with_groq(sample_video)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"⏱️ Transcription completed in {duration:.2f} seconds")
        print(f"📝 Generated {len(transcript)} transcript segments")
        
        # Display transcript segments
        print("\n📄 Transcript segments:")
        for i, segment in enumerate(transcript[:5]):  # Show first 5 segments
            print(f"  {i+1}. [{segment['start']:.2f}s - {segment['end']:.2f}s]: {segment['text']}")
        
        if len(transcript) > 5:
            print(f"  ... and {len(transcript) - 5} more segments")
        
        # Validate transcript structure
        for segment in transcript:
            if not all(key in segment for key in ['text', 'start', 'end']):
                print("❌ Invalid segment structure")
                return False
        
        print("✅ Groq API transcription test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Groq API transcription failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_transcribe_video_function():
    """Test the main transcribe_video function with different methods"""
    print("\n" + "="*60)
    print("🧪 Testing transcribe_video function")
    print("="*60)
    
    sample_video = os.path.join(project_root, "sample-files", "sample-video.mp4")
    
    if not os.path.exists(sample_video):
        print(f"❌ Sample video not found: {sample_video}")
        return False
    
    # Test with faster-whisper method
    try:
        print("🎤 Testing transcribe_video with faster-whisper method...")
        transcript = transcribe_video(sample_video, method="faster-whisper")
        
        print(f"📝 Generated {len(transcript)} segments with faster-whisper")
        print("✅ transcribe_video (faster-whisper) test passed!")
        
    except Exception as e:
        print(f"❌ transcribe_video (faster-whisper) failed: {e}")
        return False
    
    # Test with Groq method if API key is available
    if os.getenv("GROQ_API_KEY"):
        try:
            print("🎤 Testing transcribe_video with Groq method...")
            transcript = transcribe_video(sample_video, method="groq")
            
            print(f"📝 Generated {len(transcript)} segments with Groq")
            print("✅ transcribe_video (groq) test passed!")
            
        except Exception as e:
            print(f"❌ transcribe_video (groq) failed: {e}")
            return False
    else:
        print("⚠️ Skipping Groq method test - GROQ_API_KEY not available")
    
    # Test invalid method
    try:
        print("🧪 Testing invalid method handling...")
        transcribe_video(sample_video, method="invalid_method")
        print("❌ Should have raised ValueError for invalid method")
        return False
    except ValueError as e:
        print(f"✅ Correctly raised ValueError: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    
    return True

def test_transcribe_video_with_upload():
    """Test transcribe_video function with R2 upload simulation"""
    print("\n" + "="*60)
    print("🧪 Testing transcribe_video with video_id (R2 upload)")
    print("="*60)
    
    sample_video = os.path.join(project_root, "sample-files", "sample-video.mp4")
    
    if not os.path.exists(sample_video):
        print(f"❌ Sample video not found: {sample_video}")
        return False
    
    try:
        print("🎤 Testing transcribe_video with video_id parameter...")
        # Note: This will attempt R2 upload, which may fail if R2 credentials aren't configured
        # That's expected behavior for testing
        transcript = transcribe_video(
            sample_video, 
            method="faster-whisper", 
            video_id="test_video_123"
        )
        
        print(f"📝 Generated {len(transcript)} segments")
        print("✅ transcribe_video with video_id test completed!")
        print("💡 Note: R2 upload may have failed if credentials not configured - this is expected")
        
        return True
        
    except Exception as e:
        print(f"❌ transcribe_video with video_id failed: {e}")
        return False

def run_all_tests():
    """Run all transcriber tests"""
    print("🚀 Starting transcriber.py test suite")
    print(f"📁 Project root: {project_root}")
    
    tests = [
        ("faster-whisper transcription", test_faster_whisper_transcription),
        ("Groq API transcription", test_groq_transcription),
        ("transcribe_video function", test_transcribe_video_function),
        ("transcribe_video with upload", test_transcribe_video_with_upload)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST RESULTS SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("💥 Some tests failed!")
        return False

if __name__ == "__main__":
    print("=" * 80)
    print("🧪 TRANSCRIBER.PY TEST SUITE")
    print("=" * 80)
    
    success = run_all_tests()
    
    print("\n" + "=" * 80)
    if success:
        print("🎉 Test suite completed successfully!")
    else:
        print("💥 Test suite completed with failures!")
    print("=" * 80)
    
    sys.exit(0 if success else 1)