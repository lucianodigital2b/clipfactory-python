#!/usr/bin/env python3
"""
YouTube Video Transcription Test
Downloads, transcribes, and uploads a YouTube video transcription to R2 bucket
"""

import os
import sys
import tempfile
import json
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from services.downloader import download_video
from services.transcriber import transcribe_video
from services.uploader import upload_to_r2

def test_youtube_transcription_pipeline():
    """Test the complete pipeline: download -> transcribe -> upload"""
    print("\n" + "="*80)
    print("🎬 YOUTUBE VIDEO TRANSCRIPTION PIPELINE TEST")
    print("="*80)
    
    # YouTube video URL
    video_url = "https://www.youtube.com/watch?v=vuFvFKBvu_U"
    video_id = "vuFvFKBvu_U"
    
    print(f"🎯 Target video: {video_url}")
    print(f"📋 Video ID: {video_id}")
    
    # Check R2 environment variables
    print("\n🔍 Checking R2 environment variables...")
    r2_vars = {
        'R2_ACCESS_KEY': os.getenv('R2_ACCESS_KEY'),
        'R2_SECRET_KEY': os.getenv('R2_SECRET_KEY'), 
        'R2_BUCKET': os.getenv('R2_BUCKET'),
        'R2_ENDPOINT': os.getenv('R2_ENDPOINT'),
        'R2_PUBLIC_URL': os.getenv('R2_PUBLIC_URL')
    }
    
    missing_vars = [var for var, val in r2_vars.items() if not val]
    if missing_vars:
        print(f"⚠️ Missing R2 environment variables: {', '.join(missing_vars)}")
        print("💡 Upload step will fail, but download and transcription will work")
    else:
        print("✅ All R2 environment variables are configured")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            # Step 1: Download video
            print(f"\n📥 Step 1: Downloading video...")
            print(f"   URL: {video_url}")
            print(f"   Temp dir: {tmpdir}")
            
            start_time = datetime.now()
            local_video_path = download_video(video_url, tmpdir)
            download_duration = (datetime.now() - start_time).total_seconds()
            
            if not os.path.exists(local_video_path):
                raise Exception(f"Downloaded video not found: {local_video_path}")
            
            video_size = os.path.getsize(local_video_path)
            print(f"✅ Download completed in {download_duration:.1f}s")
            print(f"   File: {local_video_path}")
            print(f"   Size: {video_size:,} bytes ({video_size / (1024*1024):.1f} MB)")
            
            # Step 2: Transcribe video
            print(f"\n🎤 Step 2: Transcribing video...")
            print(f"   Method: faster-whisper")
            print(f"   Video ID: {video_id}")
            
            start_time = datetime.now()
            transcript = transcribe_video(
                video_path=local_video_path,
                method="faster-whisper",
                video_id=video_id
            )
            transcription_duration = (datetime.now() - start_time).total_seconds()
            
            if not transcript:
                raise Exception("Transcription returned empty result")
            
            print(f"✅ Transcription completed in {transcription_duration:.1f}s")
            print(f"   Segments: {len(transcript)}")
            
            # Show sample transcript segments
            print(f"\n📄 Sample transcript segments:")
            for i, segment in enumerate(transcript[:3]):
                start_time = segment.get('start', 0)
                end_time = segment.get('end', 0)
                text = segment.get('text', '').strip()
                print(f"   {i+1}. [{start_time:.1f}s - {end_time:.1f}s]: {text[:80]}{'...' if len(text) > 80 else ''}")
            
            if len(transcript) > 3:
                print(f"   ... and {len(transcript) - 3} more segments")
            
            # Step 3: Save transcription to temporary file
            print(f"\n💾 Step 3: Preparing transcription for upload...")
            
            transcript_filename = f"{video_id}_transcript.json"
            transcript_path = os.path.join(tmpdir, transcript_filename)
            
            with open(transcript_path, 'w', encoding='utf-8') as f:
                json.dump(transcript, f, indent=2, ensure_ascii=False)
            
            transcript_size = os.path.getsize(transcript_path)
            print(f"✅ Transcription saved to temporary file")
            print(f"   File: {transcript_path}")
            print(f"   Size: {transcript_size:,} bytes ({transcript_size / 1024:.1f} KB)")
            
            # Step 4: Upload to R2 bucket
            print(f"\n☁️ Step 4: Uploading transcription to R2 bucket...")
            
            if missing_vars:
                print(f"❌ Skipping upload - missing R2 credentials: {', '.join(missing_vars)}")
                print(f"💡 To enable upload, configure these environment variables in .env file")
                return False
            
            try:
                start_time = datetime.now()
                transcript_url = upload_to_r2(transcript_path, video_id)
                upload_duration = (datetime.now() - start_time).total_seconds()
                
                print(f"✅ Upload completed in {upload_duration:.1f}s")
                print(f"   URL: {transcript_url}")
                
                # Step 5: Summary
                print(f"\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
                print(f"="*80)
                print(f"📊 Summary:")
                print(f"   Video URL: {video_url}")
                print(f"   Video ID: {video_id}")
                print(f"   Video Size: {video_size / (1024*1024):.1f} MB")
                print(f"   Download Time: {download_duration:.1f}s")
                print(f"   Transcript Segments: {len(transcript)}")
                print(f"   Transcription Time: {transcription_duration:.1f}s")
                print(f"   Transcript Size: {transcript_size / 1024:.1f} KB")
                print(f"   Upload Time: {upload_duration:.1f}s")
                print(f"   Final URL: {transcript_url}")
                print(f"="*80)
                
                return True
                
            except Exception as upload_error:
                print(f"❌ Upload failed: {upload_error}")
                print(f"💡 This is expected if R2 credentials are not properly configured")
                return False
                
        except Exception as e:
            print(f"❌ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def test_with_api_endpoint():
    """Test using the API endpoint for comparison"""
    print("\n" + "="*80)
    print("🌐 TESTING WITH API ENDPOINT")
    print("="*80)
    
    import requests
    
    api_url = "http://localhost:8000/process"
    video_url = "https://www.youtube.com/watch?v=vuFvFKBvu_U"
    
    payload = {
        "video_url": video_url,
        "video_id": "vuFvFKBvu_U",
        "clip_duration": "30",  # Shorter clips for testing
        "transcription_method": "faster-whisper",
        "aspect_ratio": "original",  # Skip aspect ratio conversion
        "enable_face_focus": False,  # Skip face focus for faster processing
        "resolution": "360p"
    }
    
    try:
        print(f"🚀 Sending request to API endpoint...")
        print(f"   URL: {api_url}")
        print(f"   Payload: {json.dumps(payload, indent=2)}")
        
        response = requests.post(api_url, json=payload, timeout=30)
        
        if response.status_code == 202:
            result = response.json()
            job_id = result.get("job_id")
            
            print(f"✅ API request accepted")
            print(f"   Job ID: {job_id}")
            print(f"   Status URL: http://localhost:8000/status/{job_id}")
            print(f"💡 Check the status URL to monitor progress")
            
            return True
        else:
            print(f"❌ API request failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Could not connect to API endpoint at {api_url}")
        print(f"💡 Make sure the Flask server is running: python app.py")
        return False
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 80)
    print("🧪 YOUTUBE TRANSCRIPTION TEST SUITE")
    print("=" * 80)
    
    # Test 1: Direct pipeline
    print("\n🔧 Test 1: Direct Pipeline (download -> transcribe -> upload)")
    pipeline_success = test_youtube_transcription_pipeline()
    
    # Test 2: API endpoint
    print("\n🔧 Test 2: API Endpoint")
    api_success = test_with_api_endpoint()
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 80)
    print(f"✅ Direct Pipeline: {'PASSED' if pipeline_success else 'FAILED'}")
    print(f"✅ API Endpoint: {'PASSED' if api_success else 'FAILED'}")
    
    if pipeline_success or api_success:
        print("🎉 At least one test method succeeded!")
    else:
        print("❌ All tests failed - check configuration and dependencies")
    
    print("=" * 80)