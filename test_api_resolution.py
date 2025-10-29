#!/usr/bin/env python3
"""
Test script to verify the resolution parameter works in the API endpoint
"""

import requests
import json
import time

def test_api_resolution():
    """Test the /process endpoint with resolution parameter"""
    
    # Test data with resolution parameter
    test_data = {
        "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Rick Roll for testing
        "clip_duration": "30",
        "platform": "YouTube",
        "transcription_method": "faster-whisper",
        "aspect_ratio": "9:16",
        "enable_face_focus": True,
        "resolution": "720p"  # Test with 720p instead of default 360p
    }
    
    print("🧪 Testing API endpoint with resolution parameter...")
    print(f"📝 Test data: {json.dumps(test_data, indent=2)}")
    
    try:
        # Make request to local API
        response = requests.post("http://localhost:8000/process", json=test_data)
        
        print(f"📡 Response status: {response.status_code}")
        print(f"📄 Response body: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 202:
            print("✅ API endpoint accepts resolution parameter successfully!")
            job_id = response.json().get("job_id")
            
            if job_id:
                print(f"🔄 Job ID: {job_id}")
                print("💡 You can check job status with: GET /status/{job_id}")
            
            return True
        else:
            print(f"❌ API request failed with status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API server. Make sure it's running on http://localhost:8000")
        return False
    except Exception as e:
        print(f"❌ Error testing API: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 API Resolution Parameter Test")
    print("=" * 60)
    
    success = test_api_resolution()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 Test completed successfully!")
    else:
        print("💥 Test failed!")
    print("=" * 60)