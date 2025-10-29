#!/usr/bin/env python3
"""
Isolated R2 Upload Test
Tests the R2 upload functionality with proper error handling and debugging
"""

import os
import sys
import tempfile
import json
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from services.uploader import upload_to_r2

def test_r2_upload_with_sample_file():
    """Test R2 upload with a sample JSON file"""
    print("\n" + "="*70)
    print("🧪 ISOLATED R2 UPLOAD TEST")
    print("="*70)
    
    # Check environment variables
    print("🔍 Checking R2 environment variables...")
    r2_vars = {
        'R2_ACCESS_KEY': os.getenv('R2_ACCESS_KEY'),
        'R2_SECRET_KEY': os.getenv('R2_SECRET_KEY'), 
        'R2_BUCKET': os.getenv('R2_BUCKET'),
        'R2_ENDPOINT': os.getenv('R2_ENDPOINT'),
        'R2_PUBLIC_URL': os.getenv('R2_PUBLIC_URL')
    }
    
    for var_name, var_value in r2_vars.items():
        if var_value:
            print(f"✅ {var_name}: {'*' * min(len(var_value), 20)}...")
        else:
            print(f"❌ {var_name}: Not set")
    
    # Check if any R2 credentials are missing
    missing_vars = [var for var, val in r2_vars.items() if not val]
    if missing_vars:
        print(f"\n⚠️ Missing R2 environment variables: {', '.join(missing_vars)}")
        print("💡 This test will demonstrate the error handling when credentials are missing")
    
    # Create a sample JSON file for testing
    sample_data = {
        "test_upload": True,
        "timestamp": datetime.now().isoformat(),
        "message": "This is a test file for R2 upload functionality",
        "segments": [
            {"start": 0.0, "end": 2.5, "text": "Sample transcript segment 1"},
            {"start": 2.5, "end": 5.0, "text": "Sample transcript segment 2"},
            {"start": 5.0, "end": 7.5, "text": "Sample transcript segment 3"}
        ]
    }
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
        json.dump(sample_data, temp_file, indent=2)
        temp_file_path = temp_file.name
    
    print(f"\n📄 Created test file: {temp_file_path}")
    print(f"📏 File size: {os.path.getsize(temp_file_path)} bytes")
    
    try:
        print("\n🚀 Testing R2 upload...")
        
        # Test 1: Upload without video_id
        print("\n--- Test 1: Upload without video_id ---")
        try:
            result_url = upload_to_r2(temp_file_path)
            print(f"✅ Upload successful: {result_url}")
        except Exception as e:
            print(f"❌ Upload failed: {e}")
            print(f"🔍 Error type: {type(e).__name__}")
        
        # Test 2: Upload with video_id
        print("\n--- Test 2: Upload with video_id ---")
        try:
            result_url = upload_to_r2(temp_file_path, video_id="test_video_456")
            print(f"✅ Upload successful: {result_url}")
        except Exception as e:
            print(f"❌ Upload failed: {e}")
            print(f"🔍 Error type: {type(e).__name__}")
            
            # Additional debugging for NoneType error
            if "NoneType" in str(e):
                print("\n🔍 DEBUGGING NoneType ERROR:")
                print(f"   R2_BUCKET value: {repr(os.getenv('R2_BUCKET'))}")
                print(f"   R2_ENDPOINT value: {repr(os.getenv('R2_ENDPOINT'))}")
                print("   This error occurs when bucket name is None")
                
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_file_path)
            print(f"\n🧹 Cleaned up temporary file: {temp_file_path}")
        except Exception as cleanup_error:
            print(f"⚠️ Failed to clean up temporary file: {cleanup_error}")

def test_r2_upload_with_mock_credentials():
    """Test R2 upload behavior with mock credentials to isolate the issue"""
    print("\n" + "="*70)
    print("🧪 MOCK CREDENTIALS TEST")
    print("="*70)
    
    # Temporarily set mock environment variables
    original_vars = {}
    mock_vars = {
        'R2_ACCESS_KEY': 'mock_access_key',
        'R2_SECRET_KEY': 'mock_secret_key',
        'R2_BUCKET': 'mock_bucket',
        'R2_ENDPOINT': 'https://mock.r2.cloudflarestorage.com',
        'R2_PUBLIC_URL': 'https://mock.example.com'
    }
    
    # Save original values and set mock values
    for var_name, mock_value in mock_vars.items():
        original_vars[var_name] = os.getenv(var_name)
        os.environ[var_name] = mock_value
        print(f"🔧 Set {var_name} to mock value")
    
    # Create a sample file
    sample_data = {"mock_test": True, "timestamp": datetime.now().isoformat()}
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
        json.dump(sample_data, temp_file, indent=2)
        temp_file_path = temp_file.name
    
    try:
        print(f"\n📄 Created mock test file: {temp_file_path}")
        print("🚀 Testing R2 upload with mock credentials...")
        
        result_url = upload_to_r2(temp_file_path, video_id="mock_test_789")
        print(f"✅ Mock upload would succeed: {result_url}")
        
    except Exception as e:
        print(f"❌ Mock upload failed: {e}")
        print(f"🔍 Error type: {type(e).__name__}")
        
        # This should help us understand if the issue is credentials or code logic
        if "NoneType" in str(e):
            print("🚨 NoneType error still occurs with mock credentials!")
            print("   This indicates a code logic issue, not just missing credentials")
        else:
            print("✅ No NoneType error with mock credentials")
            print("   This confirms the issue is missing environment variables")
            
    finally:
        # Restore original environment variables
        for var_name, original_value in original_vars.items():
            if original_value is not None:
                os.environ[var_name] = original_value
            else:
                os.environ.pop(var_name, None)
        
        # Clean up temporary file
        try:
            os.unlink(temp_file_path)
        except Exception:
            pass

if __name__ == "__main__":
    print("=" * 80)
    print("🧪 R2 UPLOAD ISOLATED TEST SUITE")
    print("=" * 80)
    
    test_r2_upload_with_sample_file()
    test_r2_upload_with_mock_credentials()
    
    print("\n" + "=" * 80)
    print("🎯 Test completed - Check results above for debugging info")
    print("=" * 80)