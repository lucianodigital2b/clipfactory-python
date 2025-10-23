import os
import time
import tempfile
from services.uploader import upload_to_r2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_r2_upload():
    print("🧪 Testing R2 Upload...")
    print(f"R2_ENDPOINT: {os.getenv('R2_ENDPOINT')}")
    print(f"R2_BUCKET: {os.getenv('R2_BUCKET')}")
    print(f"R2_ACCESS_KEY: {os.getenv('R2_ACCESS_KEY')[:8]}...")
    print(f"R2_PUBLIC_URL: {os.getenv('R2_PUBLIC_URL')}")
    
    # Use the large cda.sql file for testing
    temp_file_path = "./cda.sql"
    
    if not os.path.exists(temp_file_path):
        print(f"❌ Test file not found: {temp_file_path}")
        return
        
    print(f"\n📁 Using test file: {temp_file_path}")
    
    try:
        # Test the upload
        print("\n🚀 Attempting R2 upload...")
        result_url = upload_to_r2(temp_file_path)
        print(f"✅ Upload successful!")
        print(f"📎 File URL: {result_url}")
        
    except Exception as e:
        print(f"❌ Upload failed: {str(e)}")
        
    finally:
        # Don't clean up the cda.sql file since it's part of the project
        print(f"\n✅ Test completed using project file: {temp_file_path}")

if __name__ == "__main__":
    test_r2_upload()