import boto3, os
from botocore.exceptions import ClientError
from boto3.s3.transfer import TransferConfig

def upload_to_r2(file_path, video_id=None):
    try:
        print(f"🔍 Uploading file: {file_path}")
        file_size = os.path.getsize(file_path)
        print(f"📏 File size: {file_size} bytes ({file_size / (1024*1024):.2f} MB)")
        
        session = boto3.session.Session()
        s3 = session.client(
            service_name='s3',
            endpoint_url=os.getenv("R2_ENDPOINT"),
            aws_access_key_id=os.getenv("R2_ACCESS_KEY"),
            aws_secret_access_key=os.getenv("R2_SECRET_KEY"),
            region_name='auto'
        )

        bucket = os.getenv("R2_BUCKET")
        filename = os.path.basename(file_path)
        
        # Organize files by video_id if provided
        if video_id:
            key = f"{video_id}/{filename}"
        else:
            key = filename
        
        print(f"🎯 Uploading to bucket: {bucket}, key: {key}")
        
        # Force single-part upload by setting a high multipart threshold
        # This avoids multipart upload issues with R2
        config = TransferConfig(
            multipart_threshold=1024 * 1024 * 1024,  # 1GB threshold
            max_concurrency=1,
            multipart_chunksize=1024 * 1024 * 64,    # 64MB chunks
            use_threads=False
        )
        
        # Use upload_file with custom config to avoid multipart issues
        s3.upload_file(file_path, bucket, key, Config=config)
        
        result_url = f"{os.getenv('R2_PUBLIC_URL')}/{key}"
        print(f"✅ Upload successful: {result_url}")
        return result_url
    
    except ClientError as e:
        print(f"🚨 ClientError details: {e}")
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        error_message = e.response.get('Error', {}).get('Message', 'Unknown error')
        error_details = f"R2 Upload failed - Code: {error_code}, Message: {error_message}"
        print(f"❌ {error_details}")
        raise Exception(error_details)
    except Exception as e:
        error_details = f"R2 Upload failed - {str(e)}"
        print(f"❌ {error_details}")
        raise Exception(error_details)
