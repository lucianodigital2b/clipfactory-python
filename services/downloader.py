import yt_dlp
import boto3
import os
import logging
import requests
from urllib.parse import urlparse
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DownloadError(Exception):
    """Custom exception for download errors"""
    pass

def _is_youtube_url(url: str) -> bool:
    """Check if URL is a YouTube video URL"""
    youtube_domains = ['youtube.com', 'youtu.be', 'www.youtube.com', 'm.youtube.com']
    parsed = urlparse(url)
    return parsed.netloc.lower() in youtube_domains

def _is_r2_url(url: str) -> bool:
    """Check if URL is an R2 bucket URL"""
    r2_public_url = os.getenv("R2_PUBLIC_URL")
    if not r2_public_url:
        return False
    return url.startswith(r2_public_url)

def _download_from_youtube(url: str, output_dir: str) -> str:
    """Download video from YouTube using yt-dlp"""
    try:
        logger.info(f"Downloading YouTube video from: {url}")
        
        ydl_opts = {
            "outtmpl": os.path.join(output_dir, "input.%(ext)s"),
            "format": "best[ext=mp4]/best",  # Prefer mp4, fallback to best available
            "quiet": True,
            "no_warnings": True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            file_path = ydl.prepare_filename(info)
            
            # Ensure the file exists
            if not os.path.exists(file_path):
                raise DownloadError(f"Downloaded file not found: {file_path}")
                
            logger.info(f"Successfully downloaded YouTube video to: {file_path}")
            return file_path
            
    except Exception as e:
        logger.error(f"Failed to download YouTube video: {str(e)}")
        raise DownloadError(f"YouTube download failed: {str(e)}")

def _download_from_r2(url: str, output_dir: str) -> str:
    """Download file from R2 bucket"""
    try:
        logger.info(f"Downloading from R2 bucket: {url}")
        
        # Extract filename from URL
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        
        if not filename:
            filename = "input.mp4"  # Default filename
            
        output_path = os.path.join(output_dir, filename)
        
        # Download using requests for HTTP/HTTPS URLs
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Write file in chunks
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        # Verify file was downloaded
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise DownloadError(f"Downloaded file is empty or missing: {output_path}")
            
        logger.info(f"Successfully downloaded R2 file to: {output_path}")
        return output_path
        
    except requests.RequestException as e:
        logger.error(f"Failed to download from R2: {str(e)}")
        raise DownloadError(f"R2 download failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error downloading from R2: {str(e)}")
        raise DownloadError(f"R2 download failed: {str(e)}")

def _download_generic_file(url: str, output_dir: str) -> str:
    """Download file from generic HTTP/HTTPS URL"""
    try:
        logger.info(f"Downloading generic file from: {url}")
        
        # Extract filename from URL or use default
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        
        if not filename or '.' not in filename:
            filename = "input.mp4"  # Default filename
            
        output_path = os.path.join(output_dir, filename)
        
        # Download using requests
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Write file in chunks
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        # Verify file was downloaded
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise DownloadError(f"Downloaded file is empty or missing: {output_path}")
            
        logger.info(f"Successfully downloaded generic file to: {output_path}")
        return output_path
        
    except requests.RequestException as e:
        logger.error(f"Failed to download generic file: {str(e)}")
        raise DownloadError(f"Generic file download failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error downloading generic file: {str(e)}")
        raise DownloadError(f"Generic file download failed: {str(e)}")

def download_video(url: str, output_dir: str) -> str:
    """
    Download video from various sources (YouTube, R2 bucket, or generic HTTP/HTTPS)
    
    Args:
        url (str): URL to download from
        output_dir (str): Directory to save the downloaded file
        
    Returns:
        str: Path to the downloaded file
        
    Raises:
        DownloadError: If download fails
        ValueError: If inputs are invalid
    """
    # Validate inputs
    if not url or not isinstance(url, str):
        raise ValueError("URL must be a non-empty string")
        
    if not output_dir or not isinstance(output_dir, str):
        raise ValueError("Output directory must be a non-empty string")
    
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Determine URL type and download accordingly
    try:
        if _is_youtube_url(url):
            return _download_from_youtube(url, output_dir)
        elif _is_r2_url(url):
            return _download_from_r2(url, output_dir)
        else:
            # Try as generic HTTP/HTTPS file
            return _download_generic_file(url, output_dir)
            
    except DownloadError:
        # Re-raise download errors as-is
        raise
    except Exception as e:
        logger.error(f"Unexpected error in download_video: {str(e)}")
        raise DownloadError(f"Download failed: {str(e)}")
