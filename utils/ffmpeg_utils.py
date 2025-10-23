import subprocess
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_video_duration(video_path):
    # Use specific FFmpeg path or fallback to PATH
    ffprobe_path = os.getenv('FFMPEG_PROBE_PATH', 'ffprobe')
    
    print(f"🔍 DEBUG: Using ffprobe path: {ffprobe_path}", flush=True)
    print(f"🔍 DEBUG: Video path: {video_path}", flush=True)
    
    cmd = [
        ffprobe_path, "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", video_path
    ]
    
    print(f"🔍 DEBUG: FFprobe command: {' '.join(cmd)}", flush=True)
    
    try:
        result = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        duration_str = result.decode().strip()
        print(f"🔍 DEBUG: FFprobe output: '{duration_str}'", flush=True)
        return float(duration_str)
    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR: FFprobe failed with return code {e.returncode}: {e.output.decode()}", flush=True)
        raise
    except FileNotFoundError as e:
        print(f"❌ ERROR: FFprobe executable not found: {e}", flush=True)
        print(f"❌ ERROR: Tried path: {ffprobe_path}", flush=True)
        raise
    except Exception as e:
        print(f"❌ ERROR: Unexpected error in get_video_duration: {e}", flush=True)
        raise
