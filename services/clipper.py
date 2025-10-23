import os
import json
import subprocess
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from utils.srt_utils import generate_srt
from utils.ffmpeg_utils import get_video_duration
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

MAX_WORKERS = int(os.getenv("CLIPPER_MAX_WORKERS", 4))
FFMPEG_PATH = os.getenv("FFMPEG_PATH", "ffmpeg")

logger = logging.getLogger(__name__)

def generate_clips(video_path, clip_duration, transcript, output_dir, cache_file=None, platform=None, source_id=None):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    duration = int(parse_duration(clip_duration))
    
    print(f"🔍 DEBUG: Parsed clip duration: {duration} seconds", flush=True)
    print(f"🔍 DEBUG: About to call get_video_duration for: {video_path}", flush=True)
    
    try:
        total_duration = get_video_duration(video_path)
        print(f"🔍 DEBUG: Video total duration: {total_duration} seconds", flush=True)
    except Exception as e:
        print(f"❌ ERROR: Failed to get video duration: {e}", flush=True)
        print(f"❌ ERROR: Exception type: {type(e).__name__}", flush=True)
        logger.error(f"Failed to get video duration: {e}")
        return []
    
    num_clips = int(total_duration // duration) + (1 if total_duration % duration > 0 else 0)
    print(f"🔍 DEBUG: Calculated number of clips: {num_clips}", flush=True)

    logger.info(f"🎬 Clip generation started - Video: {video_path}, Duration: {total_duration}s, Clip length: {duration}s, Expected clips: {num_clips}")
    logger.info(f"📝 Transcript has {len(transcript)} segments")
    
    # Debug transcript data
    if transcript:
        logger.info(f"📊 First transcript segment: {transcript[0]}")
        logger.info(f"📊 Last transcript segment: {transcript[-1]}")
    else:
        logger.warning("⚠️ Transcript is empty!")

    # Early return if no clips to generate
    if num_clips <= 0:
        print(f"⚠️ WARNING: No clips to generate (num_clips={num_clips})", flush=True)
        return []

    clips, cache = [], {}
    cache_lock = Lock()

    # Load existing cache
    if cache_file and os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                cache = json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Corrupted cache file: {cache_file}, starting fresh")

    def is_valid_clip(path):
        return os.path.exists(path) and os.path.getsize(path) > 0

    def process_clip(i):
        start_time = i * duration
        actual_duration = min(duration, total_duration - start_time)
        end_time = start_time + actual_duration

        clip_name = f"clip_{i+1}.mp4"
        clip_path = os.path.join(output_dir, clip_name)
        srt_path = os.path.join(output_dir, f"clip_{i+1}.srt")

        logger.info(f"🎞️ Processing clip {i+1}: {start_time}s - {end_time}s")

        # Skip cached
        if clip_name in cache and is_valid_clip(clip_path):
            logger.info(f"♻️ Using cached clip {i+1}")
            return cache[clip_name]

        clip_segments = []
        for t in transcript:
            if t["start"] < end_time and t["end"] > start_time:
                s = t.copy()
                s["start"] = max(0, t["start"] - start_time)
                s["end"] = min(actual_duration, t["end"] - start_time)
                clip_segments.append(s)

        transcript_text = " ".join([t["text"] for t in clip_segments])
        logger.info(f"📝 Clip {i+1} has {len(clip_segments)} transcript segments")

        # Extract video
        logger.info(f"🎬 Extracting video clip {i+1} using FFmpeg")
        result = subprocess.run([
            FFMPEG_PATH, "-ss", str(start_time), "-t", str(actual_duration),
            "-i", video_path, "-c", "copy", clip_path, "-y"
        ], capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"❌ FFmpeg failed for clip {i+1}: {result.stderr}")
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")
        if not is_valid_clip(clip_path):
            logger.error(f"❌ Invalid clip generated: {clip_path}")
            raise RuntimeError(f"Invalid clip: {clip_path}")

        logger.info(f"✅ Video clip {i+1} extracted successfully")

        # Generate SRT
        generate_srt(clip_segments, srt_path)
        logger.info(f"📄 SRT file generated for clip {i+1}")

        clip_data = {
            "index": i + 1,
            "video": clip_path,
            "srt": srt_path,
            "transcript_text": transcript_text,
            "start_time": start_time,
            "end_time": end_time,
            "platform": platform,
            "source_id": source_id,
        }

        with cache_lock:
            cache[clip_name] = clip_data

        logger.info(f"🎉 Clip {i+1} processed successfully")
        return clip_data

    # Run in parallel
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_clip, i): i for i in range(num_clips)}
        for future in as_completed(futures):
            i = futures[future]
            try:
                clip_data = future.result()
                clips.append(clip_data)
            except Exception:
                logger.error(f"Failed to process clip {i+1}", exc_info=True)

    # Write final cache atomically
    if cache_file:
        tmp = cache_file + ".tmp"
        with open(tmp, "w") as f:
            json.dump(cache, f, indent=2)
        os.replace(tmp, cache_file)

    clips.sort(key=lambda x: x["index"])
    return clips


def parse_duration(value):
    value = str(value).lower().strip()
    try:
        if value.endswith("m"):
            return int(value[:-1]) * 60
        elif value.endswith("s"):
            return int(value[:-1])
        return int(value)
    except ValueError:
        raise ValueError(f"Invalid duration: {value}. Use '60', '60s', or '1m'")
