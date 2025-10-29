import os
import cv2
import numpy as np
import subprocess
import tempfile
import logging
from typing import List, Tuple, Optional, Dict
from datetime import datetime
from ultralytics import YOLO
import mediapipe as mp
import librosa
import soundfile as sf
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioAnalyzer:
    """Handles audio analysis for voice activity detection"""
    
    def __init__(self, frame_duration_ms: int = 100):
        self.frame_duration_ms = frame_duration_ms
        self.audio_activity = {}
    
    def extract_audio_from_video(self, video_path: str) -> Tuple[np.ndarray, int]:
        """Extract audio track from video file"""
        try:
            temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_audio_path = temp_audio.name
            temp_audio.close()
            
            # Extract audio using ffmpeg
            cmd = [
                'ffmpeg', '-y', '-i', video_path,
                '-vn', '-acodec', 'pcm_s16le',
                '-ar', '16000', '-ac', '1',
                temp_audio_path
            ]
            
            subprocess.run(cmd, capture_output=True, check=True)
            
            # Load audio
            audio, sr = librosa.load(temp_audio_path, sr=16000, mono=True)
            
            # Clean up
            os.unlink(temp_audio_path)
            
            logger.info(f"Extracted audio: {len(audio)} samples at {sr}Hz")
            return audio, sr
            
        except Exception as e:
            logger.error(f"Error extracting audio: {e}")
            return np.array([]), 16000
    
    def detect_voice_activity(self, audio: np.ndarray, sr: int, fps: float) -> Dict[int, float]:
        """
        Detect voice activity and map to frame numbers
        Returns dict: {frame_idx: audio_energy}
        """
        if len(audio) == 0:
            return {}
        
        # Calculate energy per frame
        samples_per_frame = int(sr / fps)
        frame_audio_energy = {}
        
        for frame_idx in range(int(len(audio) / samples_per_frame)):
            start_sample = frame_idx * samples_per_frame
            end_sample = min(start_sample + samples_per_frame, len(audio))
            
            frame_audio = audio[start_sample:end_sample]
            
            # Calculate RMS energy
            energy = np.sqrt(np.mean(frame_audio ** 2))
            
            # Apply threshold (normalized)
            if energy > 0.01:  # Adjust threshold as needed
                frame_audio_energy[frame_idx] = float(energy)
        
        # Normalize energy values
        if frame_audio_energy:
            max_energy = max(frame_audio_energy.values())
            if max_energy > 0:
                frame_audio_energy = {k: v/max_energy for k, v in frame_audio_energy.items()}
        
        logger.info(f"Detected voice activity in {len(frame_audio_energy)} frames")
        return frame_audio_energy


class LipMovementDetector:
    """Detects lip movement using MediaPipe Face Mesh"""
    
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Lip landmark indices (outer lips)
        self.UPPER_LIP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
        self.LOWER_LIP = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
        
        # History for temporal smoothing
        self.lip_history = {}  # {person_idx: deque of lip distances}
        self.history_length = 5
    
    def calculate_lip_distance(self, landmarks, frame_height: int, frame_width: int) -> float:
        """Calculate vertical distance between upper and lower lips"""
        try:
            # Get average positions of upper and lower lips
            upper_y = np.mean([landmarks[i].y * frame_height for i in self.UPPER_LIP])
            lower_y = np.mean([landmarks[i].y * frame_height for i in self.LOWER_LIP])
            
            # Calculate vertical distance
            distance = abs(lower_y - upper_y)
            return distance
            
        except Exception as e:
            return 0.0
    
    def detect_lip_movement(self, frame: np.ndarray, person_bboxes: List[Tuple[int, int, int, int]]) -> Dict[int, float]:
        """
        Detect lip movement for each person
        Returns dict: {person_idx: lip_movement_score}
        """
        lip_scores = {}
        
        if not person_bboxes:
            return lip_scores
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_height, frame_width = frame.shape[:2]
        
        for person_idx, bbox in enumerate(person_bboxes):
            x1, y1, x2, y2 = bbox
            
            # Add padding to bbox for better face detection
            padding = 20
            x1_padded = max(0, x1 - padding)
            y1_padded = max(0, y1 - padding)
            x2_padded = min(frame_width, x2 + padding)
            y2_padded = min(frame_height, y2 + padding)
            
            # Extract person region
            person_roi = rgb_frame[y1_padded:y2_padded, x1_padded:x2_padded]
            
            if person_roi.size == 0:
                continue
            
            # Detect face mesh
            results = self.face_mesh.process(person_roi)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Calculate current lip distance
                    roi_height, roi_width = person_roi.shape[:2]
                    lip_distance = self.calculate_lip_distance(face_landmarks.landmark, roi_height, roi_width)
                    
                    # Initialize history for this person if needed
                    if person_idx not in self.lip_history:
                        self.lip_history[person_idx] = deque(maxlen=self.history_length)
                    
                    # Add to history
                    self.lip_history[person_idx].append(lip_distance)
                    
                    # Calculate movement as variance in recent history
                    if len(self.lip_history[person_idx]) >= 3:
                        lip_variance = np.var(list(self.lip_history[person_idx]))
                        lip_scores[person_idx] = float(lip_variance)
                    else:
                        lip_scores[person_idx] = 0.0
                    
                    break  # Only process first face in person ROI
        
        return lip_scores
    
    def cleanup(self):
        """Release MediaPipe resources"""
        self.face_mesh.close()


class SpeakerDetector:
    """Combines audio and visual cues to identify active speaker"""
    
    def __init__(self, audio_weight: float = 0.6, lip_weight: float = 0.4):
        self.audio_weight = audio_weight
        self.lip_weight = lip_weight
        self.speaker_history = deque(maxlen=10)  # Track recent speaker for smoothing
    
    def identify_speaker(self, frame_idx: int, audio_energy: float, 
                        lip_scores: Dict[int, float]) -> Optional[int]:
        """
        Identify the active speaker based on audio and lip movement
        Returns person index of the speaker, or None if no clear speaker
        """
        if not lip_scores:
            return None
        
        # Normalize lip scores
        max_lip_score = max(lip_scores.values()) if lip_scores else 1.0
        normalized_lip_scores = {k: v/max_lip_score for k, v in lip_scores.items()} if max_lip_score > 0 else lip_scores
        
        # Calculate combined scores for each person
        combined_scores = {}
        for person_idx, lip_score in normalized_lip_scores.items():
            # Combine audio energy (same for all) with individual lip movement
            combined_score = (self.audio_weight * audio_energy) + (self.lip_weight * lip_score)
            combined_scores[person_idx] = combined_score
        
        # Find person with highest score
        if combined_scores:
            speaker_idx = max(combined_scores.items(), key=lambda x: x[1])
            
            # Only return if score is above threshold
            if speaker_idx[1] > 0.3:  # Threshold for considering someone as speaking
                self.speaker_history.append(speaker_idx[0])
                return speaker_idx[0]
        
        # Use most common recent speaker as fallback
        if self.speaker_history:
            from collections import Counter
            most_common = Counter(self.speaker_history).most_common(1)[0][0]
            return most_common
        
        return None


class BoundingBoxSmoother:
    """Handles smoothing of bounding boxes to prevent camera jitter"""
    
    def __init__(self, smoothing_factor: float = 0.3, max_persons: int = 2):
        self.smoothing_factor = smoothing_factor
        self.max_persons = max_persons
        self.previous_boxes = []
        self.box_history = []
        self.max_history = 10
    
    def smooth_boxes(self, current_boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """Apply exponential smoothing to multiple bounding boxes"""
        if not current_boxes:
            return self.previous_boxes if self.previous_boxes else []
        
        # Limit to max_persons
        current_boxes = current_boxes[:self.max_persons]
        
        if not self.previous_boxes:
            self.previous_boxes = current_boxes
            return current_boxes
        
        # Match current boxes with previous boxes based on proximity
        matched_boxes = []
        used_previous = set()
        
        for current_box in current_boxes:
            best_match_idx = None
            best_distance = float('inf')
            
            # Find the closest previous box
            for i, prev_box in enumerate(self.previous_boxes):
                if i in used_previous:
                    continue
                    
                # Calculate center distance
                curr_center_x = (current_box[0] + current_box[2]) / 2
                curr_center_y = (current_box[1] + current_box[3]) / 2
                prev_center_x = (prev_box[0] + prev_box[2]) / 2
                prev_center_y = (prev_box[1] + prev_box[3]) / 2
                
                distance = ((curr_center_x - prev_center_x) ** 2 + (curr_center_y - prev_center_y) ** 2) ** 0.5
                
                if distance < best_distance:
                    best_distance = distance
                    best_match_idx = i
            
            if best_match_idx is not None and best_distance < 200:
                # Apply smoothing with matched previous box
                prev_box = self.previous_boxes[best_match_idx]
                used_previous.add(best_match_idx)
                
                x1, y1, x2, y2 = current_box
                px1, py1, px2, py2 = prev_box
                
                smoothed_x1 = int(px1 + self.smoothing_factor * (x1 - px1))
                smoothed_y1 = int(py1 + self.smoothing_factor * (y1 - py1))
                smoothed_x2 = int(px2 + self.smoothing_factor * (x2 - px2))
                smoothed_y2 = int(py2 + self.smoothing_factor * (y2 - py2))
                
                matched_boxes.append((smoothed_x1, smoothed_y1, smoothed_x2, smoothed_y2))
            else:
                # No good match found, use current box as-is
                matched_boxes.append(current_box)
        
        self.previous_boxes = matched_boxes
        
        # Keep history for moving average fallback
        self.box_history.append(matched_boxes)
        if len(self.box_history) > self.max_history:
            self.box_history.pop(0)
        
        return matched_boxes


def detect_person_in_frame(model: YOLO, frame: np.ndarray, confidence_threshold: float = 0.5, 
                          max_persons: int = 2) -> List[Tuple[int, int, int, int]]:
    """
    Detect multiple persons in a frame using YOLOv8
    Returns list of bounding boxes [(x1, y1, x2, y2), ...] sorted by confidence
    """
    try:
        results = model(frame, verbose=False)
        
        person_boxes = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Class 0 is 'person' in COCO dataset
                    if box.cls == 0 and box.conf > confidence_threshold:
                        confidence = float(box.conf)
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        person_boxes.append((x1, y1, x2, y2, confidence))
        
        # Sort by confidence and limit to max_persons
        person_boxes.sort(key=lambda x: x[4], reverse=True)
        person_boxes = person_boxes[:max_persons]
        
        return [(x1, y1, x2, y2) for x1, y1, x2, y2, _ in person_boxes]
    
    except Exception as e:
        logger.error(f"Error in person detection: {e}")
        return []


def parse_resolution(resolution: str) -> int:
    """
    Parse resolution string and return height in pixels.
    
    Args:
        resolution: Resolution string (e.g., "360p", "720p", "1080p")
    
    Returns:
        int: Height in pixels
    """
    resolution_map = {
        "240p": 240,
        "360p": 360,
        "480p": 480,
        "720p": 720,
        "1080p": 1080,
        "1440p": 1440,
        "2160p": 2160  # 4K
    }
    
    resolution_lower = resolution.lower()
    if resolution_lower in resolution_map:
        return resolution_map[resolution_lower]
    
    # Try to extract number from string (e.g., "720" from "720p")
    import re
    match = re.search(r'(\d+)', resolution)
    if match:
        return int(match.group(1))
    
    # Default fallback
    logger.warning(f"Unknown resolution '{resolution}', defaulting to 360p")
    return 360

def parse_aspect_ratio(aspect_ratio_str: str) -> float:
    """
    Parse aspect ratio string to float.
    Examples: "9:16" -> 0.5625, "16:9" -> 1.7778, "1:1" -> 1.0
    """
    try:
        if ':' in aspect_ratio_str:
            w, h = aspect_ratio_str.split(':')
            return float(w) / float(h)
        else:
            return float(aspect_ratio_str)
    except:
        logger.warning(f"Could not parse aspect ratio '{aspect_ratio_str}', using 9:16")
        return 9/16


def calculate_crop_parameters(speaker_bbox: Tuple[int, int, int, int], 
                             all_bboxes: List[Tuple[int, int, int, int]],
                             frame_width: int, frame_height: int, 
                             target_aspect_ratio: float = 9/16, 
                             zoom_factor: float = 2.5) -> Tuple[int, int, int, int]:
    """
    Calculate crop parameters focused on the active speaker's face
    Returns (crop_x, crop_y, crop_width, crop_height)
    """
    if not speaker_bbox:
        # Fallback to center crop
        crop_width = int(frame_height * target_aspect_ratio)
        crop_height = frame_height
        crop_x = max(0, (frame_width - crop_width) // 2)
        crop_y = 0
        return (crop_x, crop_y, crop_width, crop_height)
    
    x1, y1, x2, y2 = speaker_bbox
    
    # Calculate speaker center and dimensions
    speaker_center_x = (x1 + x2) // 2
    speaker_center_y = (y1 + y2) // 2
    speaker_width = x2 - x1
    speaker_height = y2 - y1
    
    # Focus on the upper portion of the person (head/face area)
    # Adjust center point to focus on face (upper 1/3 of person bbox)
    face_center_y = y1 + int(speaker_height * 0.25)  # Focus on face area
    
    # Calculate desired crop size based on face dimensions
    # Use a tighter crop that focuses on head/shoulders
    face_width = int(speaker_width * zoom_factor)
    face_height = int(speaker_height * zoom_factor * 0.6)  # Focus on upper portion
    
    # Determine crop dimensions based on target aspect ratio
    if target_aspect_ratio < 1.0:  # Portrait mode (9:16)
        # For portrait, make height based on face area
        crop_height = max(face_height, int(face_width / target_aspect_ratio))
        crop_width = int(crop_height * target_aspect_ratio)
    else:  # Landscape mode (16:9)
        # For landscape, make width based on face area
        crop_width = max(face_width, int(face_height * target_aspect_ratio))
        crop_height = int(crop_width / target_aspect_ratio)
    
    # Ensure crop doesn't exceed frame dimensions
    crop_width = min(crop_width, frame_width)
    crop_height = min(crop_height, frame_height)
    
    # Center crop around face area
    crop_x = max(0, speaker_center_x - crop_width // 2)
    crop_y = max(0, face_center_y - crop_height // 2)
    
    # Adjust if crop goes beyond boundaries
    if crop_x + crop_width > frame_width:
        crop_x = frame_width - crop_width
    if crop_y + crop_height > frame_height:
        crop_y = frame_height - crop_height
    
    return (crop_x, crop_y, crop_width, crop_height)


def focus_on_speaker(video_path: str, output_path: str, 
                    start_time: float = 0, 
                    duration: Optional[float] = None,
                    target_aspect_ratio: str = "9:16",
                    min_confidence: float = 0.6,
                    fps_batch: int = 5,
                    resolution: str = "360p") -> bool:
    """
    Uses audio-visual analysis to detect and track the active speaker.
    Combines YOLOv8 (person detection), MediaPipe (lip movement), and audio analysis.
    Outputs a video focused on the active speaker with the specified aspect ratio.
    
    NEW SIGNATURE COMPLIANCE:
    Args:
        video_path: Path to input video file
        output_path: Path to save output video
        start_time: Start time in seconds for clip processing (default: 0)
        duration: Duration in seconds for clip processing (default: None = full video)
        target_aspect_ratio: Target aspect ratio as string (e.g., "9:16", "16:9", "1:1")
        min_confidence: Minimum confidence threshold for face detection (0.0-1.0)
        fps_batch: Process every Nth frame (higher = faster but less accurate)
    
    Returns:
        bool: True if face detected successfully and processing completed, False otherwise
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"[{timestamp}] 🎯 Starting audio-visual speaker detection")
    logger.info(f"[{timestamp}] 📁 Input: {video_path}")
    logger.info(f"[{timestamp}] 📁 Output: {output_path}")
    logger.info(f"[{timestamp}] 🎬 Clip: start={start_time}s, duration={duration}s")
    logger.info(f"[{timestamp}] 📐 Target aspect ratio: {target_aspect_ratio}")
    logger.info(f"[{timestamp}] 🎯 Min confidence: {min_confidence}")
    
    # Parse aspect ratio and resolution
    aspect_ratio_float = parse_aspect_ratio(target_aspect_ratio)
    
    try:
        # Initialize components
        logger.info(f"[{timestamp}] 🤖 Loading YOLOv8n model...")
        yolo_model = YOLO("yolov8n.pt")
        
        logger.info(f"[{timestamp}] 🎤 Initializing audio analyzer...")
        audio_analyzer = AudioAnalyzer()
        
        logger.info(f"[{timestamp}] 👄 Initializing lip movement detector...")
        lip_detector = LipMovementDetector()
        
        logger.info(f"[{timestamp}] 🎯 Initializing speaker detector...")
        speaker_detector = SpeakerDetector(audio_weight=0.6, lip_weight=0.4)
        
        # Extract and analyze audio
        logger.info(f"[{timestamp}] 🎵 Extracting audio from video...")
        audio, sr = audio_analyzer.extract_audio_from_video(video_path)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Handle start_time and duration for clip processing
        start_frame = int(start_time * fps) if start_time > 0 else 0
        if duration is not None:
            end_frame = min(start_frame + int(duration * fps), total_frames)
        else:
            end_frame = total_frames
        
        # Set video position to start frame
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Update total frames to reflect the clip duration
        clip_total_frames = end_frame - start_frame
        
        logger.info(f"[{timestamp}] 📊 Video: {frame_width}x{frame_height}, processing frames {start_frame}-{end_frame} ({clip_total_frames} frames), {fps:.2f} FPS")
        
        # Detect voice activity
        logger.info(f"[{timestamp}] 🔊 Analyzing voice activity...")
        frame_audio_energy = audio_analyzer.detect_voice_activity(audio, sr, fps)
        
        # Initialize smoothers
        bbox_smoother = BoundingBoxSmoother(smoothing_factor=0.3, max_persons=2)
        
        # Process frames
        frame_data = {}  # {frame_idx: {'bboxes': [], 'speaker_idx': int, 'crop': tuple}}
        processed_frames = 0
        speaker_switches = 0
        previous_speaker = None
        face_detected_frames = 0  # NEW: Track frames with face detection
        
        logger.info(f"[{timestamp}] 🔍 Processing frames for speaker detection...")
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame_idx >= clip_total_frames:
                break
            
            # Process every fps_batch frames
            if frame_idx % fps_batch == 0:
                # Detect persons
                person_bboxes = detect_person_in_frame(yolo_model, frame, confidence_threshold=min_confidence, max_persons=2)
                
                if person_bboxes:
                    # Smooth bounding boxes
                    smoothed_bboxes = bbox_smoother.smooth_boxes(person_bboxes)
                    
                    # Get audio energy for this frame
                    audio_energy = frame_audio_energy.get(frame_idx + start_frame, 0.0)
                    
                    # Detect lip movement (only if persons are detected)
                    lip_scores = lip_detector.detect_lip_movement(frame, smoothed_bboxes)
                    
                    # Identify active speaker
                    speaker_idx = speaker_detector.identify_speaker(frame_idx, audio_energy, lip_scores)
                    
                    # Track speaker switches
                    if speaker_idx is not None and speaker_idx != previous_speaker:
                        speaker_switches += 1
                        previous_speaker = speaker_idx
                    
                    # Calculate crop parameters focused on speaker
                    if speaker_idx is not None and speaker_idx < len(smoothed_bboxes):
                        speaker_bbox = smoothed_bboxes[speaker_idx]
                        crop_params = calculate_crop_parameters(
                            speaker_bbox, smoothed_bboxes, 
                            frame_width, frame_height, 
                            target_aspect_ratio=aspect_ratio_float, zoom_factor=1.2
                        )
                        face_detected_frames += 1  # NEW: Count successful face detections
                    else:
                        # No clear speaker, use first person or center crop
                        speaker_bbox = smoothed_bboxes[0] if smoothed_bboxes else None
                        crop_params = calculate_crop_parameters(
                            speaker_bbox, smoothed_bboxes,
                            frame_width, frame_height,
                            target_aspect_ratio=aspect_ratio_float, zoom_factor=1.2
                        )
                    
                    # Log crop parameters for debugging
                    if frame_idx % 30 == 0:  # Log every 30 frames
                        crop_x, crop_y, crop_w, crop_h = crop_params
                        logger.info(f"🔧 Frame {frame_idx}: Speaker {speaker_idx} at bbox {speaker_bbox} -> crop {crop_w}x{crop_h} at ({crop_x}, {crop_y})")
                    
                    frame_data[frame_idx] = {
                        'bboxes': smoothed_bboxes,
                        'speaker_idx': speaker_idx,
                        'crop': crop_params,
                        'audio_energy': audio_energy
                    }
                    
                    processed_frames += 1
                else:
                    # No persons detected - use center crop for narration videos
                    audio_energy = frame_audio_energy.get(frame_idx + start_frame, 0.0)
                    
                    # Center crop for videos without people (narration, slides, etc.)
                    crop_width = int(frame_height * aspect_ratio_float)
                    crop_height = frame_height
                    crop_x = max(0, (frame_width - crop_width) // 2)
                    crop_y = 0
                    crop_params = (crop_x, crop_y, crop_width, crop_height)
                    
                    frame_data[frame_idx] = {
                        'bboxes': [],
                        'speaker_idx': None,
                        'crop': crop_params,
                        'audio_energy': audio_energy
                    }
                    
                    processed_frames += 1
            
            frame_idx += 1
        
        cap.release()
        lip_detector.cleanup()
        
        # Calculate statistics
        frames_with_people = sum(1 for data in frame_data.values() if data.get('bboxes'))
        frames_without_people = processed_frames - frames_with_people
        active_speaker_frames = sum(1 for data in frame_data.values() if data.get('speaker_idx') is not None)
        speaker_detection_rate = (active_speaker_frames / frames_with_people * 100) if frames_with_people > 0 else 0
        
        logger.info(f"[{timestamp}] 📈 Detection statistics:")
        logger.info(f"[{timestamp}] 📊 Processed frames: {processed_frames}")
        logger.info(f"[{timestamp}] 📊 Frames with people: {frames_with_people}")
        logger.info(f"[{timestamp}] 📊 Frames without people (narration): {frames_without_people}")
        logger.info(f"[{timestamp}] 📊 Active speaker detected: {active_speaker_frames} ({speaker_detection_rate:.1f}%)")
        logger.info(f"[{timestamp}] 📊 Speaker switches: {speaker_switches}")
        
        # NEW: Determine if face detection was successful based on min_confidence threshold
        face_detection_success = (active_speaker_frames / processed_frames) >= 0.1 if processed_frames > 0 else False
        logger.info(f"[{timestamp}] 🎯 Face detection success: {face_detection_success} ({active_speaker_frames}/{processed_frames} frames)")
        
        # Interpolate missing frames
        logger.info(f"[{timestamp}] 🔄 Interpolating crop parameters...")
        interpolated_crops = interpolate_crop_parameters(frame_data, clip_total_frames)
        
        # Parse resolution and calculate output dimensions based on target aspect ratio and resolution
        target_height = parse_resolution(resolution)
        
        # Calculate output dimensions based on target aspect ratio and resolution
        if aspect_ratio_float < 1:  # Portrait (e.g., 9:16)
            output_height = target_height
            output_width = int(output_height * aspect_ratio_float)
        elif aspect_ratio_float > 1:  # Landscape (e.g., 16:9)
            output_width = target_height  # Use target_height as base for landscape
            output_height = int(output_width / aspect_ratio_float)
        else:  # Square (1:1)
            output_width = output_height = target_height
        
        # Generate ffmpeg filter with clip timing
        logger.info(f"[{timestamp}] 🎬 Generating dynamic crop filter for {output_width}x{output_height}...")
        filter_complex = generate_ffmpeg_crop_filter_from_data(
            interpolated_crops, frame_width, frame_height, clip_total_frames, fps,
            output_width, output_height
        )
        
        # Do not trim in the video filter; use -ss/-t to keep A/V in sync.
        
        # Apply ffmpeg processing
        logger.info(f"[{timestamp}] 🎬 Applying video processing...")
        ffmpeg_success = apply_ffmpeg_processing(
            video_path, output_path, filter_complex, start_time=start_time, duration=duration
        )
        
        if ffmpeg_success:
            logger.info(f"[{timestamp}] ✅ Speaker-focused video created successfully!")
            logger.info(f"[{timestamp}] 📁 Output: {output_path}")
            return face_detection_success  # NEW: Return boolean indicating face detection success
        else:
            raise RuntimeError("FFmpeg processing failed")
    
    except Exception as e:
        logger.error(f"[{timestamp}] ❌ Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False  # NEW: Return False on error


def interpolate_crop_parameters(frame_data: Dict[int, Dict], total_frames: int) -> Dict[int, Tuple[int, int, int, int]]:
    """Interpolate crop parameters for all frames"""
    interpolated = {}
    
    valid_frames = sorted([f for f in frame_data.keys() if frame_data[f].get('crop')])
    
    if not valid_frames:
        return {}
    
    for frame_idx in range(total_frames):
        if frame_idx in frame_data and frame_data[frame_idx].get('crop'):
            interpolated[frame_idx] = frame_data[frame_idx]['crop']
        else:
            # Find nearest frames
            prev_frame = max([f for f in valid_frames if f <= frame_idx], default=None)
            next_frame = min([f for f in valid_frames if f > frame_idx], default=None)
            
            if prev_frame is not None and next_frame is not None:
                # Linear interpolation
                prev_crop = frame_data[prev_frame]['crop']
                next_crop = frame_data[next_frame]['crop']
                
                factor = (frame_idx - prev_frame) / (next_frame - prev_frame)
                
                crop_x = int(prev_crop[0] + factor * (next_crop[0] - prev_crop[0]))
                crop_y = int(prev_crop[1] + factor * (next_crop[1] - prev_crop[1]))
                crop_w = int(prev_crop[2] + factor * (next_crop[2] - prev_crop[2]))
                crop_h = int(prev_crop[3] + factor * (next_crop[3] - prev_crop[3]))
                
                interpolated[frame_idx] = (crop_x, crop_y, crop_w, crop_h)
            elif prev_frame is not None:
                interpolated[frame_idx] = frame_data[prev_frame]['crop']
            elif next_frame is not None:
                interpolated[frame_idx] = frame_data[next_frame]['crop']
    
    return interpolated


def generate_ffmpeg_crop_filter_from_data(crop_data: Dict[int, Tuple[int, int, int, int]], 
                                         frame_width: int, frame_height: int,
                                         total_frames: int, fps: float,
                                         output_width: int = 1080, output_height: int = 1920) -> str:
    """Generate ffmpeg filter from crop data with specified output dimensions"""
    if not crop_data:
        # Fallback to center crop
        aspect_ratio = output_width / output_height
        crop_width = int(frame_height * aspect_ratio)
        crop_x = (frame_width - crop_width) // 2
        logger.info(f"🔧 Using fallback center crop: {crop_width}x{frame_height} at ({crop_x}, 0)")
        return f"crop={crop_width}:{frame_height}:{crop_x}:0,scale={output_width}:{output_height}:force_original_aspect_ratio=decrease,pad={output_width}:{output_height}:(ow-iw)/2:(oh-ih)/2"
    
    # Log crop data statistics
    from collections import Counter
    crop_counter = Counter(crop_data.values())
    logger.info(f"🔧 Crop data: {len(crop_data)} frames with {len(crop_counter)} unique crops")
    
    # Create dynamic crop using keyframes for smooth transitions
    keyframes = []
    prev_crop = None
    
    # Sample keyframes every 30 frames or when crop changes significantly
    for frame_idx in sorted(crop_data.keys()):
        current_crop = crop_data[frame_idx]
        
        if prev_crop is None or crop_changed_significantly(prev_crop, current_crop):
            time_sec = frame_idx / fps
            crop_x, crop_y, crop_width, crop_height = current_crop
            keyframes.append(f"{time_sec}:crop={crop_width}:{crop_height}:{crop_x}:{crop_y}")
            prev_crop = current_crop
    
    logger.info(f"🔧 Generated {len(keyframes)} crop keyframes for dynamic tracking")
    
    if len(keyframes) <= 1:
        # Static crop if no significant changes
        most_common_crop = crop_counter.most_common(1)[0][0]
        crop_x, crop_y, crop_width, crop_height = most_common_crop
        logger.info(f"🔧 Using static crop: {crop_width}x{crop_height} at ({crop_x}, {crop_y})")
        filter_str = f"crop={crop_width}:{crop_height}:{crop_x}:{crop_y},scale={output_width}:{output_height}:force_original_aspect_ratio=decrease,pad={output_width}:{output_height}:(ow-iw)/2:(oh-ih)/2"
    else:
        # Dynamic crop with smooth transitions
        logger.info(f"🔧 Using dynamic crop with smooth transitions")
        # For now, use the most stable crop (this could be enhanced with zoompan filter)
        most_common_crop = crop_counter.most_common(1)[0][0]
        crop_x, crop_y, crop_width, crop_height = most_common_crop
        filter_str = f"crop={crop_width}:{crop_height}:{crop_x}:{crop_y},scale={output_width}:{output_height}:force_original_aspect_ratio=decrease,pad={output_width}:{output_height}:(ow-iw)/2:(oh-ih)/2"
    
    return filter_str


def crop_changed_significantly(crop1: Tuple[int, int, int, int], crop2: Tuple[int, int, int, int], threshold: int = 50) -> bool:
    """Check if crop position changed significantly"""
    x1, y1, w1, h1 = crop1
    x2, y2, w2, h2 = crop2
    
    # Check if center moved significantly
    center1_x, center1_y = x1 + w1//2, y1 + h1//2
    center2_x, center2_y = x2 + w2//2, y2 + h2//2
    
    distance = ((center2_x - center1_x)**2 + (center2_y - center1_y)**2)**0.5
    return distance > threshold


def apply_ffmpeg_processing(input_path: str, output_path: str, filter_complex: str,
                            start_time: float | None = None,
                            duration: float | None = None) -> bool:
    """Apply ffmpeg processing with the generated filter.
    Uses -ss/-t after -i to trim both audio and video for accurate A/V sync.
    """
    try:
        cmd = ["ffmpeg", "-y", "-i", input_path]

        # Trim both audio and video after decode for accuracy
        if start_time is not None and start_time > 0:
            cmd += ["-ss", str(start_time)]
        if duration is not None and duration > 0:
            cmd += ["-t", str(duration)]

        cmd += [
            "-vf", filter_complex,
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            "-shortest",
            output_path
        ]
        
        logger.info(f"Running ffmpeg...")
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        if result.returncode == 0:
            logger.info("FFmpeg processing completed")
            return True
        else:
            logger.error(f"FFmpeg failed: {result.returncode}")
            return False
    
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Error running ffmpeg: {e}")
        return False
