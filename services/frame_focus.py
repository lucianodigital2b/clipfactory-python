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


def calculate_crop_parameters(speaker_bbox: Tuple[int, int, int, int], 
                             all_bboxes: List[Tuple[int, int, int, int]],
                             frame_width: int, frame_height: int, 
                             target_aspect_ratio: float = 9/16, 
                             zoom_factor: float = 1.2) -> Tuple[int, int, int, int]:
    """
    Calculate crop parameters focused on the active speaker
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
    
    # Apply zoom factor
    zoomed_width = int(speaker_width * zoom_factor)
    zoomed_height = int(speaker_height * zoom_factor)
    
    # Calculate crop dimensions for portrait aspect ratio
    crop_height = max(zoomed_height * 2, frame_height // 2)
    crop_width = int(crop_height * target_aspect_ratio)
    
    # Ensure crop doesn't exceed frame dimensions
    crop_width = min(crop_width, frame_width)
    crop_height = min(crop_height, frame_height)
    
    # Center crop around speaker
    crop_x = max(0, speaker_center_x - crop_width // 2)
    crop_y = max(0, speaker_center_y - crop_height // 2)
    
    # Adjust if crop goes beyond boundaries
    if crop_x + crop_width > frame_width:
        crop_x = frame_width - crop_width
    if crop_y + crop_height > frame_height:
        crop_y = frame_height - crop_height
    
    return (crop_x, crop_y, crop_width, crop_height)


def focus_on_speaker(input_path: str, output_path: str, fps_batch: int = 3, 
                    start_time: float = 0, duration: Optional[float] = None) -> str:
    """
    Uses audio-visual analysis to detect and track the active speaker.
    Combines YOLOv8 (person detection), MediaPipe (lip movement), and audio analysis.
    Outputs a portrait-oriented (1080x1920) video focused on the active speaker.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"[{timestamp}] 🎯 Starting audio-visual speaker detection")
    logger.info(f"[{timestamp}] 📁 Input: {input_path}")
    logger.info(f"[{timestamp}] 📁 Output: {output_path}")
    
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
        audio, sr = audio_analyzer.extract_audio_from_video(input_path)
        
        # Open video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {input_path}")
        
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
        
        logger.info(f"[{timestamp}] 🔍 Processing frames for speaker detection...")
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame_idx >= clip_total_frames:
                break
            
            # Process every fps_batch frames
            if frame_idx % fps_batch == 0:
                # Detect persons
                person_bboxes = detect_person_in_frame(yolo_model, frame, max_persons=2)
                
                if person_bboxes:
                    # Smooth bounding boxes
                    smoothed_bboxes = bbox_smoother.smooth_boxes(person_bboxes)
                    
                    # Get audio energy for this frame
                    audio_energy = frame_audio_energy.get(frame_idx, 0.0)
                    
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
                            target_aspect_ratio=9/16, zoom_factor=1.2
                        )
                    else:
                        # No clear speaker, use first person or center crop
                        speaker_bbox = smoothed_bboxes[0] if smoothed_bboxes else None
                        crop_params = calculate_crop_parameters(
                            speaker_bbox, smoothed_bboxes,
                            frame_width, frame_height,
                            target_aspect_ratio=9/16, zoom_factor=1.2
                        )
                    
                    frame_data[frame_idx] = {
                        'bboxes': smoothed_bboxes,
                        'speaker_idx': speaker_idx,
                        'crop': crop_params,
                        'audio_energy': audio_energy
                    }
                    
                    processed_frames += 1
                else:
                    # No persons detected - skip lip detection, use center crop for narration videos
                    audio_energy = frame_audio_energy.get(frame_idx, 0.0)
                    
                    # Center crop for videos without people (narration, slides, etc.)
                    crop_width = int(frame_height * 9/16)
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
        
        # Statistics
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
        
        # Interpolate missing frames
        logger.info(f"[{timestamp}] 🔄 Interpolating crop parameters...")
        interpolated_crops = interpolate_crop_parameters(frame_data, clip_total_frames)
        
        # Generate ffmpeg filter with clip timing
        logger.info(f"[{timestamp}] 🎬 Generating dynamic crop filter...")
        filter_complex = generate_ffmpeg_crop_filter_from_data(
            interpolated_crops, frame_width, frame_height, clip_total_frames, fps
        )
        
        # Add timing parameters for clip processing
        if start_time > 0 or duration is not None:
            timing_filter = f"[0:v]trim=start={start_time}"
            if duration is not None:
                timing_filter += f":duration={duration}"
            timing_filter += ",setpts=PTS-STARTPTS[trimmed];"
            filter_complex = timing_filter + filter_complex.replace("[0:v]", "[trimmed]")
        
        # Apply ffmpeg processing
        logger.info(f"[{timestamp}] 🎬 Applying video processing...")
        success = apply_ffmpeg_processing(input_path, output_path, filter_complex)
        
        if success:
            logger.info(f"[{timestamp}] ✅ Speaker-focused video created successfully!")
            logger.info(f"[{timestamp}] 📁 Output: {output_path}")
            return output_path
        else:
            raise RuntimeError("FFmpeg processing failed")
    
    except Exception as e:
        logger.error(f"[{timestamp}] ❌ Error: {e}")
        raise


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
                                         total_frames: int, fps: float) -> str:
    """Generate ffmpeg filter from crop data"""
    if not crop_data:
        # Fallback
        crop_width = int(frame_height * 9/16)
        crop_x = (frame_width - crop_width) // 2
        return f"crop={crop_width}:{frame_height}:{crop_x}:0,scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2"
    
    # Use most common crop as static (could be enhanced with keyframes)
    from collections import Counter
    crop_counter = Counter(crop_data.values())
    most_common_crop = crop_counter.most_common(1)[0][0]
    
    crop_x, crop_y, crop_width, crop_height = most_common_crop
    
    filter_str = f"crop={crop_width}:{crop_height}:{crop_x}:{crop_y},scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2"
    
    return filter_str


def apply_ffmpeg_processing(input_path: str, output_path: str, filter_complex: str) -> bool:
    """Apply ffmpeg processing with the generated filter"""
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-vf", filter_complex,
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
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
        logger.error(f"FFmpeg error: {e}")
        return False
    except Exception as e:
        logger.error(f"Error running ffmpeg: {e}")
        return False


# Example usage
if __name__ == "__main__":
    input_video = "input.mp4"
    output_video = "output_speaker_focused.mp4"
    
    try:
        result = focus_on_speaker(input_video, output_video, fps_batch=3)
        print(f"Success! Output saved to: {result}")
    except Exception as e:
        print(f"Error: {e}")