#!/usr/bin/env python3
"""
Live speaker detection viewer for frame_focus.py
Shows real-time detection with bounding boxes and speaker identification
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from services.frame_focus import (
    detect_person_in_frame, 
    LipMovementDetector, 
    SpeakerDetector, 
    AudioAnalyzer,
    BoundingBoxSmoother,
    calculate_crop_parameters
)
from ultralytics import YOLO


class LiveDetectionViewer:
    """Live viewer for speaker detection with visual feedback"""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Video: {self.frame_width}x{self.frame_height}, {self.fps:.1f} FPS, {self.total_frames} frames")
        
        # Initialize detection components
        print("🤖 Loading YOLO model...")
        self.yolo_model = YOLO("yolov8n.pt")
        
        print("👄 Initializing lip detector...")
        self.lip_detector = LipMovementDetector()
        
        print("🎯 Initializing speaker detector...")
        self.speaker_detector = SpeakerDetector(audio_weight=0.6, lip_weight=0.4)
        
        print("📦 Initializing bbox smoother...")
        self.bbox_smoother = BoundingBoxSmoother(smoothing_factor=0.3)
        
        # Extract audio for voice activity
        print("🎵 Extracting audio...")
        self.audio_analyzer = AudioAnalyzer()
        self.audio, self.sr = self.audio_analyzer.extract_audio_from_video(video_path)
        self.frame_audio_energy = self.audio_analyzer.detect_voice_activity(self.audio, self.sr, self.fps)
        
        # Colors for visualization
        self.colors = [
            (0, 255, 0),    # Green for speaker
            (255, 0, 0),    # Blue for non-speaker
            (0, 255, 255),  # Yellow for person 2
            (255, 0, 255),  # Magenta for person 3
        ]
        
        # Detection stats
        self.frame_count = 0
        self.detection_stats = {
            'frames_processed': 0,
            'frames_with_people': 0,
            'speaker_detected': 0,
            'lip_movement_detected': 0
        }
    
    def draw_detection_info(self, frame: np.ndarray, person_boxes: list, 
                           speaker_idx: int, lip_scores: dict, 
                           audio_energy: float, crop_params: tuple) -> np.ndarray:
        """Draw detection information on frame"""
        display_frame = frame.copy()
        
        # Draw person bounding boxes
        for i, bbox in enumerate(person_boxes):
            x1, y1, x2, y2 = bbox
            
            # Choose color based on speaker status
            if i == speaker_idx:
                color = self.colors[0]  # Green for active speaker
                label = f"SPEAKER {i}"
                thickness = 3
            else:
                color = self.colors[1]  # Blue for non-speaker
                label = f"Person {i}"
                thickness = 2
            
            # Draw bounding box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Add lip movement score if available
            lip_score = lip_scores.get(i, 0.0)
            label += f" (Lip: {lip_score:.3f})"
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(display_frame, (x1, y1 - 25), (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(display_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw crop preview if speaker detected
        if speaker_idx is not None and crop_params:
            crop_x, crop_y, crop_w, crop_h = crop_params
            cv2.rectangle(display_frame, (crop_x, crop_y), 
                         (crop_x + crop_w, crop_y + crop_h), 
                         (0, 255, 255), 2)  # Yellow crop preview
            cv2.putText(display_frame, "CROP AREA", (crop_x, crop_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Draw info panel
        info_y = 30
        info_texts = [
            f"Frame: {self.frame_count}/{self.total_frames}",
            f"People: {len(person_boxes)}",
            f"Speaker: {'Person ' + str(speaker_idx) if speaker_idx is not None else 'None'}",
            f"Audio Energy: {audio_energy:.3f}",
            f"Detection Rate: {self.detection_stats['frames_with_people']}/{self.detection_stats['frames_processed']} ({100*self.detection_stats['frames_with_people']/max(1,self.detection_stats['frames_processed']):.1f}%)"
        ]
        
        for text in info_texts:
            cv2.putText(display_frame, text, (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, text, (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            info_y += 25
        
        return display_frame
    
    def run_live_detection(self, start_time: float = 0, playback_speed: float = 1.0):
        """Run live detection viewer"""
        print(f"\n🎬 Starting live detection viewer...")
        print("Controls:")
        print("  SPACE: Pause/Resume")
        print("  ESC/Q: Quit")
        print("  R: Reset to beginning")
        print("  +/-: Adjust playback speed")
        print("  Click: Jump to position")
        print()
        
        # Set start position
        if start_time > 0:
            start_frame = int(start_time * self.fps)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            self.frame_count = start_frame
        
        paused = False
        last_time = time.time()
        
        while True:
            if not paused:
                ret, frame = self.cap.read()
                if not ret:
                    print("📹 End of video reached")
                    break
                
                self.frame_count += 1
                
                # Process every 2nd frame for better performance
                if self.frame_count % 2 != 0:
                    continue
                
                self.detection_stats['frames_processed'] += 1
                
                # Get audio energy for current frame
                audio_energy = self.frame_audio_energy.get(self.frame_count, 0.0)
                
                # Detect persons
                person_boxes = detect_person_in_frame(
                    self.yolo_model, frame, 
                    confidence_threshold=0.5, max_persons=2
                )
                
                # Smooth bounding boxes
                person_boxes = self.bbox_smoother.smooth_boxes(person_boxes)
                
                if person_boxes:
                    self.detection_stats['frames_with_people'] += 1
                    
                    # Detect lip movement
                    lip_scores = self.lip_detector.detect_lip_movement(frame, person_boxes)
                    
                    if lip_scores:
                        self.detection_stats['lip_movement_detected'] += 1
                    
                    # Identify speaker
                    speaker_idx = self.speaker_detector.identify_speaker(
                        self.frame_count, audio_energy, lip_scores
                    )
                    
                    if speaker_idx is not None:
                        self.detection_stats['speaker_detected'] += 1
                        
                        # Calculate crop parameters for preview
                        speaker_bbox = person_boxes[speaker_idx] if speaker_idx < len(person_boxes) else None
                        crop_params = calculate_crop_parameters(
                            speaker_bbox, person_boxes, 
                            self.frame_width, self.frame_height,
                            target_aspect_ratio=9/16, zoom_factor=2.5
                        ) if speaker_bbox else None
                    else:
                        crop_params = None
                else:
                    lip_scores = {}
                    speaker_idx = None
                    crop_params = None
                
                # Draw detection info
                display_frame = self.draw_detection_info(
                    frame, person_boxes, speaker_idx, 
                    lip_scores, audio_energy, crop_params
                )
                
                # Resize for display if too large
                display_height = 720
                if display_frame.shape[0] > display_height:
                    scale = display_height / display_frame.shape[0]
                    new_width = int(display_frame.shape[1] * scale)
                    display_frame = cv2.resize(display_frame, (new_width, display_height))
                
                cv2.imshow('Live Speaker Detection', display_frame)
                
                # Control playback speed
                if playback_speed > 0:
                    frame_delay = (1.0 / self.fps) / playback_speed
                    current_time = time.time()
                    elapsed = current_time - last_time
                    if elapsed < frame_delay:
                        time.sleep(frame_delay - elapsed)
                    last_time = time.time()
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # ESC or Q
                break
            elif key == ord(' '):  # SPACE
                paused = not paused
                print(f"{'⏸️  Paused' if paused else '▶️  Resumed'}")
            elif key == ord('r'):  # R - Reset
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.frame_count = 0
                print("🔄 Reset to beginning")
            elif key == ord('+') or key == ord('='):  # Speed up
                playback_speed = min(playback_speed * 1.5, 5.0)
                print(f"⚡ Speed: {playback_speed:.1f}x")
            elif key == ord('-'):  # Slow down
                playback_speed = max(playback_speed / 1.5, 0.1)
                print(f"🐌 Speed: {playback_speed:.1f}x")
        
        # Print final stats
        print(f"\n📊 Final Detection Statistics:")
        print(f"   Frames processed: {self.detection_stats['frames_processed']}")
        print(f"   Frames with people: {self.detection_stats['frames_with_people']}")
        print(f"   Speaker detected: {self.detection_stats['speaker_detected']}")
        print(f"   Lip movement detected: {self.detection_stats['lip_movement_detected']}")
        print(f"   Detection rate: {100*self.detection_stats['frames_with_people']/max(1,self.detection_stats['frames_processed']):.1f}%")
        print(f"   Speaker rate: {100*self.detection_stats['speaker_detected']/max(1,self.detection_stats['frames_with_people']):.1f}%")
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        self.lip_detector.cleanup()


def main():
    """Main function to run live detection viewer"""
    print("🎯 Live Speaker Detection Viewer")
    print("=" * 50)
    
    # Use the sample video
    sample_files_dir = Path(__file__).resolve().parent / 'sample-files'
    video_file = sample_files_dir / "sample-video.mp4"
    
    if not video_file.exists():
        print(f"❌ Sample video not found: {video_file}")
        print("Please make sure the sample video exists in the sample-files directory")
        return
    
    try:
        viewer = LiveDetectionViewer(str(video_file))
        
        # Start from 60 seconds like the test with faster playback
        viewer.run_live_detection(start_time=60, playback_speed=3.0)
        
    except KeyboardInterrupt:
        print("\n⏹️  Stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()