"""
Main processing pipeline for cheating detection.
Orchestrates all models and engines.
"""

import cv2
import numpy as np
import time
from typing import List, Tuple, Optional
import math

from config import CONFIG
from models.detector import PersonDetector, Detection
from models.head_pose import HeadPoseEstimator
from models.gaze_estimator import GazeEstimator
from models.hand_detector import HandDetector
from models.action_classifier import PoseActionClassifier
from core.fusion_engine import FusionEngine
from core.rule_engine import RuleEngine, Alert, AlertType


class CheatingDetectionPipeline:
    """
    Complete cheating detection pipeline.
    Processes video frames through all detection models and generates alerts.
    """
    
    def __init__(self, device: str = None):
        """
        Initialize the detection pipeline.
        
        Args:
            device: Device to use ("mps" for M2 Mac, "cpu" for fallback)
        """
        if device is None:
            device = CONFIG.model.device
        
        print("=" * 60)
        print("Initializing Cheating Detection Pipeline")
        print(f"Device: {device}")
        print("=" * 60)
        
        # Initialize detection models
        print("\n[1/5] Loading person detector...")
        self.detector = PersonDetector(
            model_path=CONFIG.model.yolo_model,
            device=device
        )
        
        print("[2/5] Loading head pose estimator...")
        self.head_pose = HeadPoseEstimator(
            max_faces=CONFIG.model.max_faces,
            min_detection_confidence=CONFIG.model.min_detection_confidence
        )
        
        print("[3/5] Loading gaze estimator...")
        self.gaze_estimator = GazeEstimator(
            max_faces=CONFIG.model.max_faces,
            min_detection_confidence=CONFIG.model.min_detection_confidence
        )
        
        print("[4/5] Loading hand detector...")
        self.hand_detector = HandDetector(
            max_hands=CONFIG.model.max_hands,
            min_detection_confidence=CONFIG.model.min_detection_confidence
        )
        
        print("[5/5] Loading action classifier...")
        self.action_classifier = PoseActionClassifier()
        
        # Initialize processing engines
        print("\nInitializing fusion and rule engines...")
        self.fusion = FusionEngine()
        self.rule_engine = RuleEngine(self.fusion)
        
        # Frame counter and timing
        self.frame_count = 0
        self.start_time = None
        
        print("\n" + "=" * 60)
        print("Pipeline Ready!")
        print("=" * 60 + "\n")
    
    def process_frame(
        self,
        frame: np.ndarray,
        draw_annotations: bool = True
    ) -> Tuple[np.ndarray, List[Alert]]:
        """
        Process a single frame through the complete pipeline.
        
        Args:
            frame: Input BGR image
            draw_annotations: Whether to draw visualizations
            
        Returns:
            Tuple of (annotated_frame, alerts)
        """
        self.frame_count += 1
        current_time = time.time()
        
        if self.start_time is None:
            self.start_time = current_time
        
        h, w = frame.shape[:2]
        
        # Skip frames if configured for performance
        if self.frame_count % CONFIG.model.process_every_n_frames != 0:
            return frame, []
        
        # ============================================
        # Step 1: Person Detection + Tracking
        # ============================================
        detections = self.detector.detect(
            frame,
            conf=CONFIG.model.yolo_conf,
            iou=CONFIG.model.yolo_iou
        )
        
        # ============================================
        # Step 2: Head Pose Estimation (batch)
        # ============================================
        head_poses = self.head_pose.estimate_batch(frame)
        
        # ============================================
        # Step 3: Gaze Estimation (batch)
        # ============================================
        gaze_results = self.gaze_estimator.estimate_batch(frame)
        
        # ============================================
        # Step 4: Hand Detection
        # ============================================
        all_hands = self.hand_detector.detect(frame)
        
        # ============================================
        # Step 5: Process each detected person
        # ============================================
        active_track_ids = []
        
        for detection in detections:
            track_id = detection.track_id
            active_track_ids.append(track_id)
            
            # Update fusion with detection
            self.fusion.update_detection(detection)
            
            # Match head pose to this person
            face_bbox = detection.face_bbox
            head_pose = self.head_pose.match_to_bbox(head_poses, face_bbox)
            
            if head_pose:
                yaw, pitch, roll = head_pose
                self.fusion.update_head_pose(track_id, yaw, pitch, roll)
            
            # Match gaze to this person
            gaze = self.gaze_estimator.match_to_bbox(gaze_results, face_bbox)
            
            if gaze:
                h_gaze, v_gaze = gaze
                self.fusion.update_gaze(track_id, h_gaze, v_gaze)
            
            # Match hands to this person
            person_hands = self.hand_detector.match_to_person(
                all_hands,
                detection.bbox
            )
            
            # Classify hand gestures
            gestures = [
                self.hand_detector.classify_gesture(hand, detection.bbox, h)
                for hand in person_hands
            ]
            
            self.fusion.update_hands(track_id, person_hands, gestures)
            
            # Classify body actions
            actions = self.action_classifier.classify(
                track_id,
                detection.keypoints
            )
            self.fusion.update_actions(track_id, actions)
        
        # Cleanup old tracks
        self.fusion.cleanup_old_tracks(active_track_ids)
        self.action_classifier.cleanup_old_tracks(active_track_ids)
        
        # ============================================
        # Step 6: Run rule engine
        # ============================================
        alerts = self.rule_engine.process(current_time)
        
        # ============================================
        # Step 7: Visualize (optional)
        # ============================================
        if draw_annotations:
            frame = self._visualize(frame, detections, alerts)
        
        return frame, alerts
    
    def _visualize(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        alerts: List[Alert]
    ) -> np.ndarray:
        """Draw annotations on frame."""
        
        frame = frame.copy()
        vis = CONFIG.visualization
        
        # Get alerted student IDs
        alerted_ids = set()
        for alert in alerts:
            alerted_ids.update(alert.student_ids)
        
        # Draw each detection
        for detection in detections:
            track_id = detection.track_id
            state = self.fusion.student_states.get(track_id)
            
            # Determine color
            if state and state.is_teacher:
                color = vis.color_teacher
            elif track_id in alerted_ids:
                color = vis.color_alert
            else:
                color = vis.color_normal
            
            x1, y1, x2, y2 = detection.bbox.astype(int)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, vis.bbox_thickness)
            
            # Draw label
            if state and state.is_teacher:
                label = "TEACHER"
            else:
                label = f"ID:{track_id}"
            
            cv2.putText(
                frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, vis.font_scale, color, vis.font_thickness
            )
            
            # Draw additional info
            if state and vis.show_head_pose and state.head_pose_valid:
                info_y = y2 + 20
                cv2.putText(
                    frame, f"Yaw:{state.head_yaw:.0f}",
                    (x1, info_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 0), 1
                )
            
            if state and vis.show_gaze and state.gaze_valid:
                info_y = y2 + 35
                cv2.putText(
                    frame, f"Gaze:{state.gaze_horizontal:.0f}",
                    (x1, info_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 0), 1
                )
            
            # Draw keypoints
            if vis.show_keypoints:
                for kpt in detection.keypoints:
                    if kpt[2] > 0.5:
                        cx, cy = int(kpt[0]), int(kpt[1])
                        cv2.circle(frame, (cx, cy), vis.keypoint_radius, color, -1)
        
        # Draw alerts panel
        if alerts and vis.show_info_panel:
            # Semi-transparent background
            panel_height = 30 + len(alerts) * 25
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (450, panel_height), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, vis.panel_alpha, frame, 1 - vis.panel_alpha, 0)
            
            # Draw alert text
            for i, alert in enumerate(alerts):
                text = f"{alert.alert_type.value}: Students {alert.student_ids}"
                
                # Color by severity
                if alert.severity.value == "high":
                    text_color = (0, 0, 255)
                elif alert.severity.value == "medium":
                    text_color = (0, 165, 255)
                else:
                    text_color = (255, 255, 255)
                
                cv2.putText(
                    frame, text, (15, 30 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1
                )
        
        # Draw student count
        student_count = self.fusion.get_student_count()
        cv2.putText(
            frame, f"Students: {student_count}",
            (frame.shape[1] - 150, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )
        
        return frame
    
    def set_teacher(self, track_id: int):
        """Manually designate the teacher."""
        self.fusion.set_teacher(track_id)
    
    def auto_detect_teacher(self):
        """Attempt to automatically detect the teacher."""
        teacher_id = self.fusion.identify_teacher_heuristic()
        if teacher_id is not None:
            self.fusion.set_teacher(teacher_id)
            print(f"Auto-detected teacher: ID {teacher_id}")
        else:
            print("Could not auto-detect teacher")
    
    def reset(self):
        """Reset all states and statistics."""
        self.fusion.reset()
        self.rule_engine.reset_statistics()
        self.detector.reset_tracking()
        self.frame_count = 0
        self.start_time = None
        print("Pipeline reset complete")
    
    def get_statistics(self) -> dict:
        """Get pipeline statistics."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        return {
            "frames_processed": self.frame_count,
            "elapsed_time": elapsed,
            "avg_fps": self.frame_count / elapsed if elapsed > 0 else 0,
            "students_tracked": self.fusion.get_student_count(),
            "alerts": self.rule_engine.get_statistics()
        }
    
    def close(self):
        """Release all resources."""
        self.head_pose.close()
        self.gaze_estimator.close()
        self.hand_detector.close()
        print("Pipeline resources released")
