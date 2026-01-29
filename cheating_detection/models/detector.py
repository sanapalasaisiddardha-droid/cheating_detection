"""
YOLOv8-Pose detection with ByteTrack tracking.
Optimized for M2 MacBook Air.
"""

import torch
import numpy as np
from ultralytics import YOLO
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Detection:
    """Single person detection with pose keypoints."""
    
    track_id: int
    bbox: np.ndarray              # [x1, y1, x2, y2]
    keypoints: np.ndarray         # [17, 3] - x, y, confidence
    confidence: float
    
    # COCO keypoint indices
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get bounding box center."""
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2
        )
    
    @property
    def width(self) -> float:
        """Get bounding box width."""
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self) -> float:
        """Get bounding box height."""
        return self.bbox[3] - self.bbox[1]
    
    @property
    def face_bbox(self) -> np.ndarray:
        """
        Estimate face bounding box from keypoints.
        Returns [x1, y1, x2, y2]
        """
        # Face keypoints: nose, eyes, ears
        face_indices = [self.NOSE, self.LEFT_EYE, self.RIGHT_EYE, 
                       self.LEFT_EAR, self.RIGHT_EAR]
        face_kpts = self.keypoints[face_indices]
        
        # Filter by confidence
        valid_mask = face_kpts[:, 2] > 0.5
        
        if valid_mask.sum() < 2:
            # Fallback: use top portion of person bbox
            x1, y1, x2, y2 = self.bbox
            face_height = (y2 - y1) * 0.25
            return np.array([x1, y1, x2, y1 + face_height])
        
        valid_kpts = face_kpts[valid_mask]
        
        x_min = valid_kpts[:, 0].min()
        x_max = valid_kpts[:, 0].max()
        y_min = valid_kpts[:, 1].min()
        y_max = valid_kpts[:, 1].max()
        
        # Expand face box with padding
        w = x_max - x_min
        h = y_max - y_min
        padding = max(w, h) * 0.5
        
        return np.array([
            max(0, x_min - padding),
            max(0, y_min - padding),
            x_max + padding,
            y_max + padding
        ])
    
    @property
    def nose_position(self) -> Optional[Tuple[float, float]]:
        """Get nose position if visible."""
        if self.keypoints[self.NOSE, 2] > 0.5:
            return tuple(self.keypoints[self.NOSE, :2])
        return None
    
    @property
    def shoulder_center(self) -> Optional[Tuple[float, float]]:
        """Get center point between shoulders."""
        left = self.keypoints[self.LEFT_SHOULDER]
        right = self.keypoints[self.RIGHT_SHOULDER]
        
        if left[2] > 0.3 and right[2] > 0.3:
            return (
                (left[0] + right[0]) / 2,
                (left[1] + right[1]) / 2
            )
        return None
    
    @property
    def hip_center(self) -> Optional[Tuple[float, float]]:
        """Get center point between hips."""
        left = self.keypoints[self.LEFT_HIP]
        right = self.keypoints[self.RIGHT_HIP]
        
        if left[2] > 0.3 and right[2] > 0.3:
            return (
                (left[0] + right[0]) / 2,
                (left[1] + right[1]) / 2
            )
        return None
    
    def get_wrist_positions(self) -> List[Tuple[float, float]]:
        """Get visible wrist positions."""
        wrists = []
        
        for idx in [self.LEFT_WRIST, self.RIGHT_WRIST]:
            if self.keypoints[idx, 2] > 0.3:
                wrists.append(tuple(self.keypoints[idx, :2]))
        
        return wrists


class PersonDetector:
    """
    YOLOv8-Pose detector with ByteTrack tracking.
    Optimized for M2 MacBook Air using MPS backend.
    """
    
    KEYPOINT_NAMES = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    def __init__(self, model_path: str = "yolov8s-pose.pt", device: str = "mps"):
        """
        Initialize the person detector.
        
        Args:
            model_path: Path to YOLOv8-Pose model weights
            device: Device to run inference on ("mps" for M2 Mac)
        """
        self.device = device
        
        # Load YOLO model
        print(f"[PersonDetector] Loading {model_path}...")
        self.model = YOLO(model_path)
        
        # Move to appropriate device
        if device == "mps" and torch.backends.mps.is_available():
            self.model.to("mps")
            print(f"[PersonDetector] Using MPS (Metal Performance Shaders)")
        else:
            self.model.to("cpu")
            print(f"[PersonDetector] Using CPU")
        
        print(f"[PersonDetector] Initialized successfully")
    
    def detect(
        self, 
        frame: np.ndarray, 
        conf: float = 0.5,
        iou: float = 0.45
    ) -> List[Detection]:
        """
        Detect and track persons in frame.
        
        Args:
            frame: Input BGR image
            conf: Confidence threshold
            iou: IoU threshold for NMS
            
        Returns:
            List of Detection objects
        """
        # Run YOLO with tracking
        results = self.model.track(
            frame,
            persist=True,          # Maintain tracking across frames
            conf=conf,
            iou=iou,
            verbose=False,
            classes=[0]            # Only detect persons (class 0)
        )
        
        detections = []
        
        # Check if we have any detections
        if results[0].boxes is None or len(results[0].boxes) == 0:
            return detections
        
        # Extract bounding boxes
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        
        # Extract tracking IDs
        track_ids = results[0].boxes.id
        if track_ids is not None:
            track_ids = track_ids.cpu().numpy().astype(int)
        else:
            # Fallback if tracking fails
            track_ids = np.arange(len(boxes))
        
        # Extract keypoints
        if results[0].keypoints is not None:
            keypoints = results[0].keypoints.data.cpu().numpy()
        else:
            # Empty keypoints if not available
            keypoints = np.zeros((len(boxes), 17, 3))
        
        # Create Detection objects
        for i, (box, track_id, conf, kpts) in enumerate(
            zip(boxes, track_ids, confs, keypoints)
        ):
            detections.append(Detection(
                track_id=int(track_id),
                bbox=box.astype(np.float32),
                keypoints=kpts.astype(np.float32),
                confidence=float(conf)
            ))
        
        return detections
    
    def reset_tracking(self):
        """Reset the tracker state."""
        self.model.predictor = None


if __name__ == "__main__":
    # Test the detector
    import cv2
    
    detector = PersonDetector()
    
    # Create a test image
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        detections = detector.detect(frame)
        print(f"Detected {len(detections)} persons")
        for det in detections:
            print(f"  ID: {det.track_id}, Conf: {det.confidence:.2f}, "
                  f"BBox: {det.bbox}")
