"""
Lightweight action classification from body pose keypoints.
Rule-based approach optimized for M2 MacBook Air.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import deque
from enum import Enum
import math


class Action(Enum):
    """Body action classifications."""
    NORMAL = "normal"
    LEANING_LEFT = "leaning_left"
    LEANING_RIGHT = "leaning_right"
    LEANING_FORWARD = "leaning_forward"
    LEANING_BACKWARD = "leaning_backward"
    TURNING_BODY = "turning_body"
    REACHING_LEFT = "reaching_left"
    REACHING_RIGHT = "reaching_right"
    SLOUCHING = "slouching"
    STANDING = "standing"


class PoseActionClassifier:
    """
    Rule-based action classification from YOLO pose keypoints.
    Lightweight alternative to ST-GCN for M2 Mac.
    """
    
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
    
    def __init__(self, history_length: int = 15):
        """
        Initialize the action classifier.
        
        Args:
            history_length: Number of frames to keep for temporal analysis
        """
        self.history: Dict[int, deque] = {}
        self.history_length = history_length
        
        print(f"[PoseActionClassifier] Initialized with {history_length} frame history")
    
    def _get_keypoint(self, keypoints: np.ndarray, idx: int) -> Optional[np.ndarray]:
        """Get keypoint if confidence is sufficient."""
        if keypoints[idx, 2] > 0.3:
            return keypoints[idx, :2]
        return None
    
    def _get_shoulder_angle(self, keypoints: np.ndarray) -> float:
        """
        Calculate shoulder tilt angle (roll).
        Positive = right shoulder higher
        """
        left = self._get_keypoint(keypoints, self.LEFT_SHOULDER)
        right = self._get_keypoint(keypoints, self.RIGHT_SHOULDER)
        
        if left is None or right is None:
            return 0.0
        
        dx = right[0] - left[0]
        dy = right[1] - left[1]
        
        if abs(dx) < 1:
            return 0.0
        
        angle = math.degrees(math.atan2(dy, dx))
        return angle
    
    def _get_shoulder_width(self, keypoints: np.ndarray) -> float:
        """Get shoulder width in pixels."""
        left = self._get_keypoint(keypoints, self.LEFT_SHOULDER)
        right = self._get_keypoint(keypoints, self.RIGHT_SHOULDER)
        
        if left is None or right is None:
            return 100  # Default value
        
        return np.linalg.norm(right - left)
    
    def _get_body_lean(self, keypoints: np.ndarray) -> Tuple[float, float]:
        """
        Calculate body lean angles.
        
        Returns:
            (lateral_lean, forward_lean) in degrees
            Positive lateral = leaning right
            Positive forward = leaning forward
        """
        # Get shoulder and hip centers
        left_shoulder = self._get_keypoint(keypoints, self.LEFT_SHOULDER)
        right_shoulder = self._get_keypoint(keypoints, self.RIGHT_SHOULDER)
        left_hip = self._get_keypoint(keypoints, self.LEFT_HIP)
        right_hip = self._get_keypoint(keypoints, self.RIGHT_HIP)
        
        if any(p is None for p in [left_shoulder, right_shoulder, left_hip, right_hip]):
            return 0.0, 0.0
        
        shoulder_center = (left_shoulder + right_shoulder) / 2
        hip_center = (left_hip + right_hip) / 2
        
        # Lateral lean (shoulder offset from hip)
        shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)
        if shoulder_width < 10:
            return 0.0, 0.0
        
        lateral_offset = shoulder_center[0] - hip_center[0]
        lateral_lean = (lateral_offset / shoulder_width) * 45  # Scale to degrees
        
        # Forward lean (based on vertical compression)
        vertical_dist = hip_center[1] - shoulder_center[1]
        expected_dist = shoulder_width * 1.3  # Expected upright ratio
        
        if expected_dist > 0:
            compression_ratio = (expected_dist - vertical_dist) / expected_dist
            forward_lean = compression_ratio * 30  # Scale to degrees
        else:
            forward_lean = 0.0
        
        return lateral_lean, forward_lean
    
    def _get_arm_extension(self, keypoints: np.ndarray) -> Tuple[float, float]:
        """
        Calculate arm extension ratios.
        
        Returns:
            (left_extension, right_extension) as ratios of shoulder width
        """
        shoulder_width = self._get_shoulder_width(keypoints)
        if shoulder_width < 10:
            return 0.0, 0.0
        
        left_shoulder = self._get_keypoint(keypoints, self.LEFT_SHOULDER)
        right_shoulder = self._get_keypoint(keypoints, self.RIGHT_SHOULDER)
        left_wrist = self._get_keypoint(keypoints, self.LEFT_WRIST)
        right_wrist = self._get_keypoint(keypoints, self.RIGHT_WRIST)
        
        left_ext = 0.0
        right_ext = 0.0
        
        if left_shoulder is not None and left_wrist is not None:
            left_dist = np.linalg.norm(left_wrist - left_shoulder)
            left_ext = left_dist / shoulder_width
        
        if right_shoulder is not None and right_wrist is not None:
            right_dist = np.linalg.norm(right_wrist - right_shoulder)
            right_ext = right_dist / shoulder_width
        
        return left_ext, right_ext
    
    def _is_standing(self, keypoints: np.ndarray) -> bool:
        """Check if person is standing based on pose."""
        left_hip = self._get_keypoint(keypoints, self.LEFT_HIP)
        left_knee = self._get_keypoint(keypoints, self.LEFT_KNEE)
        left_ankle = self._get_keypoint(keypoints, self.LEFT_ANKLE)
        
        right_hip = self._get_keypoint(keypoints, self.RIGHT_HIP)
        right_knee = self._get_keypoint(keypoints, self.RIGHT_KNEE)
        right_ankle = self._get_keypoint(keypoints, self.RIGHT_ANKLE)
        
        # Check left leg
        if left_hip is not None and left_knee is not None:
            hip_knee_dist = abs(left_knee[1] - left_hip[1])
            
            # Standing: significant vertical distance
            if hip_knee_dist > 80:
                return True
        
        # Check right leg
        if right_hip is not None and right_knee is not None:
            hip_knee_dist = abs(right_knee[1] - right_hip[1])
            
            if hip_knee_dist > 80:
                return True
        
        return False
    
    def _get_body_orientation(self, keypoints: np.ndarray) -> float:
        """
        Estimate body orientation (yaw) from shoulder positions.
        Returns angle in degrees. 0 = facing camera.
        """
        left_shoulder = self._get_keypoint(keypoints, self.LEFT_SHOULDER)
        right_shoulder = self._get_keypoint(keypoints, self.RIGHT_SHOULDER)
        
        if left_shoulder is None or right_shoulder is None:
            return 0.0
        
        # Shoulder width in image
        apparent_width = abs(right_shoulder[0] - left_shoulder[0])
        
        # Use depth approximation from shoulder y-positions
        depth_diff = right_shoulder[1] - left_shoulder[1]
        
        # Estimate rotation from apparent width compression
        shoulder_width = self._get_shoulder_width(keypoints)
        if shoulder_width < 10:
            return 0.0
        
        # When body turns, apparent width decreases
        width_ratio = apparent_width / shoulder_width
        
        # Clamp and convert to angle
        width_ratio = max(0.1, min(1.0, width_ratio))
        yaw = math.degrees(math.acos(width_ratio))
        
        # Determine direction from depth difference
        if depth_diff > 5:
            yaw = -yaw  # Turned left
        
        return yaw
    
    def update_history(self, track_id: int, keypoints: np.ndarray):
        """Update pose history for a tracked person."""
        if track_id not in self.history:
            self.history[track_id] = deque(maxlen=self.history_length)
        
        self.history[track_id].append(keypoints.copy())
    
    def classify(
        self,
        track_id: int,
        keypoints: np.ndarray,
        lean_threshold: float = 15.0,
        arm_threshold: float = 1.5
    ) -> List[Action]:
        """
        Classify current actions based on pose.
        
        Args:
            track_id: Person tracking ID
            keypoints: [17, 3] keypoint array
            lean_threshold: Angle threshold for lean detection
            arm_threshold: Extension ratio threshold for reaching
            
        Returns:
            List of detected actions
        """
        self.update_history(track_id, keypoints)
        
        actions = []
        
        # Calculate pose metrics
        lateral_lean, forward_lean = self._get_body_lean(keypoints)
        shoulder_angle = self._get_shoulder_angle(keypoints)
        left_arm_ext, right_arm_ext = self._get_arm_extension(keypoints)
        body_yaw = self._get_body_orientation(keypoints)
        
        # Standing check
        if self._is_standing(keypoints):
            actions.append(Action.STANDING)
        
        # Lateral lean detection
        if lateral_lean > lean_threshold:
            actions.append(Action.LEANING_RIGHT)
        elif lateral_lean < -lean_threshold:
            actions.append(Action.LEANING_LEFT)
        
        # Forward/backward lean
        if forward_lean > lean_threshold:
            actions.append(Action.LEANING_FORWARD)
        elif forward_lean < -lean_threshold:
            actions.append(Action.LEANING_BACKWARD)
        
        # Body turning
        if abs(body_yaw) > 20:
            actions.append(Action.TURNING_BODY)
        
        # Arm reaching
        if left_arm_ext > arm_threshold:
            actions.append(Action.REACHING_LEFT)
        if right_arm_ext > arm_threshold:
            actions.append(Action.REACHING_RIGHT)
        
        # Slouching (shoulders rolled forward)
        if abs(shoulder_angle) > 12:
            actions.append(Action.SLOUCHING)
        
        # Default to normal if no actions detected
        if not actions:
            actions.append(Action.NORMAL)
        
        return actions
    
    def get_movement_intensity(self, track_id: int) -> float:
        """
        Calculate how much the person has moved recently.
        
        Args:
            track_id: Person tracking ID
            
        Returns:
            Movement intensity (higher = more movement)
        """
        if track_id not in self.history or len(self.history[track_id]) < 2:
            return 0.0
        
        history = list(self.history[track_id])
        
        total_movement = 0.0
        count = 0
        
        for i in range(1, len(history)):
            prev = history[i-1]
            curr = history[i]
            
            # Compare nose positions
            if prev[self.NOSE, 2] > 0.3 and curr[self.NOSE, 2] > 0.3:
                movement = np.linalg.norm(
                    curr[self.NOSE, :2] - prev[self.NOSE, :2]
                )
                total_movement += movement
                count += 1
        
        return total_movement / max(count, 1)
    
    def detect_sudden_movement(
        self,
        track_id: int,
        threshold: float = 50.0
    ) -> bool:
        """
        Detect sudden/jerky movements.
        
        Args:
            track_id: Person tracking ID
            threshold: Movement threshold
            
        Returns:
            True if sudden movement detected
        """
        if track_id not in self.history or len(self.history[track_id]) < 3:
            return False
        
        history = list(self.history[track_id])
        
        # Get last 3 frames
        prev2 = history[-3]
        prev1 = history[-2]
        curr = history[-1]
        
        # Calculate movement deltas
        if (prev2[self.NOSE, 2] > 0.3 and 
            prev1[self.NOSE, 2] > 0.3 and 
            curr[self.NOSE, 2] > 0.3):
            
            delta1 = np.linalg.norm(prev1[self.NOSE, :2] - prev2[self.NOSE, :2])
            delta2 = np.linalg.norm(curr[self.NOSE, :2] - prev1[self.NOSE, :2])
            
            # Sudden movement = large acceleration
            acceleration = abs(delta2 - delta1)
            
            return acceleration > threshold
        
        return False
    
    def cleanup_old_tracks(self, active_ids: List[int]):
        """Remove history for tracks no longer active."""
        to_remove = [tid for tid in self.history if tid not in active_ids]
        for tid in to_remove:
            del self.history[tid]


if __name__ == "__main__":
    # Test the classifier
    classifier = PoseActionClassifier()
    
    # Create mock keypoints
    mock_keypoints = np.zeros((17, 3))
    
    # Set some confidence values
    mock_keypoints[:, 2] = 0.9
    
    # Set positions (normalized to approximate real positions)
    mock_keypoints[0] = [320, 100, 0.9]   # Nose
    mock_keypoints[5] = [280, 200, 0.9]   # Left shoulder
    mock_keypoints[6] = [360, 200, 0.9]   # Right shoulder
    mock_keypoints[11] = [290, 350, 0.9]  # Left hip
    mock_keypoints[12] = [350, 350, 0.9]  # Right hip
    
    actions = classifier.classify(0, mock_keypoints)
    print(f"Detected actions: {[a.value for a in actions]}")
