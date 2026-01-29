"""
Head pose estimation using MediaPipe Face Mesh.
Optimized for M2 MacBook Air - uses geometric calculation from landmarks.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Tuple, Optional, List
import math


class HeadPoseEstimator:
    """
    Head pose estimation using MediaPipe Face Mesh.
    Extracts yaw, pitch, roll from 3D facial landmarks.
    """
    
    # Key landmark indices for pose estimation
    NOSE_TIP = 1
    CHIN = 152
    LEFT_EYE_LEFT_CORNER = 33
    RIGHT_EYE_RIGHT_CORNER = 263
    LEFT_MOUTH_CORNER = 61
    RIGHT_MOUTH_CORNER = 291
    
    # 3D model points for solvePnP (standard face model)
    MODEL_POINTS = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ], dtype=np.float64)
    
    LANDMARK_INDICES = [1, 152, 33, 263, 61, 291]
    
    def __init__(self, max_faces: int = 15, min_detection_confidence: float = 0.5):
        """
        Initialize the head pose estimator.
        
        Args:
            max_faces: Maximum number of faces to detect
            min_detection_confidence: Minimum confidence for face detection
        """
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_faces,
            refine_landmarks=True,  # Enable iris landmarks
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5
        )
        
        # Camera matrix (will be set based on frame size)
        self._camera_matrix = None
        self._dist_coeffs = np.zeros((4, 1))
        self._frame_size = None
        
        print(f"[HeadPoseEstimator] Initialized for max {max_faces} faces")
    
    def _update_camera_matrix(self, width: int, height: int):
        """Update camera matrix based on frame dimensions."""
        if self._frame_size != (width, height):
            self._frame_size = (width, height)
            focal_length = width
            center = (width / 2, height / 2)
            
            self._camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)
    
    def _rotation_matrix_to_euler(self, rotation_matrix: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert rotation matrix to Euler angles.
        
        Returns:
            (yaw, pitch, roll) in degrees
        """
        sy = math.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)
        
        singular = sy < 1e-6
        
        if not singular:
            x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            y = math.atan2(-rotation_matrix[2, 0], sy)
            z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            x = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            y = math.atan2(-rotation_matrix[2, 0], sy)
            z = 0
        
        # Convert to degrees
        pitch = math.degrees(x)
        yaw = math.degrees(y)
        roll = math.degrees(z)
        
        return yaw, pitch, roll
    
    def estimate_batch(self, frame: np.ndarray) -> Dict[Tuple[float, float], Tuple[float, float, float]]:
        """
        Estimate head pose for all faces in frame.
        
        Args:
            frame: Input BGR image
            
        Returns:
            Dictionary mapping face center (x, y) to (yaw, pitch, roll) in degrees
        """
        h, w = frame.shape[:2]
        self._update_camera_matrix(w, h)
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.face_mesh.process(rgb_frame)
        
        poses = {}
        
        if not results.multi_face_landmarks:
            return poses
        
        for face_landmarks in results.multi_face_landmarks:
            try:
                # Extract 2D image points
                image_points = np.array([
                    (face_landmarks.landmark[idx].x * w,
                     face_landmarks.landmark[idx].y * h)
                    for idx in self.LANDMARK_INDICES
                ], dtype=np.float64)
                
                # Solve PnP to get rotation and translation vectors
                success, rotation_vec, translation_vec = cv2.solvePnP(
                    self.MODEL_POINTS,
                    image_points,
                    self._camera_matrix,
                    self._dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                
                if success:
                    # Convert rotation vector to matrix
                    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
                    
                    # Convert to Euler angles
                    yaw, pitch, roll = self._rotation_matrix_to_euler(rotation_mat)
                    
                    # Get face center for matching
                    nose = face_landmarks.landmark[self.NOSE_TIP]
                    face_center = (nose.x * w, nose.y * h)
                    
                    poses[face_center] = (yaw, pitch, roll)
                    
            except Exception as e:
                continue
        
        return poses
    
    def match_to_bbox(
        self,
        poses: Dict[Tuple[float, float], Tuple[float, float, float]],
        face_bbox: np.ndarray,
        max_distance: Optional[float] = None
    ) -> Optional[Tuple[float, float, float]]:
        """
        Match a pose result to a face bounding box.
        
        Args:
            poses: Dictionary of poses from estimate_batch
            face_bbox: Face bounding box [x1, y1, x2, y2]
            max_distance: Maximum allowed distance for matching
            
        Returns:
            (yaw, pitch, roll) or None if no match found
        """
        if not poses:
            return None
        
        bbox_center_x = (face_bbox[0] + face_bbox[2]) / 2
        bbox_center_y = (face_bbox[1] + face_bbox[3]) / 2
        bbox_size = max(face_bbox[2] - face_bbox[0], face_bbox[3] - face_bbox[1])
        
        if max_distance is None:
            max_distance = bbox_size
        
        min_dist = float('inf')
        best_pose = None
        
        for face_center, pose in poses.items():
            dist = math.sqrt(
                (face_center[0] - bbox_center_x)**2 +
                (face_center[1] - bbox_center_y)**2
            )
            
            if dist < min_dist and dist < max_distance:
                min_dist = dist
                best_pose = pose
        
        return best_pose
    
    def get_head_direction_vector(
        self,
        yaw: float,
        pitch: float
    ) -> Tuple[float, float, float]:
        """
        Convert head pose to a 3D direction vector.
        
        Args:
            yaw: Yaw angle in degrees
            pitch: Pitch angle in degrees
            
        Returns:
            Normalized (x, y, z) direction vector
        """
        yaw_rad = math.radians(yaw)
        pitch_rad = math.radians(pitch)
        
        x = math.sin(yaw_rad) * math.cos(pitch_rad)
        y = math.sin(pitch_rad)
        z = math.cos(yaw_rad) * math.cos(pitch_rad)
        
        # Normalize
        length = math.sqrt(x**2 + y**2 + z**2)
        if length > 0:
            x, y, z = x/length, y/length, z/length
        
        return (x, y, z)
    
    def close(self):
        """Release resources."""
        self.face_mesh.close()


if __name__ == "__main__":
    # Test the head pose estimator
    import cv2
    
    estimator = HeadPoseEstimator()
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        poses = estimator.estimate_batch(frame)
        
        for face_center, (yaw, pitch, roll) in poses.items():
            cx, cy = int(face_center[0]), int(face_center[1])
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            cv2.putText(frame, f"Y:{yaw:.0f} P:{pitch:.0f} R:{roll:.0f}",
                       (cx - 50, cy - 20), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (0, 255, 0), 1)
        
        cv2.imshow("Head Pose", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    estimator.close()
