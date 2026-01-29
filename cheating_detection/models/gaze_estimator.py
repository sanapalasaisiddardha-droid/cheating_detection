"""
Gaze estimation using MediaPipe Face Mesh iris landmarks.
Optimized for M2 MacBook Air - lightweight geometric approach.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Tuple, Optional
import math


class GazeEstimator:
    """
    Gaze estimation using MediaPipe Face Mesh with iris refinement.
    Uses geometric calculation from iris position relative to eye corners.
    """
    
    # Iris landmark indices (requires refine_landmarks=True)
    LEFT_IRIS = [468, 469, 470, 471, 472]
    RIGHT_IRIS = [473, 474, 475, 476, 477]
    LEFT_IRIS_CENTER = 468
    RIGHT_IRIS_CENTER = 473
    
    # Eye corner landmarks
    LEFT_EYE_LEFT = 33      # Outer corner
    LEFT_EYE_RIGHT = 133    # Inner corner
    RIGHT_EYE_LEFT = 362    # Inner corner
    RIGHT_EYE_RIGHT = 263   # Outer corner
    
    # Eye top/bottom for vertical gaze
    LEFT_EYE_TOP = 159
    LEFT_EYE_BOTTOM = 145
    RIGHT_EYE_TOP = 386
    RIGHT_EYE_BOTTOM = 374
    
    def __init__(self, max_faces: int = 15, min_detection_confidence: float = 0.5):
        """
        Initialize the gaze estimator.
        
        Args:
            max_faces: Maximum number of faces to track
            min_detection_confidence: Minimum confidence threshold
        """
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_faces,
            refine_landmarks=True,  # REQUIRED for iris landmarks
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5
        )
        
        print(f"[GazeEstimator] Initialized with iris refinement for max {max_faces} faces")
    
    def _calculate_eye_gaze(
        self,
        iris_center: Tuple[float, float],
        eye_left: Tuple[float, float],
        eye_right: Tuple[float, float],
        eye_top: Tuple[float, float],
        eye_bottom: Tuple[float, float]
    ) -> Tuple[float, float]:
        """
        Calculate gaze direction for one eye.
        
        Returns:
            (horizontal_gaze, vertical_gaze) in degrees
            Positive horizontal = looking right
            Positive vertical = looking down
        """
        # Horizontal gaze calculation
        eye_width = eye_right[0] - eye_left[0]
        if eye_width < 1:
            return 0.0, 0.0
        
        eye_center_x = (eye_left[0] + eye_right[0]) / 2
        horizontal_offset = (iris_center[0] - eye_center_x) / (eye_width / 2)
        
        # Map to approximate degrees (-1 to 1 -> -45 to 45 degrees)
        horizontal_gaze = horizontal_offset * 45
        
        # Vertical gaze calculation
        eye_height = eye_bottom[1] - eye_top[1]
        if eye_height < 1:
            return horizontal_gaze, 0.0
        
        eye_center_y = (eye_top[1] + eye_bottom[1]) / 2
        vertical_offset = (iris_center[1] - eye_center_y) / (eye_height / 2)
        
        # Map to approximate degrees
        vertical_gaze = vertical_offset * 30
        
        return horizontal_gaze, vertical_gaze
    
    def estimate_batch(self, frame: np.ndarray) -> Dict[Tuple[float, float], Tuple[float, float]]:
        """
        Estimate gaze for all faces in frame.
        
        Args:
            frame: Input BGR image
            
        Returns:
            Dictionary mapping face center (x, y) to (horizontal_gaze, vertical_gaze) in degrees
        """
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = self.face_mesh.process(rgb_frame)
        
        gaze_results = {}
        
        if not results.multi_face_landmarks:
            return gaze_results
        
        for face_landmarks in results.multi_face_landmarks:
            try:
                landmarks = face_landmarks.landmark
                
                # Left eye landmarks
                left_iris = (landmarks[self.LEFT_IRIS_CENTER].x * w,
                            landmarks[self.LEFT_IRIS_CENTER].y * h)
                left_eye_left = (landmarks[self.LEFT_EYE_LEFT].x * w,
                                landmarks[self.LEFT_EYE_LEFT].y * h)
                left_eye_right = (landmarks[self.LEFT_EYE_RIGHT].x * w,
                                 landmarks[self.LEFT_EYE_RIGHT].y * h)
                left_eye_top = (landmarks[self.LEFT_EYE_TOP].x * w,
                               landmarks[self.LEFT_EYE_TOP].y * h)
                left_eye_bottom = (landmarks[self.LEFT_EYE_BOTTOM].x * w,
                                  landmarks[self.LEFT_EYE_BOTTOM].y * h)
                
                # Calculate left eye gaze
                left_h_gaze, left_v_gaze = self._calculate_eye_gaze(
                    left_iris, left_eye_left, left_eye_right,
                    left_eye_top, left_eye_bottom
                )
                
                # Right eye landmarks
                right_iris = (landmarks[self.RIGHT_IRIS_CENTER].x * w,
                             landmarks[self.RIGHT_IRIS_CENTER].y * h)
                right_eye_left = (landmarks[self.RIGHT_EYE_LEFT].x * w,
                                 landmarks[self.RIGHT_EYE_LEFT].y * h)
                right_eye_right = (landmarks[self.RIGHT_EYE_RIGHT].x * w,
                                  landmarks[self.RIGHT_EYE_RIGHT].y * h)
                right_eye_top = (landmarks[self.RIGHT_EYE_TOP].x * w,
                                landmarks[self.RIGHT_EYE_TOP].y * h)
                right_eye_bottom = (landmarks[self.RIGHT_EYE_BOTTOM].x * w,
                                   landmarks[self.RIGHT_EYE_BOTTOM].y * h)
                
                # Calculate right eye gaze
                right_h_gaze, right_v_gaze = self._calculate_eye_gaze(
                    right_iris, right_eye_left, right_eye_right,
                    right_eye_top, right_eye_bottom
                )
                
                # Average both eyes for final gaze
                avg_horizontal = (left_h_gaze + right_h_gaze) / 2
                avg_vertical = (left_v_gaze + right_v_gaze) / 2
                
                # Get face center for matching
                nose = landmarks[1]
                face_center = (nose.x * w, nose.y * h)
                
                gaze_results[face_center] = (avg_horizontal, avg_vertical)
                
            except (IndexError, AttributeError) as e:
                continue
        
        return gaze_results
    
    def match_to_bbox(
        self,
        gaze_results: Dict[Tuple[float, float], Tuple[float, float]],
        face_bbox: np.ndarray,
        max_distance: Optional[float] = None
    ) -> Optional[Tuple[float, float]]:
        """
        Match gaze result to a face bounding box.
        
        Args:
            gaze_results: Dictionary from estimate_batch
            face_bbox: Face bounding box [x1, y1, x2, y2]
            max_distance: Maximum distance for matching
            
        Returns:
            (horizontal_gaze, vertical_gaze) or None
        """
        if not gaze_results:
            return None
        
        bbox_center_x = (face_bbox[0] + face_bbox[2]) / 2
        bbox_center_y = (face_bbox[1] + face_bbox[3]) / 2
        bbox_size = max(face_bbox[2] - face_bbox[0], face_bbox[3] - face_bbox[1])
        
        if max_distance is None:
            max_distance = bbox_size
        
        min_dist = float('inf')
        best_gaze = None
        
        for face_center, gaze in gaze_results.items():
            dist = math.sqrt(
                (face_center[0] - bbox_center_x)**2 +
                (face_center[1] - bbox_center_y)**2
            )
            
            if dist < min_dist and dist < max_distance:
                min_dist = dist
                best_gaze = gaze
        
        return best_gaze
    
    def get_combined_look_direction(
        self,
        head_yaw: float,
        head_pitch: float,
        gaze_horizontal: float,
        gaze_vertical: float
    ) -> Tuple[float, float]:
        """
        Combine head pose and eye gaze for total look direction.
        
        Args:
            head_yaw: Head yaw in degrees
            head_pitch: Head pitch in degrees
            gaze_horizontal: Eye gaze horizontal in degrees
            gaze_vertical: Eye gaze vertical in degrees
            
        Returns:
            (total_horizontal, total_vertical) in degrees
        """
        # Simple additive model - eyes can add to head direction
        total_horizontal = head_yaw + gaze_horizontal
        total_vertical = head_pitch + gaze_vertical
        
        return total_horizontal, total_vertical
    
    def is_looking_at_region(
        self,
        total_horizontal: float,
        total_vertical: float,
        target_angle_h: float,
        target_angle_v: float,
        tolerance: float = 15.0
    ) -> bool:
        """
        Check if gaze direction is toward a target region.
        
        Args:
            total_horizontal: Combined horizontal look direction
            total_vertical: Combined vertical look direction
            target_angle_h: Target horizontal angle
            target_angle_v: Target vertical angle
            tolerance: Angle tolerance in degrees
            
        Returns:
            True if looking at the target region
        """
        h_diff = abs(total_horizontal - target_angle_h)
        v_diff = abs(total_vertical - target_angle_v)
        
        return h_diff < tolerance and v_diff < tolerance
    
    def close(self):
        """Release resources."""
        self.face_mesh.close()


if __name__ == "__main__":
    # Test the gaze estimator
    import cv2
    
    estimator = GazeEstimator()
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gaze_results = estimator.estimate_batch(frame)
        
        for face_center, (h_gaze, v_gaze) in gaze_results.items():
            cx, cy = int(face_center[0]), int(face_center[1])
            
            # Draw gaze indicator
            end_x = int(cx + h_gaze * 2)
            end_y = int(cy + v_gaze * 2)
            
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            cv2.arrowedLine(frame, (cx, cy), (end_x, end_y), (255, 0, 0), 2)
            cv2.putText(frame, f"H:{h_gaze:.0f} V:{v_gaze:.0f}",
                       (cx - 40, cy - 20), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (0, 255, 0), 1)
        
        cv2.imshow("Gaze", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    estimator.close()
