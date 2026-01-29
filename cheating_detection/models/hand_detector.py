"""
Hand detection and gesture classification using MediaPipe Hands.
Optimized for M2 MacBook Air.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class HandGesture(Enum):
    """Hand gesture classifications."""
    NORMAL = "normal"
    UNDER_DESK = "under_desk"
    NEAR_FACE = "near_face"
    REACHING = "reaching"
    POINTING = "pointing"
    RAISED = "raised"
    HIDDEN = "hidden"


@dataclass
class HandDetection:
    """Single hand detection with landmarks."""
    
    landmarks: np.ndarray          # [21, 3] normalized coordinates
    pixel_coords: np.ndarray       # [21, 2] pixel coordinates
    handedness: str                # "Left" or "Right"
    confidence: float
    
    # MediaPipe hand landmark indices
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20
    
    @property
    def wrist(self) -> np.ndarray:
        """Get wrist position in pixels."""
        return self.pixel_coords[self.WRIST]
    
    @property
    def center(self) -> np.ndarray:
        """Get hand center (average of all landmarks)."""
        return self.pixel_coords.mean(axis=0)
    
    @property
    def fingertips(self) -> np.ndarray:
        """Get all fingertip positions."""
        tips = [self.THUMB_TIP, self.INDEX_TIP, self.MIDDLE_TIP, 
                self.RING_TIP, self.PINKY_TIP]
        return self.pixel_coords[tips]
    
    @property
    def palm_center(self) -> np.ndarray:
        """Get palm center (average of MCP joints)."""
        mcps = [self.INDEX_MCP, self.MIDDLE_MCP, self.RING_MCP, self.PINKY_MCP]
        return self.pixel_coords[mcps].mean(axis=0)
    
    def get_hand_size(self) -> float:
        """Estimate hand size based on landmark spread."""
        return np.linalg.norm(self.pixel_coords.max(axis=0) - self.pixel_coords.min(axis=0))


class HandDetector:
    """
    MediaPipe hand detection with gesture classification.
    Detects hand positions and classifies suspicious gestures.
    """
    
    def __init__(self, max_hands: int = 20, min_detection_confidence: float = 0.5):
        """
        Initialize the hand detector.
        
        Args:
            max_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence threshold
        """
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        
        print(f"[HandDetector] Initialized for max {max_hands} hands")
    
    def detect(self, frame: np.ndarray) -> List[HandDetection]:
        """
        Detect all hands in frame.
        
        Args:
            frame: Input BGR image
            
        Returns:
            List of HandDetection objects
        """
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = self.hands.process(rgb_frame)
        
        detections = []
        
        if not results.multi_hand_landmarks:
            return detections
        
        for hand_landmarks, handedness_info in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):
            # Extract normalized landmarks
            landmarks = np.array([
                [lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark
            ], dtype=np.float32)
            
            # Convert to pixel coordinates
            pixel_coords = landmarks[:, :2] * np.array([w, h])
            
            detections.append(HandDetection(
                landmarks=landmarks,
                pixel_coords=pixel_coords.astype(np.float32),
                handedness=handedness_info.classification[0].label,
                confidence=handedness_info.classification[0].score
            ))
        
        return detections
    
    def classify_gesture(
        self,
        hand: HandDetection,
        person_bbox: np.ndarray,
        frame_height: int
    ) -> HandGesture:
        """
        Classify hand gesture relative to person's body.
        
        Args:
            hand: HandDetection object
            person_bbox: Person bounding box [x1, y1, x2, y2]
            frame_height: Frame height for normalization
            
        Returns:
            HandGesture classification
        """
        x1, y1, x2, y2 = person_bbox
        person_height = y2 - y1
        person_width = x2 - x1
        
        wrist = hand.wrist
        wrist_x, wrist_y = wrist[0], wrist[1]
        palm_center = hand.palm_center
        
        # Check if hand is near face (upper 30% of person bbox)
        face_bottom = y1 + person_height * 0.30
        if wrist_y < face_bottom and x1 - 30 < wrist_x < x2 + 30:
            return HandGesture.NEAR_FACE
        
        # Check if hand is raised above head
        if wrist_y < y1 - 20:
            return HandGesture.RAISED
        
        # Check if hand is under desk (below person bbox)
        if wrist_y > y2 + 20:
            return HandGesture.UNDER_DESK
        
        # Check if hand is reaching outside person's space
        margin = person_width * 0.3
        if wrist_x < x1 - margin or wrist_x > x2 + margin:
            return HandGesture.REACHING
        
        # Check for pointing gesture
        if self._is_pointing(hand):
            return HandGesture.POINTING
        
        # Check if hand is hidden (z-depth indicates hand behind body)
        avg_z = hand.landmarks[:, 2].mean()
        if avg_z > 0.1:  # Positive Z = away from camera
            return HandGesture.HIDDEN
        
        return HandGesture.NORMAL
    
    def _is_pointing(self, hand: HandDetection) -> bool:
        """
        Check if hand is in a pointing gesture.
        Index finger extended, other fingers curled.
        """
        # Get finger tip and base positions
        index_tip = hand.pixel_coords[hand.INDEX_TIP]
        index_mcp = hand.pixel_coords[hand.INDEX_MCP]
        
        middle_tip = hand.pixel_coords[hand.MIDDLE_TIP]
        middle_mcp = hand.pixel_coords[hand.MIDDLE_MCP]
        
        ring_tip = hand.pixel_coords[hand.RING_TIP]
        ring_mcp = hand.pixel_coords[hand.RING_MCP]
        
        pinky_tip = hand.pixel_coords[hand.PINKY_TIP]
        pinky_mcp = hand.pixel_coords[hand.PINKY_MCP]
        
        # Calculate finger extensions
        index_ext = np.linalg.norm(index_tip - index_mcp)
        middle_ext = np.linalg.norm(middle_tip - middle_mcp)
        ring_ext = np.linalg.norm(ring_tip - ring_mcp)
        pinky_ext = np.linalg.norm(pinky_tip - pinky_mcp)
        
        # Pointing: index extended, others curled
        is_index_extended = index_ext > 50
        others_curled = middle_ext < index_ext * 0.7 and \
                       ring_ext < index_ext * 0.7 and \
                       pinky_ext < index_ext * 0.7
        
        return is_index_extended and others_curled
    
    def match_to_person(
        self,
        hands: List[HandDetection],
        person_bbox: np.ndarray,
        margin: float = 80
    ) -> List[HandDetection]:
        """
        Find hands belonging to a specific person.
        
        Args:
            hands: List of all detected hands
            person_bbox: Person bounding box [x1, y1, x2, y2]
            margin: Extra margin for matching
            
        Returns:
            List of hands belonging to this person
        """
        x1, y1, x2, y2 = person_bbox
        
        # Expand search region
        search_region = np.array([
            x1 - margin,
            y1 - margin,
            x2 + margin,
            y2 + margin * 2  # Extra margin below for under-desk
        ])
        
        matched = []
        
        for hand in hands:
            wrist = hand.wrist
            
            # Check if wrist is within search region
            if (search_region[0] < wrist[0] < search_region[2] and
                search_region[1] < wrist[1] < search_region[3]):
                matched.append(hand)
        
        return matched
    
    def detect_passing_motion(
        self,
        hands_person_a: List[HandDetection],
        hands_person_b: List[HandDetection],
        threshold: float = 100
    ) -> bool:
        """
        Detect if two people might be passing something.
        
        Args:
            hands_person_a: Hands of person A
            hands_person_b: Hands of person B
            threshold: Maximum distance to consider as passing
            
        Returns:
            True if passing motion detected
        """
        for hand_a in hands_person_a:
            for hand_b in hands_person_b:
                # Check distance between hand centers
                distance = np.linalg.norm(hand_a.center - hand_b.center)
                
                if distance < threshold:
                    return True
        
        return False
    
    def get_hand_trajectory(
        self,
        current_position: np.ndarray,
        history: List[np.ndarray],
        min_movement: float = 20
    ) -> Optional[str]:
        """
        Analyze hand movement trajectory.
        
        Args:
            current_position: Current hand position
            history: List of previous positions
            min_movement: Minimum movement to consider
            
        Returns:
            Movement direction or None
        """
        if len(history) < 3:
            return None
        
        # Calculate overall movement
        start = history[0]
        end = current_position
        
        movement = end - start
        magnitude = np.linalg.norm(movement)
        
        if magnitude < min_movement:
            return None
        
        # Determine direction
        dx, dy = movement[0], movement[1]
        
        if abs(dx) > abs(dy):
            return "right" if dx > 0 else "left"
        else:
            return "down" if dy > 0 else "up"
    
    def close(self):
        """Release resources."""
        self.hands.close()


if __name__ == "__main__":
    # Test the hand detector
    import cv2
    
    detector = HandDetector()
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        h, w = frame.shape[:2]
        hands = detector.detect(frame)
        
        # Mock person bbox for testing
        person_bbox = np.array([100, 50, w - 100, h - 50])
        
        for hand in hands:
            gesture = detector.classify_gesture(hand, person_bbox, h)
            
            # Draw hand landmarks
            wrist = hand.wrist.astype(int)
            cv2.circle(frame, tuple(wrist), 8, (0, 255, 0), -1)
            
            for tip in hand.fingertips:
                cv2.circle(frame, tuple(tip.astype(int)), 5, (255, 0, 0), -1)
            
            cv2.putText(frame, f"{hand.handedness}: {gesture.value}",
                       (wrist[0] - 50, wrist[1] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imshow("Hands", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    detector.close()
