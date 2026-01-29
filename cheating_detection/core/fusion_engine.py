"""
Fusion engine to combine signals from all detection models.
Maintains state for each tracked student.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import math

from models.detector import Detection
from models.hand_detector import HandGesture, HandDetection
from models.action_classifier import Action


@dataclass
class StudentState:
    """Complete state for one tracked student."""
    
    track_id: int
    
    # Detection data
    bbox: Optional[np.ndarray] = None
    keypoints: Optional[np.ndarray] = None
    detection_confidence: float = 0.0
    
    # Head pose (degrees)
    head_yaw: float = 0.0
    head_pitch: float = 0.0
    head_roll: float = 0.0
    head_pose_valid: bool = False
    
    # Gaze direction (degrees)
    gaze_horizontal: float = 0.0
    gaze_vertical: float = 0.0
    gaze_valid: bool = False
    
    # Combined look direction
    total_look_horizontal: float = 0.0
    total_look_vertical: float = 0.0
    
    # Hand information
    hands: List[HandDetection] = field(default_factory=list)
    hand_gestures: List[HandGesture] = field(default_factory=list)
    
    # Body actions
    actions: List[Action] = field(default_factory=list)
    
    # Temporal tracking for alerts
    head_turn_start: Optional[float] = None
    gaze_away_start: Optional[float] = None
    hand_suspicious_start: Optional[float] = None
    leaning_start: Optional[float] = None
    interaction_start: Optional[float] = None
    
    # Role
    is_teacher: bool = False
    
    # Position tracking
    position_history: List[Tuple[float, float]] = field(default_factory=list)
    
    def reset_temporal(self, key: str):
        """Reset a specific temporal tracker."""
        setattr(self, key, None)
    
    def reset_all_temporal(self):
        """Reset all temporal trackers."""
        self.head_turn_start = None
        self.gaze_away_start = None
        self.hand_suspicious_start = None
        self.leaning_start = None
        self.interaction_start = None
    
    @property
    def center(self) -> Optional[Tuple[float, float]]:
        """Get person center from bbox."""
        if self.bbox is None:
            return None
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2
        )
    
    def update_position_history(self, max_length: int = 30):
        """Update position history."""
        center = self.center
        if center:
            self.position_history.append(center)
            if len(self.position_history) > max_length:
                self.position_history.pop(0)


class FusionEngine:
    """
    Combines all detection signals into unified student states.
    Handles matching between different model outputs.
    """
    
    def __init__(self):
        """Initialize the fusion engine."""
        self.student_states: Dict[int, StudentState] = {}
        self.teacher_id: Optional[int] = None
        
        print("[FusionEngine] Initialized")
    
    def get_or_create_state(self, track_id: int) -> StudentState:
        """Get existing state or create new one."""
        if track_id not in self.student_states:
            self.student_states[track_id] = StudentState(track_id=track_id)
        return self.student_states[track_id]
    
    def update_detection(self, detection: Detection):
        """Update state with YOLO detection."""
        state = self.get_or_create_state(detection.track_id)
        state.bbox = detection.bbox
        state.keypoints = detection.keypoints
        state.detection_confidence = detection.confidence
        state.update_position_history()
    
    def update_head_pose(
        self,
        track_id: int,
        yaw: float,
        pitch: float,
        roll: float
    ):
        """Update state with head pose."""
        state = self.get_or_create_state(track_id)
        state.head_yaw = yaw
        state.head_pitch = pitch
        state.head_roll = roll
        state.head_pose_valid = True
        
        # Update combined look direction
        state.total_look_horizontal = yaw + state.gaze_horizontal
        state.total_look_vertical = pitch + state.gaze_vertical
    
    def update_gaze(
        self,
        track_id: int,
        horizontal: float,
        vertical: float
    ):
        """Update state with gaze direction."""
        state = self.get_or_create_state(track_id)
        state.gaze_horizontal = horizontal
        state.gaze_vertical = vertical
        state.gaze_valid = True
        
        # Update combined look direction
        state.total_look_horizontal = state.head_yaw + horizontal
        state.total_look_vertical = state.head_pitch + vertical
    
    def update_hands(
        self,
        track_id: int,
        hands: List[HandDetection],
        gestures: List[HandGesture]
    ):
        """Update state with hand information."""
        state = self.get_or_create_state(track_id)
        state.hands = hands
        state.hand_gestures = gestures
    
    def update_actions(self, track_id: int, actions: List[Action]):
        """Update state with classified actions."""
        state = self.get_or_create_state(track_id)
        state.actions = actions
    
    def set_teacher(self, track_id: int):
        """Mark a person as the teacher."""
        self.teacher_id = track_id
        state = self.get_or_create_state(track_id)
        state.is_teacher = True
        print(f"[FusionEngine] Teacher set to track ID: {track_id}")
    
    def unset_teacher(self):
        """Remove teacher designation."""
        if self.teacher_id is not None:
            state = self.student_states.get(self.teacher_id)
            if state:
                state.is_teacher = False
        self.teacher_id = None
    
    def identify_teacher_heuristic(self) -> Optional[int]:
        """
        Try to identify teacher based on behavioral heuristics.
        Teachers typically: stand, move around more, have different position.
        """
        if len(self.student_states) < 2:
            return None
        
        for track_id, state in self.student_states.items():
            # Check if standing
            if Action.STANDING in state.actions:
                return track_id
            
            # Check for high movement (walking around)
            if len(state.position_history) >= 10:
                positions = np.array(state.position_history)
                total_movement = np.sum(np.linalg.norm(
                    np.diff(positions, axis=0), axis=1
                ))
                
                # High movement suggests teacher
                if total_movement > 500:  # Threshold depends on resolution
                    return track_id
        
        return None
    
    def get_pairwise_distances(self) -> Dict[Tuple[int, int], float]:
        """Calculate distances between all student pairs."""
        distances = {}
        track_ids = [
            tid for tid, state in self.student_states.items()
            if not state.is_teacher and state.bbox is not None
        ]
        
        for i, id_a in enumerate(track_ids):
            for id_b in track_ids[i+1:]:
                state_a = self.student_states[id_a]
                state_b = self.student_states[id_b]
                
                center_a = state_a.center
                center_b = state_b.center
                
                if center_a and center_b:
                    distance = math.sqrt(
                        (center_a[0] - center_b[0])**2 +
                        (center_a[1] - center_b[1])**2
                    )
                    distances[(id_a, id_b)] = distance
        
        return distances
    
    def get_neighbors(
        self,
        track_id: int,
        max_distance: float = 200
    ) -> List[int]:
        """Get nearby students."""
        state = self.student_states.get(track_id)
        if state is None or state.center is None:
            return []
        
        neighbors = []
        center = state.center
        
        for other_id, other_state in self.student_states.items():
            if other_id == track_id or other_state.is_teacher:
                continue
            
            other_center = other_state.center
            if other_center is None:
                continue
            
            distance = math.sqrt(
                (center[0] - other_center[0])**2 +
                (center[1] - other_center[1])**2
            )
            
            if distance < max_distance:
                neighbors.append(other_id)
        
        return neighbors
    
    def check_mutual_orientation(
        self,
        id_a: int,
        id_b: int,
        angle_threshold: float = 45
    ) -> bool:
        """Check if two students are oriented toward each other."""
        state_a = self.student_states.get(id_a)
        state_b = self.student_states.get(id_b)
        
        if state_a is None or state_b is None:
            return False
        
        if not state_a.head_pose_valid or not state_b.head_pose_valid:
            return False
        
        center_a = state_a.center
        center_b = state_b.center
        
        if center_a is None or center_b is None:
            return False
        
        # Calculate angle from A to B
        dx = center_b[0] - center_a[0]
        dy = center_b[1] - center_a[1]
        angle_a_to_b = math.degrees(math.atan2(dx, -dy))
        
        # Check if A is facing toward B
        yaw_diff_a = abs(state_a.head_yaw - angle_a_to_b)
        if yaw_diff_a > 180:
            yaw_diff_a = 360 - yaw_diff_a
        a_facing_b = yaw_diff_a < angle_threshold
        
        # Check if B is facing toward A
        angle_b_to_a = angle_a_to_b + 180
        if angle_b_to_a > 180:
            angle_b_to_a -= 360
        
        yaw_diff_b = abs(state_b.head_yaw - angle_b_to_a)
        if yaw_diff_b > 180:
            yaw_diff_b = 360 - yaw_diff_b
        b_facing_a = yaw_diff_b < angle_threshold
        
        return a_facing_b and b_facing_a
    
    def is_looking_at_neighbor(
        self,
        track_id: int,
        neighbor_id: int,
        angle_threshold: float = 30
    ) -> bool:
        """Check if student is looking toward a specific neighbor."""
        state = self.student_states.get(track_id)
        neighbor = self.student_states.get(neighbor_id)
        
        if state is None or neighbor is None:
            return False
        
        center = state.center
        neighbor_center = neighbor.center
        
        if center is None or neighbor_center is None:
            return False
        
        # Calculate expected look direction
        dx = neighbor_center[0] - center[0]
        dy = neighbor_center[1] - center[1]
        expected_yaw = math.degrees(math.atan2(dx, -dy))
        
        # Compare with actual look direction
        actual_yaw = state.total_look_horizontal
        
        yaw_diff = abs(actual_yaw - expected_yaw)
        if yaw_diff > 180:
            yaw_diff = 360 - yaw_diff
        
        return yaw_diff < angle_threshold
    
    def cleanup_old_tracks(self, active_track_ids: List[int]):
        """Remove states for tracks no longer detected."""
        to_remove = [
            track_id for track_id in self.student_states
            if track_id not in active_track_ids
        ]
        
        for track_id in to_remove:
            if track_id == self.teacher_id:
                self.teacher_id = None
            del self.student_states[track_id]
    
    def get_all_students(self) -> List[StudentState]:
        """Get all non-teacher students."""
        return [
            state for state in self.student_states.values()
            if not state.is_teacher
        ]
    
    def get_student_count(self) -> int:
        """Get count of students (excluding teacher)."""
        return sum(
            1 for state in self.student_states.values()
            if not state.is_teacher
        )
    
    def reset(self):
        """Reset all states."""
        self.student_states.clear()
        self.teacher_id = None
        print("[FusionEngine] Reset complete")
