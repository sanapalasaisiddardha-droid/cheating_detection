"""
Rule engine for cheating behavior detection.
Applies thresholds and temporal logic to fused signals.
"""

import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from core.fusion_engine import FusionEngine, StudentState
from models.hand_detector import HandGesture
from models.action_classifier import Action
from config import CONFIG


class AlertType(Enum):
    """Types of cheating alerts."""
    HEAD_TURN = "HEAD_TURN"
    LOOKING_SIDEWAYS = "LOOKING_SIDEWAYS"
    LOOKING_AT_NEIGHBOR = "LOOKING_AT_NEIGHBOR"
    HAND_UNDER_DESK = "HAND_UNDER_DESK"
    HAND_NEAR_FACE = "HAND_NEAR_FACE"
    PASSING_OBJECT = "PASSING_OBJECT"
    STUDENT_TALKING = "STUDENT_TALKING"
    EXCESSIVE_LEANING = "EXCESSIVE_LEANING"
    REACHING_TOWARD_NEIGHBOR = "REACHING_TOWARD_NEIGHBOR"
    TEACHER_LONG_INTERACTION = "TEACHER_LONG_INTERACTION"
    SUSPICIOUS_MOVEMENT = "SUSPICIOUS_MOVEMENT"


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class Alert:
    """Detection alert."""
    
    alert_type: AlertType
    student_ids: List[int]
    confidence: float
    timestamp: float
    details: str = ""
    severity: AlertSeverity = AlertSeverity.MEDIUM
    
    def __str__(self):
        ids_str = ", ".join(map(str, self.student_ids))
        return (f"[{self.severity.value.upper()}] {self.alert_type.value}: "
                f"Student(s) {ids_str} | Confidence: {self.confidence:.0%} | {self.details}")
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "type": self.alert_type.value,
            "students": self.student_ids,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "details": self.details,
            "severity": self.severity.value
        }


class RuleEngine:
    """
    Rule-based cheating detection engine.
    Applies detection rules and manages alert cooldowns.
    """
    
    def __init__(self, fusion_engine: FusionEngine):
        """
        Initialize the rule engine.
        
        Args:
            fusion_engine: FusionEngine instance for accessing student states
        """
        self.fusion = fusion_engine
        self.thresholds = CONFIG.thresholds
        
        # Alert cooldowns to prevent spam
        self.last_alerts: Dict[Tuple[AlertType, int], float] = {}
        self.cooldown = CONFIG.alerts.alert_cooldown
        
        # Alert statistics
        self.total_alerts = 0
        self.alerts_by_type: Dict[AlertType, int] = {t: 0 for t in AlertType}
        
        print("[RuleEngine] Initialized")
    
    def _can_alert(self, alert_type: AlertType, student_id: int) -> bool:
        """Check if alert is allowed (not in cooldown)."""
        key = (alert_type, student_id)
        last_time = self.last_alerts.get(key, 0)
        return time.time() - last_time > self.cooldown
    
    def _record_alert(self, alert_type: AlertType, student_id: int):
        """Record alert timestamp for cooldown."""
        key = (alert_type, student_id)
        self.last_alerts[key] = time.time()
        self.total_alerts += 1
        self.alerts_by_type[alert_type] += 1
    
    def _get_severity(self, alert_type: AlertType) -> AlertSeverity:
        """Determine alert severity."""
        if alert_type.value in CONFIG.alerts.severity_high:
            return AlertSeverity.HIGH
        elif alert_type.value in CONFIG.alerts.severity_medium:
            return AlertSeverity.MEDIUM
        return AlertSeverity.LOW
    
    def check_head_turn(
        self,
        state: StudentState,
        current_time: float
    ) -> Optional[Alert]:
        """Check for suspicious head turning."""
        
        if not state.head_pose_valid:
            return None
        
        yaw_threshold = self.thresholds.head_yaw_threshold
        duration_threshold = self.thresholds.head_turn_duration
        
        if abs(state.head_yaw) > yaw_threshold:
            # Start or continue tracking
            if state.head_turn_start is None:
                state.head_turn_start = current_time
            
            elapsed = current_time - state.head_turn_start
            
            if elapsed > duration_threshold:
                if self._can_alert(AlertType.HEAD_TURN, state.track_id):
                    self._record_alert(AlertType.HEAD_TURN, state.track_id)
                    
                    # Calculate confidence based on angle magnitude
                    confidence = min(
                        0.95,
                        self.thresholds.head_turn_confidence_base +
                        (abs(state.head_yaw) - yaw_threshold) / 60
                    )
                    
                    direction = "right" if state.head_yaw > 0 else "left"
                    
                    return Alert(
                        alert_type=AlertType.HEAD_TURN,
                        student_ids=[state.track_id],
                        confidence=confidence,
                        timestamp=current_time,
                        details=f"Head turned {direction} ({state.head_yaw:.1f}°) for {elapsed:.1f}s",
                        severity=self._get_severity(AlertType.HEAD_TURN)
                    )
        else:
            # Reset tracking
            state.head_turn_start = None
        
        return None
    
    def check_gaze(
        self,
        state: StudentState,
        current_time: float
    ) -> Optional[Alert]:
        """Check for suspicious gaze direction."""
        
        gaze_threshold = self.thresholds.gaze_horizontal_threshold
        duration_threshold = self.thresholds.gaze_duration
        
        # Use combined look direction
        total_horizontal = state.total_look_horizontal
        
        if abs(total_horizontal) > gaze_threshold:
            if state.gaze_away_start is None:
                state.gaze_away_start = current_time
            
            elapsed = current_time - state.gaze_away_start
            
            if elapsed > duration_threshold:
                if self._can_alert(AlertType.LOOKING_SIDEWAYS, state.track_id):
                    self._record_alert(AlertType.LOOKING_SIDEWAYS, state.track_id)
                    
                    confidence = min(
                        0.9,
                        self.thresholds.gaze_confidence_base +
                        (abs(total_horizontal) - gaze_threshold) / 50
                    )
                    
                    direction = "right" if total_horizontal > 0 else "left"
                    
                    return Alert(
                        alert_type=AlertType.LOOKING_SIDEWAYS,
                        student_ids=[state.track_id],
                        confidence=confidence,
                        timestamp=current_time,
                        details=f"Looking {direction} ({total_horizontal:.1f}°) for {elapsed:.1f}s",
                        severity=self._get_severity(AlertType.LOOKING_SIDEWAYS)
                    )
        else:
            state.gaze_away_start = None
        
        return None
    
    def check_looking_at_neighbor(
        self,
        state: StudentState,
        current_time: float
    ) -> Optional[Alert]:
        """Check if student is looking at a neighbor's work."""
        
        neighbors = self.fusion.get_neighbors(
            state.track_id,
            max_distance=self.thresholds.interaction_distance * 1.5
        )
        
        for neighbor_id in neighbors:
            if self.fusion.is_looking_at_neighbor(state.track_id, neighbor_id):
                if self._can_alert(AlertType.LOOKING_AT_NEIGHBOR, state.track_id):
                    self._record_alert(AlertType.LOOKING_AT_NEIGHBOR, state.track_id)
                    
                    return Alert(
                        alert_type=AlertType.LOOKING_AT_NEIGHBOR,
                        student_ids=[state.track_id],
                        confidence=0.75,
                        timestamp=current_time,
                        details=f"Looking toward student {neighbor_id}",
                        severity=AlertSeverity.HIGH
                    )
        
        return None
    
    def check_hands(
        self,
        state: StudentState,
        current_time: float
    ) -> List[Alert]:
        """Check for suspicious hand positions."""
        
        alerts = []
        
        suspicious_gestures = [HandGesture.UNDER_DESK, HandGesture.NEAR_FACE]
        has_suspicious = any(g in suspicious_gestures for g in state.hand_gestures)
        
        if has_suspicious:
            if state.hand_suspicious_start is None:
                state.hand_suspicious_start = current_time
            
            elapsed = current_time - state.hand_suspicious_start
            
            # Check for hand under desk
            if HandGesture.UNDER_DESK in state.hand_gestures:
                if elapsed > self.thresholds.hand_under_desk_duration:
                    if self._can_alert(AlertType.HAND_UNDER_DESK, state.track_id):
                        self._record_alert(AlertType.HAND_UNDER_DESK, state.track_id)
                        
                        alerts.append(Alert(
                            alert_type=AlertType.HAND_UNDER_DESK,
                            student_ids=[state.track_id],
                            confidence=self.thresholds.hand_confidence_base,
                            timestamp=current_time,
                            details=f"Hand below desk for {elapsed:.1f}s",
                            severity=self._get_severity(AlertType.HAND_UNDER_DESK)
                        ))
            
            # Check for hand near face
            if HandGesture.NEAR_FACE in state.hand_gestures:
                if elapsed > self.thresholds.hand_near_face_duration:
                    if self._can_alert(AlertType.HAND_NEAR_FACE, state.track_id):
                        self._record_alert(AlertType.HAND_NEAR_FACE, state.track_id)
                        
                        alerts.append(Alert(
                            alert_type=AlertType.HAND_NEAR_FACE,
                            student_ids=[state.track_id],
                            confidence=0.6,
                            timestamp=current_time,
                            details=f"Hand near face for {elapsed:.1f}s",
                            severity=AlertSeverity.LOW
                        ))
        else:
            state.hand_suspicious_start = None
        
        return alerts
    
    def check_student_interaction(self, current_time: float) -> List[Alert]:
        """Check for suspicious interaction between students."""
        
        alerts = []
        distances = self.fusion.get_pairwise_distances()
        
        for (id_a, id_b), distance in distances.items():
            state_a = self.fusion.student_states.get(id_a)
            state_b = self.fusion.student_states.get(id_b)
            
            if state_a is None or state_b is None:
                continue
            
            # Skip if either is teacher
            if state_a.is_teacher or state_b.is_teacher:
                continue
            
            # Check proximity and mutual orientation
            if distance < self.thresholds.interaction_distance:
                if self.fusion.check_mutual_orientation(id_a, id_b):
                    # Track interaction duration
                    if state_a.interaction_start is None:
                        state_a.interaction_start = current_time
                        state_b.interaction_start = current_time
                    
                    elapsed = current_time - state_a.interaction_start
                    
                    if elapsed > self.thresholds.interaction_duration:
                        # Use smaller ID for consistent alerting
                        alert_key_id = min(id_a, id_b)
                        
                        if self._can_alert(AlertType.STUDENT_TALKING, alert_key_id):
                            self._record_alert(AlertType.STUDENT_TALKING, alert_key_id)
                            
                            alerts.append(Alert(
                                alert_type=AlertType.STUDENT_TALKING,
                                student_ids=[id_a, id_b],
                                confidence=self.thresholds.interaction_confidence_base,
                                timestamp=current_time,
                                details=f"Mutual interaction for {elapsed:.1f}s",
                                severity=AlertSeverity.HIGH
                            ))
                else:
                    # Not facing each other, reset
                    state_a.interaction_start = None
                    state_b.interaction_start = None
            else:
                # Too far apart, reset
                state_a.interaction_start = None
                state_b.interaction_start = None
        
        return alerts
    
    def check_body_actions(
        self,
        state: StudentState,
        current_time: float
    ) -> List[Alert]:
        """Check for suspicious body actions."""
        
        alerts = []
        
        leaning_actions = [
            Action.LEANING_LEFT, Action.LEANING_RIGHT,
            Action.LEANING_FORWARD
        ]
        
        has_leaning = any(a in leaning_actions for a in state.actions)
        
        if has_leaning:
            if state.leaning_start is None:
                state.leaning_start = current_time
            
            elapsed = current_time - state.leaning_start
            
            if elapsed > self.thresholds.lean_duration:
                if self._can_alert(AlertType.EXCESSIVE_LEANING, state.track_id):
                    self._record_alert(AlertType.EXCESSIVE_LEANING, state.track_id)
                    
                    # Determine direction
                    if Action.LEANING_LEFT in state.actions:
                        direction = "left"
                    elif Action.LEANING_RIGHT in state.actions:
                        direction = "right"
                    else:
                        direction = "forward"
                    
                    alerts.append(Alert(
                        alert_type=AlertType.EXCESSIVE_LEANING,
                        student_ids=[state.track_id],
                        confidence=self.thresholds.posture_confidence_base,
                        timestamp=current_time,
                        details=f"Leaning {direction} for {elapsed:.1f}s",
                        severity=AlertSeverity.MEDIUM
                    ))
        else:
            state.leaning_start = None
        
        # Check for reaching
        reaching_actions = [Action.REACHING_LEFT, Action.REACHING_RIGHT]
        if any(a in reaching_actions for a in state.actions):
            if self._can_alert(AlertType.REACHING_TOWARD_NEIGHBOR, state.track_id):
                self._record_alert(AlertType.REACHING_TOWARD_NEIGHBOR, state.track_id)
                
                direction = "left" if Action.REACHING_LEFT in state.actions else "right"
                
                alerts.append(Alert(
                    alert_type=AlertType.REACHING_TOWARD_NEIGHBOR,
                    student_ids=[state.track_id],
                    confidence=0.6,
                    timestamp=current_time,
                    details=f"Reaching {direction}",
                    severity=AlertSeverity.MEDIUM
                ))
        
        return alerts
    
    def check_teacher_interaction(self, current_time: float) -> List[Alert]:
        """Check for extended teacher-student interaction."""
        
        alerts = []
        
        if self.fusion.teacher_id is None:
            return alerts
        
        teacher_state = self.fusion.student_states.get(self.fusion.teacher_id)
        if teacher_state is None or teacher_state.center is None:
            return alerts
        
        teacher_center = teacher_state.center
        
        for track_id, state in self.fusion.student_states.items():
            if state.is_teacher or state.center is None:
                continue
            
            student_center = state.center
            
            distance = ((teacher_center[0] - student_center[0])**2 +
                       (teacher_center[1] - student_center[1])**2) ** 0.5
            
            if distance < self.thresholds.teacher_student_distance:
                if state.interaction_start is None:
                    state.interaction_start = current_time
                
                elapsed = current_time - state.interaction_start
                
                if elapsed > self.thresholds.teacher_interaction_alert:
                    if self._can_alert(AlertType.TEACHER_LONG_INTERACTION, track_id):
                        self._record_alert(AlertType.TEACHER_LONG_INTERACTION, track_id)
                        
                        alerts.append(Alert(
                            alert_type=AlertType.TEACHER_LONG_INTERACTION,
                            student_ids=[track_id],
                            confidence=self.thresholds.teacher_confidence_base,
                            timestamp=current_time,
                            details=f"Teacher near student for {elapsed:.1f}s",
                            severity=AlertSeverity.MEDIUM
                        ))
            else:
                state.interaction_start = None
        
        return alerts
    
    def process(self, current_time: float) -> List[Alert]:
        """
        Run all detection rules and return alerts.
        
        Args:
            current_time: Current timestamp
            
        Returns:
            List of Alert objects
        """
        all_alerts = []
        
        # Process each student
        for track_id, state in self.fusion.student_states.items():
            if state.is_teacher:
                continue
            
            # Individual checks
            alert = self.check_head_turn(state, current_time)
            if alert:
                all_alerts.append(alert)
            
            alert = self.check_gaze(state, current_time)
            if alert:
                all_alerts.append(alert)
            
            alert = self.check_looking_at_neighbor(state, current_time)
            if alert:
                all_alerts.append(alert)
            
            hand_alerts = self.check_hands(state, current_time)
            all_alerts.extend(hand_alerts)
            
            action_alerts = self.check_body_actions(state, current_time)
            all_alerts.extend(action_alerts)
        
        # Pairwise checks
        interaction_alerts = self.check_student_interaction(current_time)
        all_alerts.extend(interaction_alerts)
        
        # Teacher checks
        teacher_alerts = self.check_teacher_interaction(current_time)
        all_alerts.extend(teacher_alerts)
        
        return all_alerts
    
    def get_statistics(self) -> Dict:
        """Get alert statistics."""
        return {
            "total_alerts": self.total_alerts,
            "by_type": {t.value: c for t, c in self.alerts_by_type.items() if c > 0}
        }
    
    def reset_statistics(self):
        """Reset alert statistics."""
        self.total_alerts = 0
        self.alerts_by_type = {t: 0 for t in AlertType}
        self.last_alerts.clear()
