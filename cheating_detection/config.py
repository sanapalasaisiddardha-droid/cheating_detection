"""
Configuration and thresholds for cheating detection system.
Optimized for M2 MacBook Air with Python 3.11
"""

from dataclasses import dataclass, field
from typing import List
import torch


def get_device() -> str:
    """Get the best available device for M2 Mac."""
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class ModelConfig:
    """Model configuration."""
    
    # Device settings
    device: str = field(default_factory=get_device)
    
    # YOLOv8 settings
    yolo_model: str = "yolov8s-pose.pt"  # Options: yolov8n-pose, yolov8s-pose, yolov8m-pose
    yolo_conf: float = 0.5               # Detection confidence threshold
    yolo_iou: float = 0.45               # NMS IoU threshold
    
    # Input resolution (720p recommended for M2)
    input_width: int = 1280
    input_height: int = 720
    
    # Processing optimization
    process_every_n_frames: int = 1      # Set to 2 or 3 for better FPS
    
    # MediaPipe settings
    max_faces: int = 15
    max_hands: int = 30
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


@dataclass
class DetectionThresholds:
    """Thresholds for cheating behavior detection."""
    
    # ===== HEAD TURNING =====
    head_yaw_threshold: float = 35.0          # Degrees - trigger when head turns beyond this
    head_turn_duration: float = 2.0           # Seconds - must persist for this long
    head_turn_confidence_base: float = 0.7    # Base confidence for head turn alerts
    
    # ===== GAZE DIRECTION =====
    gaze_horizontal_threshold: float = 25.0   # Degrees - looking sideways threshold
    gaze_vertical_threshold: float = 20.0     # Degrees - looking up/down threshold
    gaze_duration: float = 2.5                # Seconds - must persist for this long
    gaze_confidence_base: float = 0.65        # Base confidence for gaze alerts
    
    # ===== HAND GESTURES =====
    hand_under_desk_duration: float = 3.0     # Seconds - hand below desk
    hand_near_face_duration: float = 2.0      # Seconds - hand near mouth/face
    hand_passing_distance: float = 100        # Pixels - distance for passing detection
    hand_confidence_base: float = 0.7         # Base confidence for hand alerts
    
    # ===== STUDENT INTERACTION (TALKING) =====
    interaction_distance: float = 150         # Pixels - head-to-head distance
    interaction_mutual_gaze_angle: float = 45 # Degrees - facing each other threshold
    interaction_duration: float = 5.0         # Seconds - sustained interaction
    interaction_confidence_base: float = 0.75 # Base confidence for interaction alerts
    
    # ===== TEACHER INTERACTION =====
    teacher_student_distance: float = 120     # Pixels - proximity threshold
    teacher_interaction_alert: float = 30.0   # Seconds - time before alerting
    teacher_confidence_base: float = 0.8      # Base confidence
    
    # ===== BODY POSTURE =====
    lean_angle_threshold: float = 20.0        # Degrees - body lean threshold
    lean_duration: float = 3.0                # Seconds - sustained lean
    reach_extension_ratio: float = 1.5        # Arm extension relative to shoulder width
    posture_confidence_base: float = 0.65     # Base confidence for posture alerts


@dataclass
class AlertConfig:
    """Alert system configuration."""
    
    # Cooldown between same alert type for same student
    alert_cooldown: float = 10.0              # Seconds
    
    # Minimum confidence to trigger alert
    min_confidence: float = 0.5
    
    # Alert types
    alert_types: List[str] = field(default_factory=lambda: [
        "HEAD_TURN",
        "LOOKING_SIDEWAYS",
        "LOOKING_AT_NEIGHBOR",
        "HAND_UNDER_DESK",
        "HAND_NEAR_FACE",
        "PASSING_OBJECT",
        "STUDENT_TALKING",
        "EXCESSIVE_LEANING",
        "REACHING_TOWARD_NEIGHBOR",
        "TEACHER_LONG_INTERACTION"
    ])
    
    # Severity levels
    severity_high: List[str] = field(default_factory=lambda: [
        "PASSING_OBJECT",
        "STUDENT_TALKING",
        "LOOKING_AT_NEIGHBOR"
    ])
    
    severity_medium: List[str] = field(default_factory=lambda: [
        "HEAD_TURN",
        "LOOKING_SIDEWAYS",
        "HAND_UNDER_DESK"
    ])


@dataclass
class VisualizationConfig:
    """Visualization settings."""
    
    # Colors (BGR format)
    color_normal: tuple = (0, 255, 0)         # Green
    color_alert: tuple = (0, 0, 255)          # Red
    color_warning: tuple = (0, 165, 255)      # Orange
    color_teacher: tuple = (255, 165, 0)      # Blue-ish
    color_text: tuple = (255, 255, 255)       # White
    
    # Drawing settings
    bbox_thickness: int = 2
    keypoint_radius: int = 3
    font_scale: float = 0.6
    font_thickness: int = 2
    
    # Info panel
    show_info_panel: bool = True
    panel_alpha: float = 0.7
    
    # Debug info
    show_head_pose: bool = True
    show_gaze: bool = True
    show_keypoints: bool = True
    show_fps: bool = True


@dataclass
class Config:
    """Main configuration container."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    thresholds: DetectionThresholds = field(default_factory=DetectionThresholds)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)


# Global configuration instance
CONFIG = Config()


def update_config_for_performance(level: str = "balanced"):
    """
    Update configuration for different performance levels.
    
    Args:
        level: "fast", "balanced", or "accurate"
    """
    global CONFIG
    
    if level == "fast":
        CONFIG.model.yolo_model = "yolov8n-pose.pt"
        CONFIG.model.process_every_n_frames = 2
        CONFIG.model.input_width = 960
        CONFIG.model.input_height = 540
        CONFIG.model.max_faces = 10
        CONFIG.model.max_hands = 20
        
    elif level == "accurate":
        CONFIG.model.yolo_model = "yolov8m-pose.pt"
        CONFIG.model.process_every_n_frames = 1
        CONFIG.model.input_width = 1920
        CONFIG.model.input_height = 1080
        CONFIG.model.max_faces = 20
        CONFIG.model.max_hands = 40
        
    else:  # balanced (default)
        CONFIG.model.yolo_model = "yolov8s-pose.pt"
        CONFIG.model.process_every_n_frames = 1
        CONFIG.model.input_width = 1280
        CONFIG.model.input_height = 720


if __name__ == "__main__":
    # Print current configuration
    print(f"Device: {CONFIG.model.device}")
    print(f"YOLO Model: {CONFIG.model.yolo_model}")
    print(f"Input Resolution: {CONFIG.model.input_width}x{CONFIG.model.input_height}")
