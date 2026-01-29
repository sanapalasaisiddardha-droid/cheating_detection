"""
Visualization utilities for cheating detection.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from config import CONFIG


class Visualizer:
    """
    Additional visualization utilities.
    Creates overlays, heatmaps, and summary views.
    """
    
    def __init__(self):
        """Initialize the visualizer."""
        self.vis_config = CONFIG.visualization
    
    def draw_gaze_ray(
        self,
        frame: np.ndarray,
        start_point: Tuple[int, int],
        yaw: float,
        pitch: float,
        length: int = 100,
        color: Tuple[int, int, int] = (255, 0, 0)
    ) -> np.ndarray:
        """
        Draw a gaze direction ray on the frame.
        
        Args:
            frame: Input image
            start_point: Starting point (x, y)
            yaw: Yaw angle in degrees
            pitch: Pitch angle in degrees
            length: Length of the ray
            color: Ray color
            
        Returns:
            Frame with gaze ray drawn
        """
        import math
        
        # Convert angles to direction
        yaw_rad = math.radians(yaw)
        pitch_rad = math.radians(pitch)
        
        dx = int(length * math.sin(yaw_rad))
        dy = int(length * math.sin(pitch_rad))
        
        end_point = (start_point[0] + dx, start_point[1] + dy)
        
        cv2.arrowedLine(frame, start_point, end_point, color, 2, tipLength=0.3)
        
        return frame
    
    def draw_interaction_line(
        self,
        frame: np.ndarray,
        point_a: Tuple[int, int],
        point_b: Tuple[int, int],
        is_suspicious: bool = False
    ) -> np.ndarray:
        """
        Draw a line between two interacting students.
        
        Args:
            frame: Input image
            point_a: First point
            point_b: Second point
            is_suspicious: Whether interaction is flagged
            
        Returns:
            Frame with line drawn
        """
        color = (0, 0, 255) if is_suspicious else (255, 255, 0)
        thickness = 3 if is_suspicious else 1
        
        cv2.line(frame, point_a, point_b, color, thickness)
        
        # Draw midpoint indicator
        mid = ((point_a[0] + point_b[0]) // 2, (point_a[1] + point_b[1]) // 2)
        cv2.circle(frame, mid, 5, color, -1)
        
        return frame
    
    def draw_zone_overlay(
        self,
        frame: np.ndarray,
        zones: List[np.ndarray],
        labels: List[str] = None,
        alpha: float = 0.3
    ) -> np.ndarray:
        """
        Draw semi-transparent zone overlays.
        
        Args:
            frame: Input image
            zones: List of zone polygons
            labels: Optional zone labels
            alpha: Transparency level
            
        Returns:
            Frame with zones drawn
        """
        overlay = frame.copy()
        
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255)
        ]
        
        for i, zone in enumerate(zones):
            color = colors[i % len(colors)]
            cv2.fillPoly(overlay, [zone.astype(np.int32)], color)
            
            if labels and i < len(labels):
                centroid = zone.mean(axis=0).astype(int)
                cv2.putText(
                    overlay, labels[i],
                    tuple(centroid),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
                )
        
        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    def create_dashboard(
        self,
        frame: np.ndarray,
        stats: dict,
        alerts_count: int
    ) -> np.ndarray:
        """
        Create a dashboard overlay with statistics.
        
        Args:
            frame: Input image
            stats: Statistics dictionary
            alerts_count: Current alert count
            
        Returns:
            Frame with dashboard
        """
        h, w = frame.shape[:2]
        
        # Create dashboard panel
        panel_width = 200
        dashboard = np.zeros((h, panel_width, 3), dtype=np.uint8)
        dashboard[:] = (40, 40, 40)  # Dark gray
        
        # Title
        cv2.putText(
            dashboard, "DASHBOARD", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        cv2.line(dashboard, (10, 40), (panel_width - 10, 40), (100, 100, 100), 1)
        
        # Statistics
        y = 70
        line_height = 25
        
        stats_items = [
            ("Students", stats.get('students_tracked', 0)),
            ("Frames", stats.get('frames_processed', 0)),
            ("FPS", f"{stats.get('avg_fps', 0):.1f}"),
            ("Alerts", alerts_count),
        ]
        
        for label, value in stats_items:
            cv2.putText(
                dashboard, f"{label}:", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1
            )
            cv2.putText(
                dashboard, str(value), (100, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
            y += line_height
        
        # Alert breakdown
        y += 20
        cv2.putText(
            dashboard, "ALERTS", (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
        )
        cv2.line(dashboard, (10, y + 5), (panel_width - 10, y + 5), (100, 100, 100), 1)
        y += 25
        
        alert_stats = stats.get('alerts', {}).get('by_type', {})
        for alert_type, count in list(alert_stats.items())[:5]:
            short_name = alert_type[:15]
            cv2.putText(
                dashboard, f"{short_name}: {count}",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1
            )
            y += 20
        
        # Combine with frame
        combined = np.hstack([frame, dashboard])
        
        return combined
    
    def create_alert_notification(
        self,
        frame: np.ndarray,
        message: str,
        severity: str = "medium",
        duration_frames: int = 30
    ) -> np.ndarray:
        """
        Create a prominent alert notification.
        
        Args:
            frame: Input image
            message: Alert message
            severity: "low", "medium", or "high"
            duration_frames: How long to show (for fading)
            
        Returns:
            Frame with notification
        """
        h, w = frame.shape[:2]
        
        # Color by severity
        colors = {
            "low": (255, 255, 0),
            "medium": (0, 165, 255),
            "high": (0, 0, 255)
        }
        color = colors.get(severity, colors["medium"])
        
        # Draw notification bar
        bar_height = 50
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_height), color, -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # Draw message
        cv2.putText(
            frame, f"ALERT: {message}",
            (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )
        
        return frame
    
    def draw_skeleton(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0),
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Draw pose skeleton on frame.
        
        Args:
            frame: Input image
            keypoints: [17, 3] keypoint array
            color: Skeleton color
            threshold: Confidence threshold
            
        Returns:
            Frame with skeleton
        """
        # COCO skeleton connections
        skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6),  # Shoulders
            (5, 7), (7, 9),  # Left arm
            (6, 8), (8, 10),  # Right arm
            (5, 11), (6, 12),  # Torso
            (11, 12),  # Hips
            (11, 13), (13, 15),  # Left leg
            (12, 14), (14, 16)  # Right leg
        ]
        
        # Draw connections
        for start_idx, end_idx in skeleton:
            if (keypoints[start_idx, 2] > threshold and 
                keypoints[end_idx, 2] > threshold):
                
                start = tuple(keypoints[start_idx, :2].astype(int))
                end = tuple(keypoints[end_idx, :2].astype(int))
                
                cv2.line(frame, start, end, color, 2)
        
        # Draw keypoints
        for i, kpt in enumerate(keypoints):
            if kpt[2] > threshold:
                center = tuple(kpt[:2].astype(int))
                cv2.circle(frame, center, 4, color, -1)
        
        return frame
