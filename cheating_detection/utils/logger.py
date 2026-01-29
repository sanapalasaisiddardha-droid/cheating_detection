"""
Alert logging utility for cheating detection.
Saves alerts to file and provides analysis.
"""

import json
import csv
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from core.rule_engine import Alert


class AlertLogger:
    """
    Logger for cheating detection alerts.
    Supports JSON and CSV output formats.
    """
    
    def __init__(
        self,
        output_dir: str = "logs",
        video_name: str = None,
        enable_console: bool = True
    ):
        """
        Initialize the alert logger.
        
        Args:
            output_dir: Directory for log files
            video_name: Name of the video being processed
            enable_console: Whether to print alerts to console
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_prefix = f"{video_name}_" if video_name else ""
        
        self.json_path = self.output_dir / f"{video_prefix}alerts_{timestamp}.json"
        self.csv_path = self.output_dir / f"{video_prefix}alerts_{timestamp}.csv"
        
        self.enable_console = enable_console
        self.alerts: List[dict] = []
        
        # Initialize CSV file with headers
        self._init_csv()
        
        print(f"[AlertLogger] Logging to {self.output_dir}")
    
    def _init_csv(self):
        """Initialize CSV file with headers."""
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'frame_number', 'alert_type', 'severity',
                'student_ids', 'confidence', 'details'
            ])
    
    def log(self, alert: Alert, frame_number: int = None):
        """
        Log a single alert.
        
        Args:
            alert: Alert object to log
            frame_number: Optional frame number
        """
        alert_dict = alert.to_dict()
        alert_dict['frame_number'] = frame_number
        alert_dict['datetime'] = datetime.now().isoformat()
        
        self.alerts.append(alert_dict)
        
        # Append to CSV
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                alert_dict['datetime'],
                frame_number,
                alert.alert_type.value,
                alert.severity.value,
                str(alert.student_ids),
                f"{alert.confidence:.2f}",
                alert.details
            ])
        
        # Console output
        if self.enable_console:
            print(f"[ALERT] {alert}")
    
    def log_batch(self, alerts: List[Alert], frame_number: int = None):
        """Log multiple alerts."""
        for alert in alerts:
            self.log(alert, frame_number)
    
    def save_json(self):
        """Save all alerts to JSON file."""
        with open(self.json_path, 'w') as f:
            json.dump({
                'total_alerts': len(self.alerts),
                'alerts': self.alerts
            }, f, indent=2)
        
        print(f"[AlertLogger] Saved {len(self.alerts)} alerts to {self.json_path}")
    
    def get_summary(self) -> dict:
        """
        Get summary statistics of logged alerts.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.alerts:
            return {'total': 0}
        
        # Count by type
        by_type = {}
        for alert in self.alerts:
            alert_type = alert['type']
            by_type[alert_type] = by_type.get(alert_type, 0) + 1
        
        # Count by severity
        by_severity = {}
        for alert in self.alerts:
            severity = alert['severity']
            by_severity[severity] = by_severity.get(severity, 0) + 1
        
        # Count by student
        by_student = {}
        for alert in self.alerts:
            for student_id in alert['students']:
                by_student[student_id] = by_student.get(student_id, 0) + 1
        
        # Find most frequent offenders
        top_students = sorted(
            by_student.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return {
            'total': len(self.alerts),
            'by_type': by_type,
            'by_severity': by_severity,
            'by_student': by_student,
            'top_offenders': top_students
        }
    
    def print_summary(self):
        """Print a formatted summary to console."""
        summary = self.get_summary()
        
        print("\n" + "=" * 50)
        print("ALERT SUMMARY")
        print("=" * 50)
        
        print(f"\nTotal Alerts: {summary['total']}")
        
        if summary['total'] > 0:
            print("\nBy Type:")
            for alert_type, count in summary.get('by_type', {}).items():
                print(f"  {alert_type}: {count}")
            
            print("\nBy Severity:")
            for severity, count in summary.get('by_severity', {}).items():
                print(f"  {severity}: {count}")
            
            print("\nTop Offending Students:")
            for student_id, count in summary.get('top_offenders', []):
                print(f"  Student {student_id}: {count} alerts")
        
        print("=" * 50 + "\n")
    
    def close(self):
        """Finalize logging and save files."""
        self.save_json()
        self.print_summary()
