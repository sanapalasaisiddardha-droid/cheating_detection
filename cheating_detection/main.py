#!/usr/bin/env python3
"""
Cheating Detection System - Main Entry Point

Usage:
    python main.py --source video.mp4
    python main.py --source 0  # webcam
    python main.py --source video.mp4 --save output.mp4
    python main.py --source video.mp4 --performance fast
"""

import cv2
import time
import argparse
from pathlib import Path

from config import CONFIG, update_config_for_performance
from core.pipeline import CheatingDetectionPipeline
from utils.logger import AlertLogger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Exam Cheating Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --source exam_video.mp4
    python main.py --source 0 --width 1280 --height 720
    python main.py --source video.mp4 --save output.mp4 --log-dir logs
    python main.py --source video.mp4 --performance fast
        """
    )
    
    parser.add_argument(
        "--source", type=str, default="0",
        help="Video source: path to video file or camera index (default: 0 for webcam)"
    )
    
    parser.add_argument(
        "--width", type=int, default=1280,
        help="Frame width for processing (default: 1280)"
    )
    
    parser.add_argument(
        "--height", type=int, default=720,
        help="Frame height for processing (default: 720)"
    )
    
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: 'mps' for M2 Mac, 'cpu' for fallback (default: auto-detect)"
    )
    
    parser.add_argument(
        "--save", type=str, default=None,
        help="Path to save output video (default: None)"
    )
    
    parser.add_argument(
        "--log-dir", type=str, default="logs",
        help="Directory for log files (default: logs)"
    )
    
    parser.add_argument(
        "--performance", type=str, default="balanced",
        choices=["fast", "balanced", "accurate"],
        help="Performance preset (default: balanced)"
    )
    
    parser.add_argument(
        "--no-display", action="store_true",
        help="Disable video display (for headless processing)"
    )
    
    parser.add_argument(
        "--teacher-id", type=int, default=None,
        help="Manually set teacher track ID"
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Apply performance preset
    update_config_for_performance(args.performance)
    print(f"Performance preset: {args.performance}")
    
    # Update config with command line arguments
    CONFIG.model.input_width = args.width
    CONFIG.model.input_height = args.height
    
    if args.device:
        CONFIG.model.device = args.device
    
    # Initialize pipeline
    pipeline = CheatingDetectionPipeline(device=CONFIG.model.device)
    
    # Initialize logger
    video_name = Path(args.source).stem if not args.source.isdigit() else "webcam"
    logger = AlertLogger(
        output_dir=args.log_dir,
        video_name=video_name,
        enable_console=True
    )
    
    # Open video source
    if args.source.isdigit():
        source = int(args.source)
        print(f"Opening webcam {source}...")
    else:
        source = args.source
        print(f"Opening video: {source}")
    
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source: {args.source}")
        return 1
    
    # Set resolution for webcam
    if args.source.isdigit():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {frame_width}x{frame_height} @ {fps:.1f} FPS")
    if total_frames > 0:
        print(f"Total frames: {total_frames}")
    
    # Setup video writer
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            args.save, fourcc, fps, (frame_width, frame_height)
        )
        print(f"Saving output to: {args.save}")
    
    # Set teacher if specified
    if args.teacher_id is not None:
        pipeline.set_teacher(args.teacher_id)
    
    # Print controls
    print("\n" + "=" * 50)
    print("CONTROLS:")
    print("  q     - Quit")
    print("  t     - Mark first detected person as teacher")
    print("  a     - Auto-detect teacher")
    print("  r     - Reset all states")
    print("  s     - Print statistics")
    print("  SPACE - Pause/Resume")
    print("=" * 50 + "\n")
    
    # Processing loop
    frame_count = 0
    fps_counter = 0
    fps_start = time.time()
    fps_display = 0.0
    paused = False
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                
                if not ret:
                    print("\nEnd of video reached")
                    break
                
                frame_count += 1
                
                # Process frame
                processed_frame, alerts = pipeline.process_frame(frame)
                
                # Log alerts
                if alerts:
                    logger.log_batch(alerts, frame_count)
                
                # Calculate FPS
                fps_counter += 1
                elapsed = time.time() - fps_start
                if elapsed >= 1.0:
                    fps_display = fps_counter / elapsed
                    fps_counter = 0
                    fps_start = time.time()
                
                # Draw FPS
                cv2.putText(
                    processed_frame, f"FPS: {fps_display:.1f}",
                    (processed_frame.shape[1] - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )
                
                # Draw progress for video files
                if total_frames > 0:
                    progress = frame_count / total_frames
                    cv2.putText(
                        processed_frame,
                        f"Progress: {progress:.1%}",
                        (processed_frame.shape[1] - 180, frame_height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                    )
                
                # Save frame
                if writer:
                    writer.write(processed_frame)
                
                # Display
                if not args.no_display:
                    cv2.imshow("Cheating Detection", processed_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1 if not paused else 100) & 0xFF
            
            if key == ord('q'):
                print("\nQuitting...")
                break
            
            elif key == ord('t'):
                # Set first detected person as teacher
                states = list(pipeline.fusion.student_states.keys())
                if states:
                    pipeline.set_teacher(states[0])
                    print(f"Set teacher: ID {states[0]}")
            
            elif key == ord('a'):
                # Auto-detect teacher
                pipeline.auto_detect_teacher()
            
            elif key == ord('r'):
                # Reset
                pipeline.reset()
                print("States reset")
            
            elif key == ord('s'):
                # Print statistics
                stats = pipeline.get_statistics()
                print("\n--- Statistics ---")
                print(f"Frames processed: {stats['frames_processed']}")
                print(f"Average FPS: {stats['avg_fps']:.1f}")
                print(f"Students tracked: {stats['students_tracked']}")
                print(f"Total alerts: {stats['alerts']['total_alerts']}")
                print("------------------\n")
            
            elif key == ord(' '):
                # Pause/Resume
                paused = not paused
                print("PAUSED" if paused else "RESUMED")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        # Save logs and print summary
        logger.close()
        
        # Print final statistics
        stats = pipeline.get_statistics()
        print("\n" + "=" * 50)
        print("FINAL STATISTICS")
        print("=" * 50)
        print(f"Total frames processed: {stats['frames_processed']}")
        print(f"Processing time: {stats['elapsed_time']:.1f}s")
        print(f"Average FPS: {stats['avg_fps']:.1f}")
        print(f"Students tracked: {stats['students_tracked']}")
        print("=" * 50)
        
        # Release pipeline resources
        pipeline.close()
    
    return 0


if __name__ == "__main__":
    exit(main())
