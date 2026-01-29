# Exam Cheating Detection System

A computer vision system for detecting potential cheating behaviors in exam settings. Optimized for M2 MacBook Air with Python 3.11.

## Features

- **Person Detection & Tracking**: YOLOv8-Pose with ByteTrack
- **Head Pose Estimation**: MediaPipe Face Mesh
- **Gaze Estimation**: Iris landmark tracking
- **Hand Detection**: MediaPipe Hands with gesture classification
- **Body Action Classification**: Pose-based action recognition

## Detected Behaviors

| Behavior | Detection Method |
|----------|-----------------|
| Head turning | Head yaw angle > 35° for > 2s |
| Looking sideways | Combined head + eye gaze direction |
| Looking at neighbor | Gaze direction toward specific student |
| Hand under desk | Hand position below person bbox |
| Hand near face | Hand in upper face region |
| Students talking | Mutual orientation + proximity |
| Excessive leaning | Body lean angle > 20° |
| Reaching toward neighbor | Arm extension beyond personal space |
| Teacher long interaction | Teacher near student > 30s |

## Installation

```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Requirements

- Python 3.11.14
- M2 MacBook Air (or any Mac with MPS support)
- Webcam or video files

## Usage

### Basic Usage

```bash
# Process webcam
python main.py --source 0

# Process video file
python main.py --source exam_video.mp4

# Save output video
python main.py --source exam_video.mp4 --save output.mp4
```

### Performance Presets

```bash
# Fast (lower accuracy, higher FPS)
python main.py --source video.mp4 --performance fast

# Balanced (default)
python main.py --source video.mp4 --performance balanced

# Accurate (higher accuracy, lower FPS)
python main.py --source video.mp4 --performance accurate
```

### Additional Options

```bash
# Custom resolution
python main.py --source 0 --width 1280 --height 720

# Headless processing (no display)
python main.py --source video.mp4 --no-display --save output.mp4

# Specify log directory
python main.py --source video.mp4 --log-dir my_logs

# Manually set teacher ID
python main.py --source video.mp4 --teacher-id 1
```

## Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `t` | Mark first detected person as teacher |
| `a` | Auto-detect teacher |
| `r` | Reset all states |
| `s` | Print statistics |
| `SPACE` | Pause/Resume |

## Project Structure

```
cheating_detection/
├── main.py                 # Entry point
├── config.py               # Configuration
├── requirements.txt        # Dependencies
├── README.md               # This file
│
├── models/                 # Detection models
│   ├── detector.py         # YOLOv8-Pose + ByteTrack
│   ├── head_pose.py        # Head pose estimation
│   ├── gaze_estimator.py   # Gaze direction
│   ├── hand_detector.py    # Hand detection + gestures
│   └── action_classifier.py # Body actions
│
├── core/                   # Processing core
│   ├── pipeline.py         # Main pipeline
│   ├── fusion_engine.py    # Signal fusion
│   └── rule_engine.py      # Detection rules
│
├── utils/                  # Utilities
│   ├── logger.py           # Alert logging
│   └── visualization.py    # Visualization tools
│
└── logs/                   # Output logs (created at runtime)
```

## Pipeline Flow

```
Video Frame
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  YOLOv8-Pose + ByteTrack                                      │
│  → Bounding boxes, 17 body keypoints, tracking IDs            │
└──────────────────────────────────────────────────────────────┘
    │
    ├──► Face crop ──► MediaPipe Face Mesh ──► Head Pose (yaw, pitch, roll)
    │                                      ──► Gaze (horizontal, vertical)
    │
    ├──► Hand region ──► MediaPipe Hands ──► Gesture classification
    │
    └──► Body keypoints ──► Action Classifier ──► Body actions
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  Fusion Engine                                                │
│  → Combines all signals per student                           │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  Rule Engine                                                  │
│  → Applies thresholds and temporal logic                      │
│  → Generates alerts                                           │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
  Alerts (type, student IDs, confidence, timestamp)
```

## Configuration

Edit `config.py` to adjust:

- Detection thresholds (angles, durations)
- Alert cooldowns
- Model selection
- Visualization options

### Key Thresholds

```python
# Head turning
head_yaw_threshold: 35.0       # degrees
head_turn_duration: 2.0        # seconds

# Gaze
gaze_horizontal_threshold: 25.0 # degrees
gaze_duration: 2.5             # seconds

# Interaction
interaction_distance: 150      # pixels
interaction_duration: 5.0      # seconds

# Teacher
teacher_interaction_alert: 30.0 # seconds
```

## Output

### Console Output

```
[ALERT] [MEDIUM] HEAD_TURN: Student(s) 2 | Confidence: 78% | Head turned left (-42.3°) for 2.5s
[ALERT] [HIGH] STUDENT_TALKING: Student(s) 3, 5 | Confidence: 75% | Mutual interaction for 6.2s
```

### Log Files

- `logs/video_alerts_TIMESTAMP.json` - Detailed JSON log
- `logs/video_alerts_TIMESTAMP.csv` - CSV for analysis

## Expected Performance (M2 MacBook Air)

| Preset | Resolution | Expected FPS |
|--------|-----------|--------------|
| Fast | 960x540 | 20-25 FPS |
| Balanced | 1280x720 | 15-20 FPS |
| Accurate | 1920x1080 | 10-12 FPS |

## Troubleshooting

### Low FPS
- Use `--performance fast` preset
- Reduce resolution with `--width 960 --height 540`
- Close other applications

### MPS Not Available
- Ensure macOS 12.3+
- Update PyTorch: `pip install --upgrade torch`

### MediaPipe Issues
- Reinstall: `pip install --force-reinstall mediapipe`

## License

MIT License

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [MediaPipe](https://mediapipe.dev/)
- [OpenCV](https://opencv.org/)
