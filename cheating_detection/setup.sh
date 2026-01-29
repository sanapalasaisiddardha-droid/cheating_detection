#!/bin/bash

# Cheating Detection System - Setup Script
# For M2 MacBook Air with Python 3.11

echo "=========================================="
echo "Cheating Detection System - Setup"
echo "=========================================="

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: $PYTHON_VERSION"

if [[ "$PYTHON_VERSION" != "3.11" ]]; then
    echo "Warning: This project is optimized for Python 3.11"
    echo "Current version: $PYTHON_VERSION"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create virtual environment
echo ""
echo "[1/4] Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "[2/4] Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "[3/4] Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "[4/4] Installing requirements..."
pip install -r requirements.txt

# Verify installation
echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="

python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')

import ultralytics
print(f'Ultralytics version: {ultralytics.__version__}')

import mediapipe as mp
print(f'MediaPipe version: {mp.__version__}')

import cv2
print(f'OpenCV version: {cv2.__version__}')

print('')
print('All dependencies installed successfully!')
"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run the system:"
echo "  python main.py --source 0              # Webcam"
echo "  python main.py --source video.mp4     # Video file"
echo ""
echo "For help:"
echo "  python main.py --help"
echo ""
