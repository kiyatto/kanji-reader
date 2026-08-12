#!/bin/bash
# Downloads the MediaPipe hand landmarker model used by script.py.
# Run this once after cloning the repo.

set -e

mkdir -p models

if [ -f "models/hand_landmarker.task" ]; then
    echo "hand_landmarker.task already present, skipping download."
else
    echo "Downloading hand_landmarker.task..."
    curl -L -o models/hand_landmarker.task \
        https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
    echo "Done."
fi