# Age and Gender Detection

Real-time age and gender detection from a webcam feed using a pre-trained deep learning model and OpenCV.

## Features
- Real-time face detection with dlib's HOG-based detector
- Predicts gender and coarse age range per face
- Dynamic, configurable paths for model and labels (no hardcoded paths)
- Command-line options for camera index and frame size

## Project Files
- `Detect.py` — Main script
- `Age_Gender_Model.h5` — Pre-trained model file (kept in repo)
- `labels.txt` — Class labels mapping
- `output.txt` — Sample output (optional)

## Installation
Create and activate a virtual environment (recommended), then install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Notes:
- On Windows, OpenCV works out of the box via `opencv-python`.
- `dlib` requires Visual Studio Build Tools and CMake for source builds. We recommend using a Python version with prebuilt dlib wheels available (3.8–3.11). If `pip install dlib` fails, see: https://learn.microsoft.com/windows/python/samples/dlib

## Usage
From the project directory:

```bash
python Detect.py [--model PATH] [--labels PATH] [--camera INDEX] [--width N] [--height N]
```

Examples:
- Default webcam, default model/labels (files in the same folder):
   ```bash
   python Detect.py
   ```
- Use an external USB camera (index 1) with 1280x720 frames:
   ```bash
   python Detect.py --camera 1 --width 1280 --height 720
   ```
- Load model/labels from custom locations:
   ```bash
   python Detect.py --model models\\Age_Gender_Model.h5 --labels data\\labels.txt
   ```

Press `Esc` to quit the preview window.

## Requirements
See `requirements.txt` for all Python dependencies.

## License
MIT License
