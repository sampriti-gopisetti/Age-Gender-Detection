# Age and Gender Detection

This project uses deep learning to detect age and gender from images. It leverages a pre-trained model and OpenCV for image processing.

## Features
- Predicts age and gender from input images
- Uses a pre-trained Keras model (`Age_Gender_Model.h5`)
- Simple command-line interface

## Files
- `Detect.py`: Main script for running predictions
- `Age_Gender_Model.h5`: Pre-trained model file
- `labels.txt`: Contains class labels
- `output.txt`: Output results

## Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the detection script:
   ```bash
   python Detect.py
   ```

## Requirements
See `requirements.txt` for Python dependencies.

## License
MIT License
