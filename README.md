# Real-Time Emotion Detection using Webcam

## Overview

This project implements a real-time emotion detection system using a webcam. It captures live video, detects faces using OpenCV, and predicts emotions using a machine learning model or a placeholder function.

## Features

* Real-time webcam video processing
* Face detection using Haar Cascade
* Emotion classification (Angry, Happy, Sad, etc.)
* Lightweight and fast pipeline
* Built with Streamlit

## Tech Stack

* Python
* OpenCV
* NumPy
* Streamlit
* streamlit-webrtc

## Project Pipeline

1. Capture live video from webcam
2. Convert frame into NumPy array
3. Convert to grayscale
4. Detect faces using Haar Cascade
5. Extract face region (48x48)
6. Normalize and reshape input
7. Predict emotion
8. Display results on video

## Project Structure

```
emotion-app/
│
├── app_local.py
├── requirements.txt
├── haarcascade_frontalface_default.xml
├── emotion_model.tflite   (optional)
└── README.md
```

## How to Run Locally

### Clone repository

```
git clone https://github.com/swethakanumuri25-cpu/Emotion-detection-app.git
cd emotion-detection-app
```

### Install dependencies

```
pip install -r requirements.txt
```

### Run application

```
streamlit run app_local.py
```

### Open browser

```
http://localhost:8501
```

## Notes

* Webcam permission is required
* Works best in Chrome
* Close other applications using the camera

## Limitations

* Cloud platforms may not support webcam streaming properly
* Model accuracy depends on training quality

## Future Improvements

* Integrate trained deep learning model
* Add confidence scores
* Store predictions
* Improve UI

## Resume Description

Developed a real-time emotion detection system using OpenCV and Streamlit that detects facial expressions from live webcam input.

## Author

Swetha Kanumuri
Master’s in Data Science
University of North Texas

## Acknowledgments

OpenCV Haar Cascades
Streamlit Web
