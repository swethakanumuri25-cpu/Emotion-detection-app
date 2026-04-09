# Real-Time Emotion Detection using CNN

## Overview
This project implements a real-time emotion detection system using a webcam. It captures live video, detects faces using OpenCV, and predicts human emotions such as Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.

This project was originally developed as part of my Bachelor’s (B.Tech) coursework and later enhanced into a real-time application using Streamlit.

## Features
- Real-time webcam video processing  
- Face detection using Haar Cascade  
- Emotion classification (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise)  
- Lightweight and fast pipeline  
- Interactive UI using Streamlit  

## Tech Stack
- Python  
- OpenCV  
- NumPy  
- Streamlit  
- streamlit-webrtc  
- TensorFlow Lite (optional)  

## Project Pipeline
1. Capture live video from webcam  
2. Convert frame into NumPy array  
3. Convert image to grayscale  
4. Detect faces using Haar Cascade  
5. Extract face region (48x48)  
6. Normalize and reshape input  
7. Predict emotion  
8. Display results on video stream  

## Project Structure
Emotion-detection-app/

app.py  
requirements.txt  
haarcascade_frontalface_default.xml  
emotion_model.tflite (optional)  
assets/app_demo_video.mp4  
README.md  

## How to Run Locally

Clone repository:
git clone https://github.com/swethakanumuri25-cpu/Emotion-detection-app.git  
cd Emotion-detection-app  

Install dependencies:
pip install -r requirements.txt  

Run application:
streamlit run app.py  

Open browser:
http://localhost:8501  

## Demo
Demo video is available in:
assets/app_demo_video.mp4  

## Notes
- Webcam permission is required  
- Works best in Google Chrome  
- Close other applications using the camera  

## Limitations
- Cloud platforms may not support real-time webcam streaming  
- Model accuracy depends on training quality  

## Future Improvements
- Improve CNN model accuracy  
- Add confidence scores  
- Store predictions  
- Enhance UI  

## Resume Description
Developed a real-time emotion detection system using OpenCV and Streamlit that detects facial expressions from live webcam input.

## Author
Swetha Kanumuri  
Master’s in Data Science  
University of North Texas  

## Acknowledgments
- OpenCV  
- Streamlit  
- TensorFlow Lite  
