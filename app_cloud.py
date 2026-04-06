import streamlit as st
import cv2
import numpy as np

st.title("😊 Emotion Detection (Upload Image)")

# Dummy prediction (replace later with real model)
def predict_emotion(face):
    return np.random.choice(['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise'])

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi = cv2.resize(gray[y:y+h, x:x+w], (48,48))
        roi = roi.astype('float32') / 255.0
        roi = np.expand_dims(roi, axis=(0, -1))

        label = predict_emotion(roi)

        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,255), 2)
        cv2.putText(img, label, (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    st.image(img, channels="BGR")