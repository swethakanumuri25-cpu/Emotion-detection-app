import streamlit as st
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# Load model
@st.cache_resource
def load_model():
    interpreter = Interpreter(model_path="emotion_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

st.title("😊 Live Emotion Detection")

# REAL-TIME PROCESSING CLASS
class EmotionDetector(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48,48))

            roi = roi_gray.astype('float32') / 255.0
            roi = np.expand_dims(roi, axis=-1)
            roi = np.expand_dims(roi, axis=0)

            interpreter.set_tensor(input_details[0]['index'], roi.astype('float32'))
            interpreter.invoke()

            prediction = interpreter.get_tensor(output_details[0]['index'])

            label = emotion_labels[np.argmax(prediction)]

            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,255), 2)
            cv2.putText(img, label, (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        return img

# START STREAM
webrtc_streamer(
    key="emotion",
    video_transformer_factory=EmotionDetector,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)