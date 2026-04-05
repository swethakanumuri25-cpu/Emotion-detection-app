import tensorflow as tf
import cv2
import numpy as np

# LOAD MODEL (use .keras file)
classifier = tf.keras.models.load_model("Emotion_little_vgg.keras")

# LOAD FACE CASCADE (Mac path)
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Camera not detected ❌")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,255), 2)

        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48,48))

        roi = roi_gray.astype('float32') / 255.0
        roi = np.expand_dims(roi, axis=-1)   # add channel
        roi = np.expand_dims(roi, axis=0)    # add batch

        prediction = classifier.predict(roi, verbose=0)
        label = emotion_labels[np.argmax(prediction)]

        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0,255,0), 2)

    cv2.imshow('Emotion Detector', frame)

    # press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()