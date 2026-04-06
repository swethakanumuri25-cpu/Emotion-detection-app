import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model("Emotion_little_vgg.h5", compile=False)

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save file
with open("emotion_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Converted to TFLite!")