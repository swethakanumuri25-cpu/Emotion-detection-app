RUN pip install --upgrade pip

# 🔥 Install WITHOUT building av
RUN pip install \
    streamlit==1.33.0 \
    opencv-python-headless==4.8.1.78 \
    numpy==1.23.5 \
    tflite-runtime \
    streamlit-webrtc==0.47.1 \
    --no-deps

# Install required dependencies manually (excluding av)
RUN pip install \
    aiortc \
    aioice \
    pyee \
    pylibsrtp