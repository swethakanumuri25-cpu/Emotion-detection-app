FROM python:3.10-slim

WORKDIR /app
COPY . .

# System dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    gcc \
    g++ \
    pkg-config \
    libopus-dev \
    libvpx-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Python dependencies
RUN pip install \
    streamlit==1.33.0 \
    opencv-python-headless==4.8.1.78 \
    numpy==1.23.5 \
    tflite-runtime \
    streamlit-webrtc==0.47.1 \
    aiortc

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]