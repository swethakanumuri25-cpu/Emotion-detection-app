FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip install \
    streamlit==1.33.0 \
    opencv-python-headless==4.8.1.78 \
    numpy==1.23.5 \
    tflite-runtime \
    streamlit-webrtc==0.47.1 \
    av==10.0.0

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]