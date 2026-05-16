# Python slika (Trixie ne podpira mediapipe)
FROM python:3.11-slim

# Namestitev sistemskih knjižnic za OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Namestitev knjižnic
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt