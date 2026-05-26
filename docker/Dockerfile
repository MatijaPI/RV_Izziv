# Python slika (Trixie ne podpira mediapipe)
FROM python:3.11-slim

# Sistemske knjižnice za OpenCV + MediaPipe
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    libgles2 \
    libegl1 \
    && rm -rf /var/lib/apt/lists/*

# Namestitev knjižnic
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Delovni imenik
WORKDIR /workspace

# Privzeti način: strežnik
ENV RV_MODE=server

# src/ mora biti v PYTHONPATH da main.py najde config.py
ENV PYTHONPATH=/workspace/src