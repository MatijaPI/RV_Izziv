# Python slika (Trixie ne podpira mediapipe)
FROM python:3.11-slim

# Sistemske knjižnice za OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Namestitev knjižnic
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Delovni imenik (za lokalni zagon)
WORKDIR /workspace

# Privzeti način: strežnik (ko je /workspace in /data montiran)
# Prepiši z -e RV_MODE=local za lokalni zagon
ENV RV_MODE=server

# src/ mora biti v PYTHONPATH da main.py najde config.py
ENV PYTHONPATH=/workspace/src