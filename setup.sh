#!/bin/bash

# Update package list and install system dependencies
apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    libgl1

# Install Python dependencies
pip install -r requirements.txt
