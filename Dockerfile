# ==============================================================================
# AstroFlow AI — Production & Edge Container Image
# Multi-stage build supporting CPU & CUDA acceleration
# ==============================================================================

FROM python:3.10-slim AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=3000

# Install system dependencies for OpenCV, MediaPipe, audio, and Bun
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    unzip \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    espeak \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Bun
RUN curl -fsSL https://bun.sh/install | bash
ENV PATH="/root/.bun/bin:${PATH}"

WORKDIR /app

# Copy dependency specifications first for Docker caching
COPY package.json bun.lock* ./
RUN bun install --frozen-lockfile || bun install

# Copy Python requirements & install
COPY ai_engine/ ./ai_engine/
RUN pip install --no-cache-dir \
    "numpy>=1.24.0,<2.0.0" \
    "mediapipe==0.10.14" \
    "opencv-python-headless>=4.8.0" \
    "ultralytics>=8.0.0" \
    "torch" \
    "torchvision" \
    "fastapi" \
    "uvicorn[standard]" \
    "websockets" \
    "pyttsx3" \
    "python-multipart" \
    "requests"

# Copy full application code
COPY . .

# Build Next.js console
RUN bun run build

EXPOSE 3000 8080

CMD ["bun", "run", "dev:all"]
