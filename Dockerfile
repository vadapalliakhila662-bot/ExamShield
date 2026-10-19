# Dockerfile for ExamShield: OpenCV 5 & AWS Agentic Vision Pipeline
FROM python:3.13-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies required for OpenCV 5 headless operation and multimedia handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency specifications first for Docker layer caching
COPY requirements.txt .

# Install pinned Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Verify OpenCV 5 installation at build time
RUN python -c "import cv2; assert cv2.__version__.startswith('5'), f'Expected OpenCV 5, got {cv2.__version__}'; print('Docker Build Check: OpenCV 5 successfully verified: ' + cv2.__version__)"

# Copy application source code
COPY . .

# Expose API port
EXPOSE 8000

# Health check to ensure service responds
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Default launch command: runs FastAPI ASGI server
CMD ["uvicorn", "examshield.web.api:app", "--host", "0.0.0.0", "--port", "8000"]
