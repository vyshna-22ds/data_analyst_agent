# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Environment setup
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    HOST=0.0.0.0 \
    WORKERS=2

# Install system dependencies (optional: for numpy/pandas performance)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY . .

# Expose port
EXPOSE 7860

# Start Uvicorn with multiple workers
CMD ["bash", "-lc", "uvicorn app:app --host ${HOST} --port ${PORT} --workers ${WORKERS} --timeout-keep-alive 65"]

# Dockerfile snippet
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir networkx==2.8.8

