# Use a slim Python image — small footprint, fast cold-start on HF Spaces
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# HF Spaces expects the app to listen on 7860 by default
ENV PORT=7860

WORKDIR /app

# Install dependencies first (better Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app + model artifacts
COPY app.py .
COPY model_artifacts/ ./model_artifacts/
COPY frontend/ ./frontend/

EXPOSE 7860

# HF Spaces invokes this directly
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
