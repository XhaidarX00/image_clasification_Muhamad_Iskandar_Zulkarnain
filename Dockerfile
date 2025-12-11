# Dockerfile for Cat vs Dog Image Classification
# Uses Python 3.10 with TensorFlow

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV DEBIAN_FRONTEND=noninteractive

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies (no system deps needed for this app)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY templates/ ./templates/
COPY static/ ./static/
COPY model/ ./model/
COPY data/ ./data/

# Create necessary directories
RUN mkdir -p /app/static/uploads /app/data

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
