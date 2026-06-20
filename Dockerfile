FROM python:3.10-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirement and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY api.py compare_service.py report_service.py pose3d_service.py utils.py dto.py config.py ./

# Create output and cache directories
RUN mkdir -p /app/output /app/cache

EXPOSE 8000

# Start server using Gunicorn and Uvicorn workers for production
CMD ["gunicorn", "api:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
