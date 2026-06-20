FROM python:3.10-slim as builder

WORKDIR /app/data

RUN pip install --no-cache-dir gdown

RUN gdown 124t_JEyiavo_qYcFj6iSKVudMm268brG -O TCPFormer_ap3d_81.pth.tr
# RUN gdown 1_EjMWL9Rd9hPXaSahzShxm1-Ud2f4o5r -O train.pkl

FROM runpod/pytorch:1.0.3-cu1281-torch280-ubuntu2404

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirement and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY api.py compare_service.py report_service.py pose2d_service.py pose3d_service.py utils.py dto.py config.py ./

# Create output and cache directories
RUN mkdir -p /app/output /app/cache

# Copy large model weight last (minimize rebuilds of earlier layers)
COPY --from=builder /app/data/* ./

EXPOSE 8000

# Start server using Gunicorn and Uvicorn workers for production
CMD ["gunicorn", "api:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
