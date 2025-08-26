# Base image with CUDA 12.8 and Ubuntu 22.04
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04
 
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV PATH=/usr/local/cuda/bin:$PATH
 
# Switch to faster HTTPS mirrors
RUN sed -i 's|http://archive.ubuntu.com|https://archive.ubuntu.com|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com|https://security.ubuntu.com|g' /etc/apt/sources.list
 
# Install system dependencies and Python 3.11
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3-pip \
    build-essential libgl1-mesa-glx libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*
 
# Set default Python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1
 
# Set working directory
WORKDIR /app/backend
 
# Copy backend code
COPY backend /app/backend
 
# Copy requirements file
COPY requirements.txt /app/backend/
 
# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 && \
    pip install --no-cache-dir uvicorn opencv-python-headless==4.5.5.64
    

 
# Expose FastAPI port
EXPOSE 8080
 
# Run FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "7"]
 
 