# 1. Base Image: NVIDIA CUDA 12.1
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# 2. Setup System
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/usr/bin:${PATH}"

# 3. Install System Deps & Python 3.9 PPA
# FIX: Added 'cmake' to this list
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3.9-venv \
    python3.9-distutils \
    git \
    cmake \
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libosmesa6-dev \
    libglew-dev \
    ffmpeg \
    wget \
    curl \
    build-essential \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

# 4. Install pip for Python 3.9
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.9 get-pip.py && \
    rm get-pip.py

# 5. Alias python -> python3.9
RUN ln -sf /usr/bin/python3.9 /usr/bin/python && \
    ln -sf /usr/local/bin/pip3 /usr/bin/pip

# 6. Install Pip Requirements (WITH BUILD ISOLATION FIX)
WORKDIR /app
COPY requirements.txt /app/
# FIX: Downgrade setuptools and use --no-build-isolation for pyhash
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "setuptools<58.0.0" wheel && \
    pip install --no-cache-dir --no-build-isolation -r requirements.txt

# 7. Install Flash Attention (MAX_JOBS=2 prevents crash)
ENV MAX_JOBS=4
RUN pip install flash-attn==2.8.3 --no-build-isolation

# 8. Install YOUR Patched EGL Probe
COPY egl_probe /app/egl_probe
RUN cd /app/egl_probe && pip install .

# 9. Install LIBERO
COPY LIBERO /app/LIBERO
RUN cd /app/LIBERO && pip install -e .

# 10. Copy Source Code & Configs
COPY flower /app/flower
COPY conf /app/conf
# COPY preprocess /app/preprocess

# 11. Copy the Script
COPY run_robot.py /app/run_robot.py

# 12. Envs
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl

# Add this to your Dockerfile to permanently enable color prompt
RUN sed -i 's/#force_color_prompt=yes/force_color_prompt=yes/' /root/.bashrc
# Note: Adjust '/home/vscode/' to '/root/' depending on which user you log in as.

# CMD ["python", "run_robot.py"]