# PT2RKNN 转换工具 Docker 镜像
# 基于 Python 3.8 Ubuntu 20.04
FROM python:3.8-slim

LABEL maintainer="pt2rknn"
LABEL description="RKNN Model Conversion Tool with Web UI"

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 复制 requirements.txt
COPY requirements.txt .

# 安装 PyTorch CPU 版本（升级到 2.0.1 以支持完整的量化模块）
RUN pip install --no-cache-dir \
    torch==2.0.1+cpu torchvision==0.15.2+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

# 安装其他 Python 依赖（除了 rknn-toolkit2）
RUN pip install --no-cache-dir \
    Flask==3.0.0 \
    Werkzeug==3.0.1 \
    numpy>=1.23.0 \
    netron \
    onnx>=1.14.0 \
    ultralytics

# ===== 重要：安装 rknn-toolkit2 =====
# 先升级 pip，再从本地复制 whl 文件并安装
# 使用 --no-deps 跳过自动依赖安装（不安装 TensorFlow）
RUN pip install --upgrade pip setuptools wheel
COPY rknn_toolkit2-*.whl /tmp/
RUN pip install --no-cache-dir --no-deps /tmp/rknn_toolkit2-*.whl && rm /tmp/rknn_toolkit2-*.whl

# 手动安装 rknn-toolkit2 的必要依赖（排除 TensorFlow）
RUN pip install --no-cache-dir \
    tqdm \
    protobuf \
    opencv-python \
    ruamel.yaml \
    scipy \
    psutil \
    onnx-simplifier \
    onnxoptimizer

# 复制项目文件
COPY . .

# 创建必要的目录
RUN mkdir -p uploads output calibration_data templates

# 设置环境变量
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# 暴露端口
EXPOSE 5600

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5600/ || exit 1

# 启动命令
CMD ["python", "app.py"]
