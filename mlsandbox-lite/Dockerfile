# =========================================================================
# Stage 1: The Builder Stage
# 使用功能齐全但臃肿的 "devel" 镜像来安装和编译所有依赖
# 这个阶段的所有内容最终都会被丢弃
# =========================================================================
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS builder

# 设置环境变量，避免交互式提示
ENV DEBIAN_FRONTEND=noninteractive

# --- 1. 安装系统级依赖 ---
# 一次性安装所有需要的编译工具和地理信息处理库的依赖
# GDAL/GEOS/PROJ 是地理信息处理的核心
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    # Python 编译和环境管理
    software-properties-common \
    build-essential \
    # 地理信息库 (GDAL, GEOS, PROJ) 依赖
    gdal-bin \
    libgdal-dev \
    # 其他常用工具
    curl \
    git \
    wget \
    ca-certificates && \
    # 清理APT缓存
    rm -rf /var/lib/apt/lists/*

# --- 2. 安装 Python 3.12 ---
RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y --no-install-recommends python3.12 python3.12-dev python3.12-venv && \
    rm -rf /var/lib/apt/lists/*

# --- 3. 设置 Python 3.12 为默认，并创建虚拟环境 ---
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    # 创建一个干净的虚拟环境，所有Python包都将安装在这里
    python3 -m venv /opt/venv

# 激活虚拟环境，后续所有 pip 命令都在此环境下执行
# 这样可以避免任何系统Python的冲突 (如PEP 668)
ENV PATH="/opt/venv/bin:$PATH"

# --- 4. 安装所有 Python 库 ---
# 先升级 pip，然后在一个 RUN 命令中安装所有库，利用层缓存
# 使用 --no-cache-dir 来减小体积
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
    # 核心数据科学库
    numpy \
    pandas \
    scipy \
    scikit-learn \
    # 绘图库
    matplotlib \
    seaborn \
    plotly \
    # 地理信息处理 (这些库会使用到系统安装的GDAL)
    geopandas \
    rasterio \
    fiona \
    pyproj \
    shapely \
    # 深度学习 (根据需要选择一个或两个)
    # PyTorch 会自动找到CUDA版本
    torch torchvision torchaudio \
    # 如果也需要TensorFlow (会增加很多体积)
    # tensorflow \
    # 其他常用库
    jupyterlab \
    ipywidgets \
    opencv-python-headless \
    pillow \
    requests \
    beautifulsoup4 \
    lxml \
    openpyxl \
    statsmodels \
    xgboost \
    optuna \
    hyperopt \
    "ray[tune]" \
    scikit-optimize \
    lightgbm

# =========================================================================
# Stage 2: The Final Stage (mlsandbox:lite)
# 使用极度精简的 "runtime" 镜像作为最终的生产环境
# =========================================================================
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# 再次设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# --- 1. 安装运行时必要的系统库 ---
# 在 builder 阶段安装的 gdal-bin 等库，其运行时依赖也需要在这里安装
# 我们需要找出 geopandas/rasterio 等库的运行时共享库 (.so 文件)
# 通常是 libgdal.so 等
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    # 只安装 gdal 的运行时库，而不是开发包 libgdal-dev
    libgdal30 \
    # 为 python 3.12 运行时提供基础
    libpython3.12-minimal \
    && \
    rm -rf /var/lib/apt/lists/*

# --- 2. 拷贝产物 ---
# [关键步骤] 从 builder 阶段，只拷贝我们需要的、编译好的 Python 虚拟环境
COPY --from=builder /opt/venv /opt/venv

# --- 3. 设置环境 ---
# 将虚拟环境的 python 添加到 PATH，这样可以直接运行 `python`
# 确保这个路径在最前面
ENV PATH="/opt/venv/bin:$PATH"

# 设置工作目录
WORKDIR /app

# 设置默认启动命令，例如bash，以便调试
CMD ["/bin/bash"]