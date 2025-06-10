# 使用一个官方的Ubuntu基础镜像
#FROM ubuntu:22.04
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
    software-properties-common \
    build-essential \
    curl \
    git \
    wget \
    ca-certificates \
    # python3-venv # python3.12-venv 会被一起安装
    && rm -rf /var/lib/apt/lists/*

# --- 安装 Python 3.12 ---
RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y python3.12 python3.12-dev python3.12-venv && \
    # 不在这里安装 python3-pip，我们将使用python3.12自带的pip模块
    rm -rf /var/lib/apt/lists/*

# --- 设置 Python 3.12 为默认的 python3 ---
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    # 我们将直接使用 python3.12 -m pip，所以不需要特别设置 pip3 的 alternatives
    # 如果确实需要一个全局的 pip3 指向 python3.12 的 pip，可以这样做：
    # 先确保 python3.12 -m pip install --upgrade pip 成功，然后获取其路径
    # RUN python3.12 -m pip install --upgrade pip
    # RUN ln -s /usr/local/bin/pip /usr/bin/pip3 # 假设pip安装到了/usr/local/bin/pip (需要确认)
    # 或者更简单，直接用 python3.12 -m pip
    # 为了避免PEP 668，我们先不强制创建全局pip3链接，而是显式调用
    # 我们需要一个pip，就用python3.12的
    # 让python3.12 -m ensurepip --upgrade 来确保pip是最新且与python3.12关联
    python3.12 -m ensurepip --upgrade

# 验证Python版本
RUN python3 --version
# 验证pip (通过python3.12 -m pip)
RUN python3.12 -m pip --version

# 设置工作目录
WORKDIR /app

# 安装常用的机器学习库
# 使用 python3.12 -m pip 来确保使用的是与python3.12关联的pip
# 并且添加 --break-system-packages 选项来处理PEP 668，因为我们是在构建Docker镜像，
# 这个环境是隔离的。
# 或者，更好的方式是创建一个虚拟环境并在其中安装，但这会改变后续执行命令的方式。
# 对于Docker镜像，通常我们会直接安装到Python路径下。
# 考虑使用一个标记文件来允许pip修改系统包，这比直接用--break-system-packages更“官方”一些
# RUN touch /usr/lib/python3.12/EXTERNALLY-MANAGED # 这行通常不起作用，是发行版控制的

# 方案A: 使用 --break-system-packages (更直接，在Docker构建中风险可控)
RUN python3.12 -m pip install --no-cache-dir --upgrade pip --break-system-packages && \
    python3.12 -m pip install --no-cache-dir --break-system-packages \
    numpy \
    pandas \
    scikit-learn \
    matplotlib \
    seaborn \
    jupyterlab 
    # torch torchvision torchaudio 
    # tensorflow 

# (可选) 如果你有本地的脚本或配置文件需要复制到镜像中
# COPY ./my_script.py /app/my_script.py

# 清理apt缓存以减小镜像大小
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*