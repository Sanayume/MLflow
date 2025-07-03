#!/bin/bash

# 启动脚本 - Enhanced AutoML Workflow Agent
# Enhanced startup script for AutoML Workflow Agent

echo "🚀 正在启动 Enhanced AutoML Workflow Agent..."

# 检查虚拟环境
if [ ! -d "$HOME/.venv" ]; then
    echo "❌ 未找到虚拟环境 ~/.venv"
    echo "请先创建虚拟环境: python3 -m venv ~/.venv"
    exit 1
fi

# 激活虚拟环境
echo "📦 激活虚拟环境..."
source ~/.venv/bin/activate

# 检查必要的环境变量
if [ -z "$GOOGLE_API_KEY" ]; then
    echo "⚠️  警告: GOOGLE_API_KEY 环境变量未设置"
    echo "请设置 GOOGLE_API_KEY 或在配置文件中配置其他供应商"
fi

# 创建必要的目录
echo "📁 创建工作目录..."
mkdir -p agent_workspace

# 启动应用
echo "🌟 启动 Streamlit 应用..."
echo "📱 应用将在 http://localhost:8502 启动"
echo "🔧 你可以在侧边栏的'供应商配置'选项卡中配置API供应商"
echo ""

# 设置端口（如果未指定）
PORT=${PORT:-8502}

# 启动 Streamlit
python -m streamlit run app.py \
    --server.port=$PORT \
    --server.headless=true \
    --server.runOnSave=true \
    --theme.base=dark