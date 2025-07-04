#!/bin/bash

# 激活虚拟环境
source ~/.venv/bin/activate

# 设置环境变量以跳过 Streamlit 初始化
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# 启动 Streamlit 应用
echo "🚀 启动 Enhanced AutoML Workflow Agent..."
echo "📍 应用将在 http://localhost:8502 运行"
echo "⚠️  如果是首次运行，请在侧边栏配置API供应商"

streamlit run app.py --server.port=8502