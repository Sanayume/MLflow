# Enhanced AutoML Workflow Agent

🚀 **具备智能记忆、任务规划和增强推理能力的AI助手**

一个功能强大的AutoML工作流代理，支持多API供应商、智能对话、资源监控和高级机器学习工具。

## ✨ 主要特性

### 🧠 智能Agent系统
- **多层推理**: 3层推理架构（字面理解、上下文推理、深度洞察）
- **记忆系统**: 支持短期、工作、长期、情节和语义记忆
- **任务规划**: 自动任务分解和依赖管理
- **智能提示**: 基于模板的动态提示生成

### 🔧 多API供应商支持
- **Google Gemini**: 支持最新的Gemini模型
- **OpenAI**: 支持GPT-4、GPT-3.5等模型
- **Azure OpenAI**: 企业级Azure部署
- **Anthropic Claude**: Claude系列模型
- **自定义API**: 兼容OpenAI格式的本地部署

### 🛠️ 增强ML工具
- **自动数据分析**: 数据质量评估、分布分析、相关性检测
- **智能预处理**: 自动缺失值处理、异常值检测、特征编码
- **自动模型选择**: 基于数据特点推荐最佳算法
- **高级可视化**: 交互式图表，支持分布图、相关性热图、趋势图

### 📊 资源监控
- **实时监控**: CPU、内存、磁盘、网络、GPU使用率
- **智能告警**: 自定义阈值和告警级别
- **历史追踪**: 资源使用历史记录和分析
- **容器管理**: Docker容器资源限制和监控

### 🗃️ 完整历史系统
- **对话历史**: 智能对话记录和搜索
- **执行历史**: 代码执行结果和性能追踪
- **ML结果**: 模型训练结果和指标记录
- **系统事件**: 错误、警告和系统状态记录

## 🚀 快速开始

### 1. 环境准备
```bash
# 创建虚拟环境
python3 -m venv ~/.venv
source ~/.venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置API供应商
编辑 `config.yml` 文件或在Web界面中配置：

```yaml
# 设置环境变量
export GOOGLE_API_KEY="your-google-api-key"
export OPENAI_API_KEY="your-openai-api-key"
```

### 3. 启动应用
```bash
# 使用启动脚本
./start.sh

# 或直接启动
source ~/.venv/bin/activate
python -m streamlit run app.py --server.port=8502
```

### 4. 访问应用
打开浏览器访问: http://localhost:8502

## 📱 界面功能

### 侧边栏控制面板
- **供应商配置**: 添加、编辑、测试API供应商
- **会话控制**: 查看Agent状态和统计信息
- **历史搜索**: 智能搜索对话和执行历史
- **ML工具**: 启用/禁用各种ML增强功能
- **资源监控**: 实时系统资源状态

### 主界面
- **智能对话**: 支持多轮对话和上下文理解
- **任务分析**: 自动分析用户意图和任务类型
- **推理展示**: 显示多层推理过程和相关记忆
- **实时状态**: Agent状态、系统资源、GPU信息

## 🏗️ 架构设计

```
Enhanced AutoML Workflow Agent
├── 🧠 Enhanced Agent (enhanced_agent.py)
│   ├── 记忆系统 (EnhancedMemorySystem)
│   ├── 任务规划器 (EnhancedTaskPlanner)
│   └── 多层推理引擎
├── 🔧 配置管理 (enhanced_config.py)
│   ├── 多供应商支持
│   ├── 动态配置更新
│   └── 连接测试
├── 🛠️ ML工具 (enhanced_ml_tools.py)
│   ├── 自动数据分析
│   ├── 智能预处理
│   ├── 模型选择
│   └── 高级可视化
├── 📊 资源监控 (resource_monitor.py)
│   ├── 系统监控
│   ├── 容器管理
│   └── 告警系统
├── 🗃️ 历史管理 (enhanced_history.py)
│   ├── SQLite数据库
│   ├── 智能搜索
│   └── 数据导出
└── 🎨 智能提示 (intelligent_prompts.py)
    ├── 模板系统
    ├── 上下文增强
    └── 动态生成
```

## 🎯 使用场景

### 数据科学工作流
```python
# 自动数据分析
"分析这个数据集的质量和特征分布"

# 智能预处理
"对数据进行清洗和预处理，处理缺失值和异常值"

# 自动建模
"训练一个分类模型来预测客户流失，自动选择最佳算法"

# 可视化分析
"创建交互式可视化来展示模型性能和特征重要性"
```

### 系统监控
- 实时查看系统资源使用情况
- 设置自定义告警阈值
- 监控Docker容器资源限制
- 分析历史性能趋势

### 多团队协作
- 不同团队使用不同的API供应商
- 共享配置模板和最佳实践
- 统一的历史记录和知识管理

## 🛠️ 配置文件详解

### config.yml
```yaml
# 默认供应商
default_provider: "google"

# 供应商配置
providers:
  google:
    name: "Google Gemini"
    type: "google"
    enabled: true
    models: ["gemini-2.5-flash-preview-05-20"]
    default_model: "gemini-2.5-flash-preview-05-20"
    api_key: "${GOOGLE_API_KEY}"
  
  openai:
    name: "OpenAI"
    type: "openai"
    enabled: true
    base_url: "https://api.openai.com/v1"
    models: ["gpt-4o", "gpt-4-turbo"]
    default_model: "gpt-4o"
    api_key: "${OPENAI_API_KEY}"

# 功能开关
features:
  enhanced_agent: true
  memory_system: true
  task_planning: true
  resource_monitoring: true
  ml_tools: true

# 资源监控设置
resource_monitoring:
  enabled: true
  interval_seconds: 5
  cpu_threshold: 80
  memory_threshold: 80
  disk_threshold: 85
```

## 🔍 故障排除

### 常见问题

**Q: 启动时提示缺少依赖**
```bash
# 安装完整依赖
pip install streamlit langchain langchain-google-genai langchain-openai
pip install scikit-learn matplotlib seaborn plotly psutil docker
pip install mysql-connector-python pyyaml
```

**Q: API连接失败**
- 检查API密钥是否正确设置
- 使用Web界面的"测试连接"功能
- 查看控制台错误日志

**Q: 资源监控不工作**
- 确保Docker服务正在运行
- 检查系统权限
- 查看资源监控日志

**Q: 历史记录无法保存**
- 检查agent_workspace目录权限
- 确保SQLite数据库可写
- 查看数据库连接日志

## 📈 性能优化

### 推荐配置
- **内存**: 最少4GB RAM
- **CPU**: 多核处理器推荐
- **存储**: SSD推荐，至少1GB可用空间
- **网络**: 稳定的互联网连接（API调用）

### 优化建议
1. 启用资源监控以跟踪性能
2. 定期清理历史数据
3. 使用本地模型减少API调用
4. 配置合适的Docker资源限制

## 🤝 贡献指南

欢迎贡献！请遵循以下步骤：

1. Fork 项目
2. 创建功能分支: `git checkout -b feature/amazing-feature`
3. 提交更改: `git commit -m 'Add amazing feature'`
4. 推送分支: `git push origin feature/amazing-feature`
5. 创建Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [Streamlit](https://streamlit.io/) - 优秀的Web应用框架
- [LangChain](https://langchain.com/) - 强大的LLM应用开发框架
- [Plotly](https://plotly.com/) - 交互式可视化库
- [scikit-learn](https://scikit-learn.org/) - 机器学习工具包

---

**⭐ 如果这个项目对你有帮助，请给它一个星标！**

## 📊 项目统计

- **总代码行数**: ~4000+ 行
- **功能模块**: 8个主要模块
- **支持的API供应商**: 5个
- **ML工具功能**: 15+ 种
- **支持的可视化类型**: 10+ 种
- **历史记录类型**: 6种

---

*开发时间: 2025年7月*  
*版本: 1.0.0*  
*作者: Enhanced AutoML Team*