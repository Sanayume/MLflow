# app.py

import streamlit as st
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain_core.tools import StructuredTool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
import os
import datetime
import json
import logging

# 导入增强配置管理器
from enhanced_config import config_manager, render_provider_config_ui, create_llm_from_config

# 导入增强模块
from enhanced_agent import EnhancedAIAgent, MemoryType, TaskType
from enhanced_history import (
    EnhancedHistoryManager, HistoryHelpers, HistoryType, 
    EventSeverity, SearchFilters
)
from enhanced_ml_tools import EnhancedMLTools, MLTaskType, VisualizationType
from resource_monitor import SystemResourceMonitor, ResourceAlert, ResourceType
from local_tools import (
    safe_python_executor, 
    read_local_file, 
    write_local_file, 
    # list_directory_contents, # 可以替换掉旧的
    list_directory_items_with_paths, # 新的工具
    make_web_request 
)
from sandbox_main import (
    execute_ml_code_in_docker,
    CONTAINER_CODE_EXECUTION_BASE_TARGET,
    CONTAINER_OUTPUTS_MOUNT_TARGET,
    CONTAINER_DATASETS_MOUNT_TARGET,
    HOST_EXECUTION_LOGS_ROOT,
    HOST_OUTPUTS_MOUNT_SOURCE,
    HOST_DATASETS_MOUNT_SOURCE,
    HOST_ROOT_WORKSPACE,
    DOCKER_IMAGE_NAME,
    SandboxExecutionInput
)
from langchain_community.tools.file_management import (
    CopyFileTool,
    DeleteFileTool,
    FileSearchTool,
    ListDirectoryTool,
    MoveFileTool,
    ReadFileTool,
    WriteFileTool,
)
from db_tools import (
    SaveMLResultInput,
    DatabaseQueryInput,
    QueryMLResultsInput,
    save_ml_result_to_db,
    query_execution_history,
    query_ml_results_from_db,
)
from config import (
        SYSTEM_PROMPT, 
        EXECUTE_CODE_TOOL_DESCRIPTION,
        SYSTEM_PROMPT_2,
        QUERY_EXEC_LOGS_TOOL_DESCRIPTION,
        QUERY_ML_RESULTS_TOOL_DESCRIPTION,
        SAVE_ML_RESULT_TOOL_DESCRIPTION
)

# 获取应用设置
app_settings = config_manager.get_app_settings()

# 配置页面
st.set_page_config(
    page_title=app_settings.get("title", "AutoML Workflow Agent"),
    page_icon=app_settings.get("page_icon", "🤖"),
    layout=app_settings.get("layout", "wide"),
    initial_sidebar_state=app_settings.get("sidebar_state", "expanded")
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 获取可用的供应商
available_providers = config_manager.get_enabled_providers()
if not available_providers:
    st.error("❌ 没有配置可用的API供应商！请先配置至少一个供应商。")
    st.stop()

# 选择当前使用的供应商
if "current_provider_id" not in st.session_state:
    default_provider = config_manager.get_default_provider()
    st.session_state.current_provider_id = list(available_providers.keys())[0] if available_providers else None

current_provider = config_manager.get_provider(st.session_state.current_provider_id)
if not current_provider or not current_provider.enabled or not current_provider.api_key:
    st.error(f"❌ 当前选择的供应商 {st.session_state.current_provider_id} 未正确配置！")
    st.stop()

# 1. 初始化LLM
try:
    llm = create_llm_from_config(current_provider)
    st.success(f"✅ 已连接到 {current_provider.name} - {current_provider.default_model}")
except Exception as e:
    st.error(f"❌ 初始化LLM失败: {str(e)}")
    st.stop()

# 2. 初始化增强组件
@st.cache_resource
def initialize_enhanced_components():
    """初始化增强组件（使用缓存避免重复初始化）"""
    history_manager = EnhancedHistoryManager()
    ml_tools = EnhancedMLTools()
    resource_monitor = SystemResourceMonitor()
    return history_manager, ml_tools, resource_monitor

history_manager, ml_tools, resource_monitor = initialize_enhanced_components()

execute_in_ml_sandbox_tool = StructuredTool.from_function(
    func=execute_ml_code_in_docker, # 指向我们健壮的后端函数
    name="ExecutePythonInMLSandbox", # 工具的调用名称
    description=EXECUTE_CODE_TOOL_DESCRIPTION, # 上面定义的详细描述
    args_schema=SandboxExecutionInput # Pydantic模型定义输入参数
)
query_system_execution_logs_tool = StructuredTool.from_function(
    func=query_execution_history, # 指向 db_utils.py 中的函数
    name="QuerySystemExecutionLogs",
    description=QUERY_EXEC_LOGS_TOOL_DESCRIPTION,
    args_schema=DatabaseQueryInput # 使用 DatabaseQueryInput
)
save_ml_result_tool = StructuredTool.from_function(
    func=save_ml_result_to_db, # 指向 db_utils.py 中的函数
    name="SaveMachineLearningResult",
    description=SAVE_ML_RESULT_TOOL_DESCRIPTION,
    args_schema=SaveMLResultInput # 使用 SaveMLResultInput
)
query_ml_results_tool = StructuredTool.from_function(
    func=query_ml_results_from_db, # 指向 db_utils.py 中的函数
    name="QueryMachineLearningResults",
    description=QUERY_ML_RESULTS_TOOL_DESCRIPTION,
    args_schema=QueryMLResultsInput # 使用 QueryMLResultsInput
)

tools = [
    execute_in_ml_sandbox_tool,
    query_system_execution_logs_tool,
    save_ml_result_tool,
    query_ml_results_tool
]

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT_2), # 使用我们上面定义的详细系统提示
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# --- Agent和AgentExecutor的创建 (与您之前的代码类似) ---
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,  # 处理解析错误
)

# ------------- Streamlit UI部分 -------------
# ==============================================================================

# 初始化会话状态
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "session_id" not in st.session_state:
    st.session_state.session_id = f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
if "enhanced_agent" not in st.session_state:
    st.session_state.enhanced_agent = EnhancedAIAgent(llm, tools, current_provider.api_key)
if "resource_monitoring" not in st.session_state:
    st.session_state.resource_monitoring = False

# 侧边栏
with st.sidebar:
    st.header("🎛️ 控制面板")
    
    # 添加供应商配置选项卡
    tab1, tab2 = st.tabs(["🔧 供应商配置", "📊 会话控制"])
    
    with tab1:
        # 当前供应商显示
        st.subheader("🤖 当前供应商")
        provider_options = {k: f"{v.name}" for k, v in available_providers.items()}
        
        selected_provider_id = st.selectbox(
            "选择API供应商",
            options=list(provider_options.keys()),
            format_func=lambda x: provider_options[x],
            index=list(provider_options.keys()).index(st.session_state.current_provider_id),
            key="provider_selector"
        )
        
        if selected_provider_id != st.session_state.current_provider_id:
            st.session_state.current_provider_id = selected_provider_id
            st.rerun()
        
        current_provider_display = config_manager.get_provider(st.session_state.current_provider_id)
        st.info(f"**模型**: {current_provider_display.default_model}\n**类型**: {current_provider_display.type.value}")
        
        # 供应商配置界面
        render_provider_config_ui(config_manager)
    
    with tab2:
        # 会话信息
        st.subheader("📊 会话信息")
        st.info(f"**会话ID:** {st.session_state.session_id}")
        
        # Agent状态
        agent_status = st.session_state.enhanced_agent.get_agent_status()
        st.json(agent_status)
    
    # 历史记录搜索
    st.subheader("🔍 历史记录搜索")
    search_text = st.text_input("搜索关键词")
    history_type_filter = st.selectbox(
        "记录类型",
        ["全部"] + [ht.value for ht in HistoryType],
        index=0
    )
    
    # 时间范围
    time_range = st.selectbox(
        "时间范围",
        ["最近1小时", "最近24小时", "最近7天", "最近30天", "全部"],
        index=1
    )
    
    if st.button("🔍 搜索历史"):
        # 构建搜索过滤器
        filters = SearchFilters()
        if search_text:
            filters.search_text = search_text
        if history_type_filter != "全部":
            filters.history_types = [HistoryType(history_type_filter)]
        
        # 设置时间范围
        if time_range != "全部":
            hours_map = {
                "最近1小时": 1,
                "最近24小时": 24,
                "最近7天": 24 * 7,
                "最近30天": 24 * 30
            }
            hours = hours_map[time_range]
            filters.end_time = datetime.datetime.now()
            filters.start_time = filters.end_time - datetime.timedelta(hours=hours)
        
        # 执行搜索
        entries, total_count = history_manager.search_entries(filters, limit=20)
        
        st.write(f"找到 {total_count} 条记录")
        for entry in entries:
            with st.expander(f"{entry.title} - {entry.timestamp.strftime('%Y-%m-%d %H:%M')}"):
                st.write(f"**类型:** {entry.history_type.value}")
                st.write(f"**严重程度:** {entry.severity.value}")
                st.write(f"**描述:** {entry.description}")
                if entry.tags:
                    st.write(f"**标签:** {', '.join(entry.tags)}")
    
    # 导出功能
    st.subheader("📤 导出数据")
    if st.button("导出会话数据"):
        session_data = st.session_state.enhanced_agent.export_session_data()
        st.download_button(
            label="下载JSON",
            data=json.dumps(session_data, ensure_ascii=False, indent=2),
            file_name=f"session_{st.session_state.session_id}.json",
            mime="application/json"
        )
    
    # ML工具控制
    st.subheader("🧠 增强ML工具")
    auto_analysis = st.checkbox("自动数据分析", value=True)
    smart_preprocessing = st.checkbox("智能预处理", value=True)
    auto_model_selection = st.checkbox("自动模型选择", value=False)
    advanced_viz = st.checkbox("高级可视化", value=True)
    
    # 资源监控控制
    st.subheader("📊 资源监控")
    if st.button("🔍 查看当前资源状态"):
        current_usage = resource_monitor.get_current_usage()
        if current_usage:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("CPU使用率", f"{current_usage.cpu_percent:.1f}%")
            with col2:
                st.metric("内存使用率", f"{current_usage.memory_percent:.1f}%")
            with col3:
                st.metric("磁盘使用率", f"{current_usage.disk_percent:.1f}%")
    
    # 启动/停止监控
    if st.button("▶️ 启动资源监控" if not st.session_state.resource_monitoring else "⏹️ 停止资源监控"):
        if not st.session_state.resource_monitoring:
            resource_monitor.start_monitoring()
            st.session_state.resource_monitoring = True
            st.success("资源监控已启动")
        else:
            resource_monitor.stop_monitoring()
            st.session_state.resource_monitoring = False
            st.success("资源监控已停止")
    
    # 显示最近告警
    recent_alerts = resource_monitor.get_alerts(resolved=False, hours=1)
    if recent_alerts:
        st.warning(f"⚠️ 发现 {len(recent_alerts)} 个未解决的资源告警")
        for alert in recent_alerts[:3]:  # 显示最近3个
            st.error(f"{alert['resource_type'].upper()}: {alert['message']}")
    
    # 清理功能
    st.subheader("🧹 维护")
    if st.button("清理旧记录"):
        cleaned_count = history_manager.cleanup_old_entries(days=30)
        st.success(f"清理了 {cleaned_count} 条旧记录")
    
    if st.button("清理资源监控数据"):
        resource_monitor.cleanup_old_data(days=7)
        st.success("清理了7天前的资源监控数据")

# 主界面
app_title = app_settings.get("title", "🤖 Enhanced AutoML Workflow Agent")
app_description = app_settings.get("description", "具备智能记忆、任务规划和增强推理能力的AI助手")

st.title(app_title)
st.markdown(f"### {app_description}")
st.markdown(f"**当前使用**: {current_provider.name} - {current_provider.default_model}")

# 显示Agent状态概要
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("记忆条目", agent_status["memory_summary"]["total_memories"])
with col2:
    st.metric("总任务", agent_status["task_summary"]["total_tasks"])
with col3:
    st.metric("平均完成度", f"{agent_status['task_summary']['avg_completion']:.1f}%")
with col4:
    st.metric("上下文窗口", agent_status["context_window_size"])

# 显示系统资源状态
if st.session_state.resource_monitoring:
    st.subheader("📊 系统资源状态")
    current_usage = resource_monitor.get_current_usage()
    if current_usage:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            cpu_color = "red" if current_usage.cpu_percent > 80 else "orange" if current_usage.cpu_percent > 60 else "green"
            st.metric("CPU使用率", f"{current_usage.cpu_percent:.1f}%", 
                     delta=None, delta_color=cpu_color)
        with col2:
            mem_color = "red" if current_usage.memory_percent > 80 else "orange" if current_usage.memory_percent > 60 else "green"
            st.metric("内存使用率", f"{current_usage.memory_percent:.1f}%", 
                     delta=f"{current_usage.memory_used_gb:.1f}GB", delta_color=mem_color)
        with col3:
            disk_color = "red" if current_usage.disk_percent > 80 else "orange" if current_usage.disk_percent > 60 else "green"
            st.metric("磁盘使用率", f"{current_usage.disk_percent:.1f}%", 
                     delta=f"{current_usage.disk_used_gb:.1f}GB", delta_color=disk_color)
        with col4:
            net_mb = (current_usage.network_io_bytes.get('bytes_recv', 0) + 
                     current_usage.network_io_bytes.get('bytes_sent', 0)) / (1024*1024)
            st.metric("网络IO", f"{net_mb:.1f}MB", delta="总计")
        
        # 显示GPU信息（如果有）
        if current_usage.gpu_usage and current_usage.gpu_usage.get('gpus'):
            st.subheader("🎮 GPU状态")
            for gpu in current_usage.gpu_usage['gpus']:
                st.write(f"**{gpu['name']}**: {gpu['gpu_utilization']}% 使用率, "
                        f"{gpu['memory_percent']:.1f}% 显存, {gpu['temperature_c']}°C")

# 显示对话历史
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage) and message.content:
        with st.chat_message("assistant"):
            st.markdown(message.content)

# 获取用户输入
initial_user_input = st.chat_input("请输入你的指令...")

if initial_user_input:
    # 立即显示并记录用户的初次输入
    with st.chat_message("user"):
        st.markdown(initial_user_input)
    st.session_state.chat_history.append(HumanMessage(content=initial_user_input))

    # 使用增强的Agent处理用户输入
    with st.spinner("🧠 智能分析中..."):
        try:
            # 构建增强上下文
            enhanced_context = {
                "chat_history_length": len(st.session_state.chat_history),
                "session_id": st.session_state.session_id,
                "ml_tools_available": {
                    "auto_analysis": auto_analysis,
                    "smart_preprocessing": smart_preprocessing,
                    "auto_model_selection": auto_model_selection,
                    "advanced_viz": advanced_viz
                },
                "resource_monitoring": st.session_state.resource_monitoring
            }
            
            # 使用增强的Agent处理输入
            enhanced_response = st.session_state.enhanced_agent.process_user_input(
                user_input=initial_user_input,
                context=enhanced_context
            )
            
            # 记录对话到历史
            conversation_entry = HistoryHelpers.create_conversation_entry(
                session_id=st.session_state.session_id,
                user_input=initial_user_input,
                agent_response=enhanced_response["response"]["content"]
            )
            history_manager.add_entry(conversation_entry)
            
            # 显示增强的响应
            with st.chat_message("assistant"):
                st.markdown(enhanced_response["response"]["content"])
                
                # 显示意图分析（可折叠）
                if enhanced_response.get("intent_analysis"):
                    with st.expander("🎯 意图分析"):
                        st.json(enhanced_response["intent_analysis"])
                
                # 显示任务计划（如果有）
                if enhanced_response.get("task_plan"):
                    with st.expander("📋 任务计划"):
                        task_plan = enhanced_response["task_plan"]
                        st.write(f"**任务ID:** {task_plan['task_id']}")
                        st.write(f"**类型:** {task_plan['task_type']}")
                        st.write(f"**优先级:** {task_plan['priority']}")
                        st.write(f"**预估时长:** {task_plan['estimated_duration']}分钟")
                        
                        if task_plan['subtasks']:
                            st.write("**子任务:**")
                            for subtask in task_plan['subtasks']:
                                st.write(f"- {subtask['description']}")
                
                # 显示相关记忆（如果有）
                if enhanced_response.get("relevant_memories"):
                    with st.expander("🧠 相关记忆"):
                        for memory in enhanced_response["relevant_memories"]:
                            st.write(f"**{memory['timestamp'][:19]}:** {memory['content']}")
                            if memory['tags']:
                                st.caption(f"标签: {', '.join(memory['tags'])}")
            
            # 将增强响应添加到历史
            st.session_state.chat_history.append(AIMessage(content=enhanced_response["response"]["content"]))
            
            # 检查是否需要ML工具增强处理
            should_use_ml_tools = any(keyword in initial_user_input.lower() for keyword in 
                ["分析", "数据", "模型", "训练", "预测", "可视化", "特征"])
            
            if should_use_ml_tools:
                with st.spinner("🧠 使用增强ML工具处理..."):
                    try:
                        # 这里可以根据用户输入调用相应的ML工具
                        ml_context = {
                            "auto_analysis_enabled": auto_analysis,
                            "smart_preprocessing_enabled": smart_preprocessing,
                            "auto_model_selection_enabled": auto_model_selection,
                            "advanced_viz_enabled": advanced_viz
                        }
                        
                        # 显示ML工具建议
                        with st.chat_message("assistant"):
                            st.markdown("### 🧠 ML工具建议")
                            if auto_analysis and "分析" in initial_user_input:
                                st.info("💡 建议使用自动数据分析工具来快速了解数据特征")
                            if smart_preprocessing and ("预处理" in initial_user_input or "清洗" in initial_user_input):
                                st.info("💡 建议使用智能预处理工具来自动处理数据质量问题")
                            if auto_model_selection and ("模型" in initial_user_input or "训练" in initial_user_input):
                                st.info("💡 建议使用自动模型选择工具来找到最适合的算法")
                            if advanced_viz and "可视化" in initial_user_input:
                                st.info("💡 建议使用高级可视化工具来创建交互式图表")
                        
                    except Exception as ml_e:
                        logger.warning(f"ML tools enhancement failed: {str(ml_e)}")
            
            # 如果需要执行代码，调用原有的agent_executor
            if "代码" in initial_user_input or "执行" in initial_user_input or "运行" in initial_user_input:
                with st.spinner("🔧 执行代码中..."):
                    try:
                        # 调用原有的agent_executor执行具体任务
                        execution_response = agent_executor.invoke({
                            "input": initial_user_input,
                            "chat_history": st.session_state.chat_history[:-2]
                        })
                        
                        execution_content = execution_response["output"]
                        
                        # 显示执行结果
                        with st.chat_message("assistant"):
                            st.markdown("### 🔧 执行结果")
                            st.markdown(execution_content)
                        
                        # 记录执行结果
                        st.session_state.chat_history.append(AIMessage(content=f"执行结果: {execution_content}"))
                        
                        # 记录执行到历史
                        execution_entry = HistoryHelpers.create_execution_entry(
                            session_id=st.session_state.session_id,
                            code=initial_user_input,
                            result=execution_content,
                            execution_time=2.0,  # 简化处理
                            success=True
                        )
                        history_manager.add_entry(execution_entry)
                        
                    except Exception as exec_e:
                        error_message = f"执行过程中发生错误: {str(exec_e)}"
                        with st.chat_message("assistant"):
                            st.error(error_message)
                        
                        # 记录错误到历史
                        error_entry = HistoryHelpers.create_error_entry(
                            session_id=st.session_state.session_id,
                            error_message=str(exec_e),
                            error_type="execution_error",
                            context={"user_input": initial_user_input}
                        )
                        history_manager.add_entry(error_entry)

        except Exception as e:
            # 如果invoke在任何步骤失败了（包括我们关心的ValidationError）
            print(f"--- [Agent Error Caught] 捕获到异常: {e} ---")
            
            # 格式化错误信息，准备作为新的输入反馈给Agent
            error_feedback_prompt = f"""
            我在尝试执行你上一步的计划时遇到了一个错误。请分析下面的错误信息，并修正你之前的工具调用或计划。

            **错误信息:**
            ```
            {str(e)}
            ```

            请根据这个错误，重新生成一个正确的工具调用。
            """
            
            # 将这个错误反馈作为新的输入，再次调用Agent
            with st.spinner("检测到错误，尝试自我修正..."):
                try:
                    # 我们将错误反馈也作为HumanMessage添加到历史中，让上下文更清晰
                    st.session_state.chat_history.append(HumanMessage(content=error_feedback_prompt))

                    # 再次调用，但这次的input是我们的错误反馈
                    # 历史记录现在包含了原始输入和我们的错误反馈
                    corrected_response = agent_executor.invoke({
                        "input": error_feedback_prompt, 
                        "chat_history": st.session_state.chat_history[:-1] # 传递包含原始输入但不包含我们刚发的错误反馈的历史
                    })

                    ai_response_content = corrected_response["output"]

                    # 显示修正后的最终回复
                    with st.chat_message("assistant"):
                        st.markdown(ai_response_content)
                    
                    # 将最终的成功回复添加到历史
                    st.session_state.chat_history.append(AIMessage(content=ai_response_content))

                except Exception as final_e:
                    # 如果在修正后再次调用仍然失败
                    final_error_message = f"抱歉，在尝试自我修正后，我仍然遇到了一个问题：\n\n```\n{final_e}\n```"
                    with st.chat_message("assistant"):
                        st.error(final_error_message)
                    st.session_state.chat_history.append(AIMessage(content=final_error_message))