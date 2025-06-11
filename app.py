# app.py

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import StructuredTool
import os
from local_tools import (
    safe_python_executor, 
    read_local_file, 
    write_local_file, 
    # list_directory_contents, # 可以替换掉旧的
    list_directory_items_with_paths, # 新的工具
    make_web_request 
)
from sanbox_main import (
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
from config import (SYSTEM_PROMPT, 
        EXECUTE_CODE_TOOL_DESCRIPTION,
        SYSTEM_PROMPT_2,
        QUERY_EXEC_LOGS_TOOL_DESCRIPTION,
        QUERY_ML_RESULTS_TOOL_DESCRIPTION,
        SAVE_ML_RESULT_TOOL_DESCRIPTION
)

st.set_page_config(page_title="AutoML Workflow Agent", layout="wide")

# 加载环境变量 (OPENAI_API_KEY)

# 导入我们本地定义的工具 
from local_tools import safe_python_executor

# 1. 初始化LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-05-20", google_api_key=) # 或者 gpt-4-turbogem

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
#st.set_page_config(page_title="AutoML Workflow Agent", layout="wide")
st.title("🤖 AutoML Workflow Agent")

# 初始化对话历史 (仍然使用Message对象)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

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

    # --- 最终优化的错误反馈逻辑 ---
    
    # 准备第一次调用的输入
    current_input = initial_user_input
    
    with st.spinner("思考中..."):
        try:
            # 直接调用agent_executor
            response = agent_executor.invoke({
                "input": current_input,
                "chat_history": st.session_state.chat_history[:-1] # 传递到当前用户输入之前的所有历史
            })
            
            # 如果invoke成功，没有抛出任何异常
            ai_response_content = response["output"]

            # 显示AI的最终回复
            with st.chat_message("assistant"):
                st.markdown(ai_response_content)
                
            # 将AI的最终成功回复添加到历史
            st.session_state.chat_history.append(AIMessage(content=ai_response_content))

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