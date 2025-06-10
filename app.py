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
from docker_sandbox_tool import  SandboxExecutionInput
from execute_ml_code_in_docker import execute_ml_code_in_docker
from langchain_community.tools.file_management import (
    CopyFileTool,
    DeleteFileTool,
    FileSearchTool,
    ListDirectoryTool,
    MoveFileTool,
    ReadFileTool,
    WriteFileTool,
)

st.set_page_config(page_title="AutoML Workflow Agent", layout="wide")
HOST_ROOT_WORKSPACE = os.path.abspath(os.path.join(os.getcwd(), "agent_workspace"))
CONTAINER_DATASETS_MOUNT_TARGET = "/sandbox/datasets"
CONTAINER_OUTPUTS_MOUNT_TARGET = "/sandbox/outputs"
CONTAINER_CODE_EXECUTION_BASE_TARGET = "/sandbox/code"

# 加载环境变量 (OPENAI_API_KEY)

# 导入我们本地定义的工具 
from local_tools import safe_python_executor

# 1. 初始化LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-05-20", google_api_key="AIzaSyC6ZrlkGvbvmTQ_zPCLsgS-frZH8QDX5mI") # 或者 gpt-4-turbogem

# In app.py
# ...
'''
tools = [
    Tool(
        name="safe_python_executor",
        func=safe_python_executor,
        description="在隔离的环境中安全地执行多行Python代码字符串..." # (同前)
    ),
    Tool(
        name="read_local_file",
        func=read_local_file,
        description="读取指定路径的本地文件的内容。文件路径必须位于预定义的安全工作目录内..." # (同前)
    ),
    Tool(
        name="write_local_file",
        # 修改func，让它接收一个字典，然后解包参数
        func=lambda tool_input: write_local_file(
            file_path=tool_input['file_path'],
            content=tool_input['content'],
            current_ai_relative_dir=st.session_state.ai_current_relative_dir
        ),
        description="""将指定内容写入相对于AI当前工作目录的文件。
        如果父目录不存在则创建。所有操作都在安全根工作区内。
        输入必须是一个JSON对象（字典），包含 'file_path' (字符串) 和 'content' (字符串) 两个键。
        例如：{"file_path": "my_folder/output.txt", "content": "这是文件内容"}
        """,
    ),
    Tool(
        name="list_directory_items_with_paths", # 新工具名称
        func=list_directory_items_with_paths, # 对应的函数
        description="""
        列出指定本地目录下的项目（文件和子目录）及其类型和相对于安全工作目录的路径。
        输入参数 'directory_path' (字符串, 可选, 默认为'.') 是相对于安全工作目录的路径。
        例如：'.' 列出安全工作目录的根，'my_subdir' 列出安全工作目录下的my_subdir子目录。
        返回一个包含 'items' (一个对象列表，每个对象包含 'name', 'type', 'relative_path') 或 'error' 的字典。
        'relative_path' 可以直接用于其他文件操作工具（如 read_local_file, write_local_file）的 file_path 参数。
        """
    ),
     # 如果你添加了网络请求工具:
    Tool(
         name="make_web_request",
         func=make_web_request,
         description="""
         向指定的URL发出一个GET网络请求，并返回响应文本的前500个字符。
         警告：使用此工具时需谨慎，确保URL是可信的。仅用于获取公开信息。
         返回一个包含'content' (部分响应文本)或'error'的字典。
         """
    ),
]
'''
#加入我们的docker_sandbox_tool
# --- 创建 ExecuteInMLSandboxTool ---
# 这是关键的工具描述！
EXECUTE_CODE_TOOL_DESCRIPTION = f"""
此工具名为 'ExecutePythonInMLSandbox'，用于在一个安全的、隔离的Docker容器（基于'mlsandbox:latest'镜像）中执行你提供的Python代码。
该沙箱环境预装了Python 3.12和常用的机器学习库，包括 Pandas, Scikit-learn, NumPy, Matplotlib, Seaborn, 以及PyTorch和TensorFlow的CPU版本。

**核心功能与规则**:
1.  **代码执行**: 你提供完整的Python代码字符串 (`code_string`)，工具会将其保存为一个脚本文件并在沙箱中执行。
2.  **脚本组织**:
    *   通过 `script_relative_path` 参数，你可以指定脚本在本次执行的代码区根目录 (`{CONTAINER_CODE_EXECUTION_BASE_TARGET}/`) 内的相对存放路径 (例如 '.', 'preprocessing', 'training_scripts')。
    *   通过 `script_filename` 参数，你为脚本指定一个文件名 (必须以 '.py' 结尾，例如 'data_cleaning.py')。
3.  **代码描述与目的**:
    *   通过 `ai_code_description` 参数，提供对你代码的简短描述。
    *   通过 `ai_code_purpose` 参数，说明执行该代码的目的。
4.  **资源配置**:
    *   **GPU支持**: 通过 `use_gpu` 参数请求GPU资源。
    *   **资源限制 (未来功能)**: 你可以指定 `cpu_core_limit` 和 `memory_limit` 参数。尽管它们目前不会实际限制资源，但会被记录下来，用于未来的资源管理。
5.  **文件系统约定 (极其重要，必须在你的 `code_string` 中严格遵守)**:
    *   **所有文件路径**: 在你的 `code_string` 中，所有文件路径都 **必须** 使用Linux风格的正斜杠 `/` 作为路径分隔符。
    *   **读取数据集**: 你的代码应该从容器内的 `{CONTAINER_DATASETS_MOUNT_TARGET}/` 目录读取数据。
        *   例如: `import pandas as pd; df = pd.read_csv('{CONTAINER_DATASETS_MOUNT_TARGET}/my_input_data.csv')`
        *   你需要假设用户已经将必要的数据文件放置在宿主机映射到此路径的目录 (`{HOST_ROOT_WORKSPACE}/datasets/`) 中。如果需要特定文件，请先询问用户是否已准备好。
    *   **保存输出**: 你的代码应该将所有输出（如训练好的模型、结果CSV文件、图片、日志文件等）保存到容器内的 `{CONTAINER_OUTPUTS_MOUNT_TARGET}/` 目录下的子路径和文件名，这些由你的代码逻辑决定。
        *   例如: `import joblib; joblib.dump(model, '{CONTAINER_OUTPUTS_MOUNT_TARGET}/my_models/logistic_regression_v1.pkl')`
        *   或: `results_df.to_csv('{CONTAINER_OUTPUTS_MOUNT_TARGET}/experiment_results/run_001/predictions.csv')`
        *   这些文件会自动出现在宿主机映射到此路径的目录 (`{HOST_ROOT_WORKSPACE}/outputs/`) 中。
    *   **临时文件**: 如果代码需要创建不希望持久化到宿主机的临时文件，可以在容器内的 `/tmp/` 目录操作。
6.  **Python代码要求**:
    *   你的 `code_string` 必须是完整且可直接执行的Python脚本内容。
    *   必须包含所有必要的 `import` 语句。
    *   使用 `print()` 语句输出所有重要的结果、指标、状态或日志信息，以便它们能通过stdout返回并被记录。
7.  **错误处理**: 如果代码执行出错，`stderr`会包含错误信息。更重要的是，请检查 `error_type` 字段来快速判断错误的类别。

**工具输出**:
该工具会返回一个JSON对象（字典），包含以下关键字段：
- `execution_id` (字符串): 本次执行的唯一ID。
- `success` (布尔值): 如果脚本成功执行 (退出码为0)，则为 True。
- `exit_code` (整数): Python脚本的退出码。
- `error_type` (字符串或null): **新增字段**，提供结构化的错误分类。可能的值包括:
    - `'PREPARATION_ERROR'`: 容器启动前的错误（如Docker连接、镜像查找）。
    - `'RUNTIME_ERROR'`: 容器内Python代码执行时出错。
    - `'TIMEOUT_ERROR'`: 容器执行超时。
    - `'DOCKER_API_ERROR'`: 其他与Docker守护进程交互时发生的运行时错误。
    - `null`: 执行成功时此字段为空。
- `stdout` (字符串): Python脚本的标准输出内容。
- `stderr` (字符串): Python脚本的标准错误输出内容。
- `log_directory_host_path` (字符串): 本次执行的所有日志文件在宿主机上保存的目录路径。
- `execution_duration_seconds` (浮点数): 容器内代码实际执行的耗时。
- `cpu_core_limit_set` (浮点数或null): **新增字段**，记录你请求的CPU核心限制。
- `memory_limit_set` (字符串或null): **新增字段**，记录你请求的内存限制。
- `error_message_preprocessing` / `error_message_runtime`: 具体的错误信息文本。
**工具输出**:
该工具会返回一个JSON对象（字典），包含以下关键字段：
- `execution_id` (字符串): 本次执行的唯一ID，可用于追踪日志。
- `success` (布尔值): 如果脚本成功执行 (退出码为0)，则为 True。
- `exit_code` (整数): Python脚本的退出码。
- `stdout` (字符串): Python脚本的标准输出内容。
- `stderr` (字符串): Python脚本的标准错误输出内容 (如果有错误)。
- `log_directory_host_path` (字符串): 本次执行的所有日志文件（代码副本、stdout、stderr、元数据）在宿主机上保存的目录路径。
- `executed_script_container_path` (字符串): 在容器内实际执行的脚本的完整路径。
- `execution_duration_seconds` (浮点数): 容器内代码实际执行的耗时（秒）。
- `error_message_preprocessing` (字符串或null): 如果在执行代码前发生错误（如Docker问题），则包含错误信息。
- `error_message_runtime` (字符串或null): 如果在容器执行期间或之后发生错误（如超时），则包含错误信息。

**使用场景**:
当你需要执行数据预处理、特征工程、模型训练、模型评估、数据可视化（结果保存为文件）、或任何其他需要Python和已安装的机器学习库的计算任务时，请使用此工具。
在生成 `code_string` 之前，请仔细思考任务的步骤，并确保代码的健壮性。
执行后，请务必分析工具返回的 `success`, `stdout`, 和 `stderr` 来判断任务是否成功，并从中提取关键信息进行后续决策或向用户报告。
"""

execute_in_ml_sandbox_tool = StructuredTool.from_function(
    func=execute_ml_code_in_docker, # 指向我们健壮的后端函数
    name="ExecutePythonInMLSandbox", # 工具的调用名称
    description=EXECUTE_CODE_TOOL_DESCRIPTION, # 上面定义的详细描述
    args_schema=SandboxExecutionInput # Pydantic模型定义输入参数
)
tools = [
    WriteFileTool(),
    ReadFileTool(),
    ListDirectoryTool(),
    FileSearchTool(),
    CopyFileTool(),
    MoveFileTool(),
    DeleteFileTool(),
    execute_in_ml_sandbox_tool,
]
memory = MemorySaver()
# ... (Agent的Prompt, Agent创建, Agent Executor创建等保持不变)

# 3. 创建Agent的Prompt
# 参考: https://python.langchain.com/docs/modules/agents/agent_types/openai_tools
# 你可以根据需要定制这个prompt
# app.py (继续)

# --- 更新Agent的System Prompt ---
SYSTEM_PROMPT = f"""
你是一个名为 "AutoML Workflow Agent" 的高级AI助手，专门负责通过在安全的Docker沙箱环境中执行Python代码来完成复杂的机器学习任务和实验。

**你的核心工作流程是：**
1.  **理解用户请求**: 仔细分析用户的目标和提供的任何关于数据或任务的上下文。
2.  **规划执行步骤**: 将复杂的任务分解为一系列可通过Python代码执行的逻辑步骤（例如：数据加载与探索 -> 数据预处理与特征工程 -> 模型选择与训练 -> 模型评估与结果保存 -> 结果分析与报告）。
3.  **生成Python代码**: 为每个步骤编写完整、健壮、可直接执行的Python代码。
4.  **调用沙箱工具执行代码**: 使用你唯一的代码执行工具 `ExecutePythonInMLSandbox` 来运行你生成的Python代码。
5.  **分析执行结果**: 仔细检查工具返回的 `success`状态、`stdout`（标准输出）和`stderr`（标准错误）。
6.  **迭代或报告**: 如果成功，从`stdout`中提取关键结果并向用户报告，或进行下一步规划。如果失败，分析`stderr`中的错误信息，尝试理解原因，你可以选择：
    *   向用户报告错误并请求澄清或修正。
    *   尝试修改你的Python代码并重新执行（如果错误是可修复的编码问题）。
    *   如果需要，向用户请求更多信息。

**与 `ExecutePythonInMLSandbox` 工具交互的关键指南:**

*   **工具调用**: 当你需要执行Python代码时，你必须调用 `ExecutePythonInMLSandbox` 工具。
*   **输入参数**:
    *   `code_string`: 你生成的完整Python代码。
    *   `script_relative_path`: 你为这个脚本规划的在本次执行的代码区内的相对路径 (例如 '.', 'data_preparation', 'model_training')。
    *   `script_filename`: 你为脚本指定的文件名 (例如 'load_data.py', 'train_xgboost.py')。
    *   `ai_code_description`: 对你这段代码功能的简短描述。
    *   `ai_code_purpose`: 执行这段代码的目的。
    *   `use_gpu`: 如果你认为任务适合GPU加速且代码支持，则设为 `True`。
*   **Python代码 (`code_string`) 编写规范**:
    *   **完整性**: 代码必须是自包含的，包含所有必要的 `import` 语句。
    *   **路径分隔符**: **所有、所有、所有** 文件路径都 **必须** 使用Linux风格的正斜杠 `/`。例如: `'/sandbox/outputs/my_model.pkl'`。
    *   **读取数据集**:
        *   代码应从容器内的 `{CONTAINER_DATASETS_MOUNT_TARGET}/` 目录读取。例如: `df = pd.read_csv('{CONTAINER_DATASETS_MOUNT_TARGET}/user_data/titanic.csv')`。
        *   **重要**: 在尝试读取文件前，如果用户没有明确说明文件已存在于该路径，你应该先礼貌地询问用户：“请确认您已将名为 'titanic.csv' 的数据集放置在宿主机的 `{HOST_ROOT_WORKSPACE}/datasets/user_data/` 目录下，以便我可以通过 `{CONTAINER_DATASETS_MOUNT_TARGET}/user_data/titanic.csv` 进行访问。准备好后请告诉我。” 或者类似措辞。不要假设文件一定存在。
    *   **保存输出/工件**:
        *   所有代码生成的输出（模型文件、结果CSV、图片、日志等）都应保存到容器内的 `{CONTAINER_OUTPUTS_MOUNT_TARGET}/` 目录下的、由你（AI）决定的有意义的子路径和文件名。
        *   例如: `model.save('{CONTAINER_OUTPUTS_MOUNT_TARGET}/experiments/exp001/model_v1.h5')` 或 `fig.savefig('{CONTAINER_OUTPUTS_MOUNT_TARGET}/plots/feature_importance.png')`。
        *   这些文件之后可以在宿主机的 `{HOST_ROOT_WORKSPACE}/outputs/` 下找到。
    *   **打印输出**: 使用 `print()` 语句输出所有重要的中间结果、最终指标、状态信息或调试信息。工具的 `stdout` 将捕获这些打印内容。这是你向我（以及用户）展示工作进展和结果的主要方式。
*   **结果解读**:
    *   检查返回的 `success` (布尔值) 和 `exit_code` (整数)。`success: True` 且 `exit_code: 0` 表示代码执行无误。
    *   仔细阅读 `stdout` 获取代码的打印输出和结果。
    *   如果 `success: False` 或 `exit_code != 0`，必须检查 `stderr` 以了解详细的错误信息和Traceback。
    *   `log_directory_host_path` 字段告诉你本次执行的所有相关文件（你的代码副本、stdout日志、stderr日志、元数据json）在宿主机上保存的位置，这主要用于调试和审计，你通常不需要直接操作它。

**你的行为准则**:
*   **务实**: 一步一步地解决问题。如果任务复杂，将其分解。
*   **清晰**: 你的代码和解释都应该清晰易懂。
*   **严谨**: 特别是在处理文件路径和解读工具输出时。
*   **主动沟通**: 如果用户指令不明确，或者你需要确认数据准备情况，请主动提问。
*   **错误处理**: 不要害怕错误，但要学会从错误中学习并尝试解决。

现在，请等待用户的指令。
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT), # 使用我们上面定义的详细系统提示
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