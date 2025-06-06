import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from dotenv import load_dotenv
from local_tools import (
    safe_python_executor, 
    read_local_file, 
    write_local_file, 
    # list_directory_contents, # 可以替换掉旧的
    list_directory_items_with_paths, # 新的工具
    make_web_request 
)

# 加载环境变量 (OPENAI_API_KEY)

# 导入我们本地定义的工具
from local_tools import safe_python_executor

# 1. 初始化LLM
llm = ChatOpenAI(model="gemini-2.5-flash-preview-05-20-nothinking",base_url="https://yunwu.ai/v1", api_key="sk-l49VZoOM4U0WrNca7q1mJaxQzhEAunTiEnUg9ph70Pv4OtyY") # 或者 gpt-4-turbogem

# In app.py
# ...

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

# ... (Agent的Prompt, Agent创建, Agent Executor创建等保持不变)

# 3. 创建Agent的Prompt
# 参考: https://python.langchain.com/docs/modules/agents/agent_types/openai_tools
# 你可以根据需要定制这个prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个乐于助人的AI助手。你可以使用工具来回答问题或执行任务。"),
        MessagesPlaceholder(variable_name="chat_history", optional=True), # 用于存储对话历史
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"), # Agent思考和工具调用的中间步骤
    ]
)

# 4. 创建Agent
# create_openai_tools_agent 会使用OpenAI的tool calling功能
agent = create_openai_tools_agent(llm, tools, prompt)

# 5. 创建Agent Executor (负责实际运行Agent、调用工具、处理循环)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True) # verbose=True 会打印Agent的思考过程

# ------------- Streamlit UI部分 -------------
st.title("本地工具交互AI助手 (LangChain + Streamlit)")

# 初始化对话历史 (在Streamlit的session_state中)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 显示已有的对话消息
for message_data in st.session_state.chat_history:
    if message_data["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message_data["content"])
    elif message_data["role"] == "assistant":
        with st.chat_message("assistant"):
            st.markdown(message_data["content"])

# 获取用户输入
user_input = st.chat_input("请输入你的指令或问题...")

if user_input:
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # 添加用户消息到对话历史
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # 调用Agent Executor处理输入
    # 将 LangChain 的 HumanMessage/AIMessage 转换为字典以便存储
    langchain_chat_history = []
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            langchain_chat_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            # 这里的content可能包含复杂的AIMessage对象，需要处理
            # 简单起见，我们假设之前的assistant消息是纯文本
            # 实际上，如果之前的AIMessage包含tool_calls，处理会更复杂
            # 对于简单的历史，可以只传递文本内容
            langchain_chat_history.append(AIMessage(content=msg["content"]))
            # 如果AIMessage有tool_calls，需要正确地构造它。
            # agent_executor需要的是Message对象列表

    # 准备调用agent_executor的输入
    # 注意：LangChain Agent期望的chat_history是Message对象的列表
    # 我们需要转换存储的字典历史
    processed_history = []
    # 查找AIMessage中可能存在的tool_calls，因为它们是agent_executor正确工作所必需的
    # 这是简化处理，实际应用中可能需要更仔细地管理ToolMessage的重建
    temp_langchain_history = []
    for h_msg in st.session_state.chat_history[:-1]: # 除了当前用户输入外的所有历史
        if h_msg['role'] == 'user':
            temp_langchain_history.append(HumanMessage(content=h_msg['content']))
        elif h_msg['role'] == 'assistant':
             # 这里需要更精细地处理，如果AIMessage有tool_calls，则需要重建AIMessage(content="", tool_calls=[...])
             # 简单处理：
            temp_langchain_history.append(AIMessage(content=h_msg['content']))


    response = agent_executor.invoke({
        "input": user_input,
        "chat_history": temp_langchain_history # 传递处理过的对话历史
    })
    
    ai_response_content = response["output"]

    # 显示AI回复
    with st.chat_message("assistant"):
        st.markdown(ai_response_content)
        
    # 添加AI回复到对话历史
    st.session_state.chat_history.append({"role": "assistant", "content": ai_response_content})