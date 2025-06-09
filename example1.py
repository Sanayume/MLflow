from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.tools.file_management import (
    CopyFileTool,
    DeleteFileTool,
    FileSearchTool,
    ListDirectoryTool,
    MoveFileTool,
    ReadFileTool,
    WriteFileTool,
)
from dotenv import load_dotenv
import os

load_dotenv(".env")


BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY_YUNWU")

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-05-20-nothinking",base_url=BASE_URL, api_key=API_KEY)
memory = MemorySaver()
tools = [
    WriteFileTool(),
    ReadFileTool(),
    ListDirectoryTool(),
    FileSearchTool(),
    CopyFileTool(),
    MoveFileTool(),
    DeleteFileTool(),
]

agent = create_react_agent(
    model=model,
    tools=tools,
    memory=memory,
    verbose=True,
)
agent.invoke({"input": "Create a new file called 'test.txt' in the current directory and write 'Hello, world!' to it."})




