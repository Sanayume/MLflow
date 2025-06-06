# In local_tools.py

import os

# 定义一个安全的工作目录 (可以从环境变量或配置中读取)
# 为了简单起见，我们先硬编码一个子目录
SAFE_WORKING_DIRECTORY = os.path.join(os.getcwd(), "ai_workspace") 
# 确保这个目录存在
if not os.path.exists(SAFE_WORKING_DIRECTORY):
    os.makedirs(SAFE_WORKING_DIRECTORY, exist_ok=True)

def read_local_file(file_path: str) -> dict:
    """
    读取指定路径的本地文件的内容。
    文件路径必须位于预定义的安全工作目录内。
    返回一个包含'content' (文件内容字符串)或'error' (错误信息)的字典。
    """
    print(f"--- [本地工具日志] 尝试读取文件: {file_path} ---")
    try:
        # 安全性检查：确保路径在允许的目录下
        full_path = os.path.abspath(os.path.join(SAFE_WORKING_DIRECTORY, file_path))
        
        if not full_path.startswith(os.path.abspath(SAFE_WORKING_DIRECTORY)):
            error_msg = "错误：禁止访问指定路径之外的文件。"
            print(f"--- [本地工具日志] {error_msg} ---")
            return {"content": None, "error": error_msg}

        if not os.path.exists(full_path) or not os.path.isfile(full_path):
            error_msg = f"错误：文件 '{file_path}' 不存在或不是一个文件。"
            print(f"--- [本地工具日志] {error_msg} ---")
            return {"content": None, "error": error_msg}

        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"--- [本地工具日志] 文件读取成功，内容长度: {len(content)} ---")
        return {"content": content, "error": None}
    except Exception as e:
        error_msg = f"读取文件时发生错误: {str(e)}"
        print(f"--- [本地工具日志] {error_msg} ---")
        return {"content": None, "error": error_msg}