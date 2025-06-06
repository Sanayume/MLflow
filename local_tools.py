import subprocess
import os
import tempfile

def safe_python_executor(code_string: str) -> dict:
    """
    在隔离的子进程中安全地执行多行Python代码字符串，并返回其标准输出和标准错误。
    适用于执行用户提供的或AI生成的Python片段。
    """
    print(f"--- [本地工具日志] 准备执行代码:\n{code_string}\n---")
    
    # 创建一个临时文件来保存代码
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_file:
        tmp_file.write(code_string)
        script_path = tmp_file.name

    try:
        # 使用subprocess在新进程中运行脚本
        # 你可以进一步考虑使用Docker进行更强隔离，但subprocess是基础起点
        process = subprocess.Popen(
            ["python", script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8' # 确保正确处理各种字符
        )
        stdout, stderr = process.communicate(timeout=30) # 设置超时

        if process.returncode == 0:
            print(f"--- [本地工具日志] 执行成功，输出:\n{stdout}\n---")
            return {"output": stdout, "error": stderr if stderr else None}
        else:
            print(f"--- [本地工具日志] 执行错误，错误信息:\n{stderr}\n---")
            return {"output": stdout if stdout else None, "error": stderr}
            
    except subprocess.TimeoutExpired:
        print("--- [本地工具日志] 代码执行超时 ---")
        return {"output": None, "error": "Code execution timed out after 30 seconds."}
    except Exception as e:
        error_message = f"An unexpected error occurred during execution: {str(e)}"
        print(f"--- [本地工具日志] {error_message} ---")
        return {"output": None, "error": error_message}
    finally:
        # 清理临时文件
        if os.path.exists(script_path):
            os.remove(script_path)

# (可选) 定义更多本地工具，例如：
# def read_local_file(file_path: str) -> dict: ...
# def query_local_mysql(query: str) -> dict: ...

# In local_tools.py

# In local_tools.py

import os

# 定义一个安全的工作目录 (可以从环境变量或配置中读取)
# 为了简单起见，我们先硬编码一个子目录
SAFE_WORKING_DIRECTORY = os.path.join(os.getcwd(), "ai_workspace") 
# 确保这个目录存在
if not os.path.exists(SAFE_WORKING_DIRECTORY):
    os.makedirs(SAFE_WORKING_DIRECTORY, exist_ok=True)

# In local_tools.py
def read_local_file(file_path: str, current_ai_relative_dir: str = ".") -> dict:
    """
    读取指定路径的本地文件的内容。
    文件路径是相对于AI当前工作目录的。
    所有操作都不能超出预定义的安全根工作区。
    返回一个包含'content'或'error'的字典。
    """
    print(f"--- [本地工具日志] 尝试读取文件: '{file_path}' (AI当前相对目录: '{current_ai_relative_dir}') ---")
    try:
        base_dir_for_operation = os.path.abspath(os.path.join(SAFE_WORKING_DIRECTORY, current_ai_relative_dir))
        if not base_dir_for_operation.startswith(SAFE_WORKING_DIRECTORY):
            return {"content": None, "error": "错误：AI当前相对目录解析后超出了安全根工作区。"}

        absolute_target_file_path = os.path.abspath(os.path.join(base_dir_for_operation, file_path))

        if not absolute_target_file_path.startswith(SAFE_WORKING_DIRECTORY):
            return {"content": None, "error": f"错误：目标文件路径 '{absolute_target_file_path}' 超出了安全根工作区。"}

        if not os.path.exists(absolute_target_file_path) or not os.path.isfile(absolute_target_file_path):
            return {"content": None, "error": f"错误：文件 '{file_path}' (在 {absolute_target_file_path}) 不存在或不是一个文件。"}

        with open(absolute_target_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"--- [本地工具日志] 文件读取成功，内容长度: {len(content)} ---")
        return {"content": content, "error": None}
    except Exception as e:
        return {"content": None, "error": f"读取文件时发生错误: {str(e)}"}
# In local_tools.py
def list_directory_items_with_paths(directory_path: str = ".", current_ai_relative_dir: str = ".") -> dict:
    """
    列出指定本地目录下的项目及其类型和相对于安全根工作区的路径。
    'directory_path'是相对于'current_ai_relative_dir'的。
    'current_ai_relative_dir'是相对于安全根工作区的。
    返回 'items' 列表或 'error'。
    """
    print(f"--- [本地工具日志] 尝试列出目录: '{directory_path}' (AI当前相对目录: '{current_ai_relative_dir}') ---")
    try:
        base_dir_for_operation = os.path.abspath(os.path.join(SAFE_WORKING_DIRECTORY, current_ai_relative_dir))
        if not base_dir_for_operation.startswith(SAFE_WORKING_DIRECTORY):
            return {"items": None, "error": "错误：AI当前相对目录解析后超出了安全根工作区。"}

        path_to_list_input = directory_path # 用户输入的，可能是 "." 或 "subdir"
        absolute_path_to_list = os.path.abspath(os.path.join(base_dir_for_operation, path_to_list_input))

        if not absolute_path_to_list.startswith(SAFE_WORKING_DIRECTORY):
            return {"items": None, "error": f"错误：要列出的目录路径 '{absolute_path_to_list}' 超出了安全根工作区。"}
        
        if not os.path.exists(absolute_path_to_list) or not os.path.isdir(absolute_path_to_list):
            return {"items": None, "error": f"错误：目录 '{path_to_list_input}' (在 {absolute_path_to_list}) 不存在或不是一个目录。"}
            
        items_details = []
        for item_name in os.listdir(absolute_path_to_list):
            item_abs_path_in_listed_dir = os.path.join(absolute_path_to_list, item_name)
            relative_path_to_root = os.path.relpath(item_abs_path_in_listed_dir, SAFE_WORKING_DIRECTORY)
            item_type = "file" if os.path.isfile(item_abs_path_in_listed_dir) else "directory" if os.path.isdir(item_abs_path_in_listed_dir) else "unknown"
            
            items_details.append({
                "name": item_name,
                "type": item_type,
                "relative_path_from_root": relative_path_to_root 
            })
            
        print(f"--- [本地工具日志] 目录项目详情: {items_details} ---")
        return {"items": items_details, "error": None}
    except Exception as e:
        return {"items": None, "error": f"列出目录项目时发生错误: {str(e)}"}
def write_local_file(file_path: str, content: str) -> dict:
    """
    将指定内容写入本地文件。
    文件路径必须位于预定义的安全工作目录内。如果文件已存在，它将被覆盖。
    返回一个包含'status' ('success'或'error')和'message'的字典。
    """
    print(f"--- [本地工具日志] 尝试写入文件: {file_path}，内容长度: {len(content)} ---")
    try:
        # 安全性检查
        full_path = os.path.abspath(os.path.join(SAFE_WORKING_DIRECTORY, file_path))
        
        if not full_path.startswith(os.path.abspath(SAFE_WORKING_DIRECTORY)):
            error_msg = "错误：禁止在指定路径之外写入文件。"
            print(f"--- [本地工具日志] {error_msg} ---")
            return {"status": "error", "message": error_msg}

        # 确保目标目录存在 (如果file_path包含子目录)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
        success_msg = f"文件 '{file_path}' 写入成功。"
        print(f"--- [本地工具日志] {success_msg} ---")
        return {"status": "success", "message": success_msg}
    except Exception as e:
        error_msg = f"写入文件时发生错误: {str(e)}"
        print(f"--- [本地工具日志] {error_msg} ---")
        return {"status": "error", "message": error_msg}
    
# In local_tools.py
import requests

# 简单的请求示例，生产环境中需要更严格的控制
def make_web_request(url: str) -> dict:
    """
    向指定的URL发出一个GET网络请求，并返回响应文本的前500个字符。
    警告：使用此工具时需谨慎，确保URL是可信的。
    返回一个包含'content' (部分响应文本)或'error'的字典。
    """
    print(f"--- [本地工具日志] 尝试发出GET请求到: {url} ---")
    try:
        # TODO: 在生产环境中，应该有URL白名单或更严格的验证
        if not url.startswith("http://") and not url.startswith("https://"):
            return {"content": None, "error": "无效的URL格式。必须以http://或https://开头。"}

        response = requests.get(url, timeout=10) # 设置超时
        response.raise_for_status() # 如果是4xx或5xx错误，则抛出异常
        
        content_preview = response.text[:500] # 只取前500个字符作为预览
        print(f"--- [本地工具日志] 请求成功，状态码: {response.status_code} ---")
        return {"content": content_preview, "error": None}
    except requests.exceptions.RequestException as e:
        error_msg = f"进行网络请求时发生错误: {str(e)}"
        print(f"--- [本地工具日志] {error_msg} ---")
        return {"content": None, "error": error_msg}
    except Exception as e:
        error_msg = f"处理网络请求时发生意外错误: {str(e)}"
        print(f"--- [本地工具日志] {error_msg} ---")
        return {"content": None, "error": error_msg}

