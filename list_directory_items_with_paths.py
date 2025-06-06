<<<<<<< HEAD
# In local_tools.py

import os

# (SAFE_WORKING_DIRECTORY 定义同前)
SAFE_WORKING_DIRECTORY = os.path.join(os.getcwd(), "ai_workspace") 
# if not os.path.exists(SAFE_WORKING_DIRECTORY):
#     os.makedirs(SAFE_WORKING_DIRECTORY, exist_ok=True)

def list_directory_items_with_paths(directory_path: str = ".") -> dict:
    """
    列出指定本地目录下的项目（文件和子目录）及其类型和相对路径。
    路径是相对于预定义的安全工作目录的。如果留空或使用'.'，则列出工作目录的根。
    返回一个包含 'items' (一个对象列表，每个对象包含 'name', 'type', 'relative_path') 
    或 'error' 的字典。
    """
    print(f"--- [本地工具日志] 尝试列出目录项目及其路径: {directory_path} ---")
    try:
        # 安全性检查: 确保路径在允许的目录下
        # base_dir_to_list 是用户请求的、相对于SAFE_WORKING_DIRECTORY的目录
        # full_path_to_list 是其在系统上的绝对路径
        full_path_to_list = os.path.abspath(os.path.join(SAFE_WORKING_DIRECTORY, directory_path))
        
        if not full_path_to_list.startswith(os.path.abspath(SAFE_WORKING_DIRECTORY)):
            error_msg = "错误：禁止访问指定路径之外的目录。"
            print(f"--- [本地工具日志] {error_msg} ---")
            return {"items": None, "error": error_msg}

        if not os.path.exists(full_path_to_list) or not os.path.isdir(full_path_to_list):
            error_msg = f"错误：目录 '{directory_path}' (在安全工作区内解析为 '{full_path_to_list}') 不存在或不是一个目录。"
            print(f"--- [本地工具日志] {error_msg} ---")
            return {"items": None, "error": error_msg}
            
        items_details = []
        for item_name in os.listdir(full_path_to_list):
            item_abs_path = os.path.join(full_path_to_list, item_name)
            
            # 计算相对于 SAFE_WORKING_DIRECTORY 的路径
            # 我们需要从 item_abs_path 中移除 SAFE_WORKING_DIRECTORY 前缀
            # 并确保处理好路径分隔符
            relative_path_to_safe_dir = os.path.relpath(item_abs_path, SAFE_WORKING_DIRECTORY)
            # 确保路径分隔符在不同系统上一致性（可选，通常os.path系列函数会处理）
            # relative_path_to_safe_dir = relative_path_to_safe_dir.replace(os.sep, '/')


            item_type = "file" if os.path.isfile(item_abs_path) else "directory" if os.path.isdir(item_abs_path) else "unknown"
            
            items_details.append({
                "name": item_name,
                "type": item_type,
                "relative_path": relative_path_to_safe_dir 
            })
            
        print(f"--- [本地工具日志] 目录项目详情: {items_details} ---")
        return {"items": items_details, "error": None}
    except Exception as e:
        error_msg = f"列出目录项目时发生错误: {str(e)}"
        print(f"--- [本地工具日志] {error_msg} ---")
=======
# In local_tools.py

import os

# (SAFE_WORKING_DIRECTORY 定义同前)
SAFE_WORKING_DIRECTORY = os.path.join(os.getcwd(), "ai_workspace") 
# if not os.path.exists(SAFE_WORKING_DIRECTORY):
#     os.makedirs(SAFE_WORKING_DIRECTORY, exist_ok=True)

def list_directory_items_with_paths(directory_path: str = ".") -> dict:
    """
    列出指定本地目录下的项目（文件和子目录）及其类型和相对路径。
    路径是相对于预定义的安全工作目录的。如果留空或使用'.'，则列出工作目录的根。
    返回一个包含 'items' (一个对象列表，每个对象包含 'name', 'type', 'relative_path') 
    或 'error' 的字典。
    """
    print(f"--- [本地工具日志] 尝试列出目录项目及其路径: {directory_path} ---")
    try:
        # 安全性检查: 确保路径在允许的目录下
        # base_dir_to_list 是用户请求的、相对于SAFE_WORKING_DIRECTORY的目录
        # full_path_to_list 是其在系统上的绝对路径
        full_path_to_list = os.path.abspath(os.path.join(SAFE_WORKING_DIRECTORY, directory_path))
        
        if not full_path_to_list.startswith(os.path.abspath(SAFE_WORKING_DIRECTORY)):
            error_msg = "错误：禁止访问指定路径之外的目录。"
            print(f"--- [本地工具日志] {error_msg} ---")
            return {"items": None, "error": error_msg}

        if not os.path.exists(full_path_to_list) or not os.path.isdir(full_path_to_list):
            error_msg = f"错误：目录 '{directory_path}' (在安全工作区内解析为 '{full_path_to_list}') 不存在或不是一个目录。"
            print(f"--- [本地工具日志] {error_msg} ---")
            return {"items": None, "error": error_msg}
            
        items_details = []
        for item_name in os.listdir(full_path_to_list):
            item_abs_path = os.path.join(full_path_to_list, item_name)
            
            # 计算相对于 SAFE_WORKING_DIRECTORY 的路径
            # 我们需要从 item_abs_path 中移除 SAFE_WORKING_DIRECTORY 前缀
            # 并确保处理好路径分隔符
            relative_path_to_safe_dir = os.path.relpath(item_abs_path, SAFE_WORKING_DIRECTORY)
            # 确保路径分隔符在不同系统上一致性（可选，通常os.path系列函数会处理）
            # relative_path_to_safe_dir = relative_path_to_safe_dir.replace(os.sep, '/')


            item_type = "file" if os.path.isfile(item_abs_path) else "directory" if os.path.isdir(item_abs_path) else "unknown"
            
            items_details.append({
                "name": item_name,
                "type": item_type,
                "relative_path": relative_path_to_safe_dir 
            })
            
        print(f"--- [本地工具日志] 目录项目详情: {items_details} ---")
        return {"items": items_details, "error": None}
    except Exception as e:
        error_msg = f"列出目录项目时发生错误: {str(e)}"
        print(f"--- [本地工具日志] {error_msg} ---")
>>>>>>> 93dff192fe868db871c4399f613347ee41f3e3a1
        return {"items": None, "error": error_msg}