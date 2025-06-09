# docker_sandbox_tool.py
import docker
import tempfile # 仍然可以用它来生成唯一的临时目录名组件，如果需要
import os
import shutil
from typing import Dict, Any, Optional, Tuple # 引入Tuple
import datetime
import json
import traceback # 用于更详细的错误追踪
import requests

# --- 配置常量 ---
DOCKER_IMAGE_NAME = "mlsandbox:latest"
HOST_ROOT_WORKSPACE = os.path.abspath(os.path.join(os.getcwd(), "agent_workspace"))

HOST_DATASETS_MOUNT_SOURCE = os.path.join(HOST_ROOT_WORKSPACE, "datasets")
HOST_OUTPUTS_MOUNT_SOURCE = os.path.join(HOST_ROOT_WORKSPACE, "outputs")
HOST_EXECUTION_LOGS_ROOT = os.path.join(HOST_ROOT_WORKSPACE, "execution_logs") # 持久日志根目录

CONTAINER_DATASETS_MOUNT_TARGET = "/sandbox/datasets"
CONTAINER_OUTPUTS_MOUNT_TARGET = "/sandbox/outputs"
CONTAINER_CODE_EXECUTION_BASE_TARGET = "/sandbox/code"

# --- 辅助函数：确保目录存在 ---
def _ensure_directory_exists(dir_path: str) -> None:
    """确保指定的目录存在，如果不存在则创建它。如果创建失败则抛出异常。"""
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path, exist_ok=True)
            print(f"--- [系统日志] 已创建目录: {dir_path} ---")
        except OSError as e:
            error_msg = f"创建关键目录失败: {dir_path} - {e}"
            print(f"--- [系统日志] {error_msg} ---")
            raise  # 关键目录创建失败，应终止操作

# --- 辅助函数：清理脚本名用于文件名/目录名 ---
def _sanitize_filename(name: str) -> str:
    """清理文件名，移除不安全或不合适的字符，替换为空格或下划线。"""
    # 移除路径分隔符，避免创建意外的子目录结构
    name = name.replace("/", "_").replace("\\", "_")
    # 保留字母、数字、下划线、连字符、点
    sanitized = "".join(c if c.isalnum() or c in ('_', '-', '.') else '_' for c in name)
    # 避免以点或下划线开头（某些系统可能隐藏）
    if sanitized.startswith('.') or sanitized.startswith('_'):
        sanitized = "file_" + sanitized
    # 避免过长文件名 (可选，根据文件系统限制)
    return sanitized[:100] # 限制长度

# --- 辅助函数：写入文件 ---
def _write_to_file(file_path: str, content: str, mode: str = "w", encoding: str = "utf-8") -> None:
    """将内容写入指定文件，包含错误处理。"""
    try:
        with open(file_path, mode, encoding=encoding) as f:
            f.write(content)
        print(f"--- [文件操作] 内容已写入: {file_path} ---")
    except IOError as e:
        print(f"--- [文件操作] 写入文件失败: {file_path} - {e} ---")
        # 根据需求，这里可以决定是否抛出异常或仅记录错误

# --- 主执行函数 ---
def execute_ml_code_in_docker(
    code_string: str,
    script_relative_path: str,
    script_filename: str,
    use_gpu: bool = False,
    timeout_seconds: int = 3600,
    ai_code_description: Optional[str] = "N/A",
    ai_code_purpose: Optional[str] = "N/A"
) -> Dict[str, Any]:
    """
    在隔离的Docker容器中安全地执行由AI指定的Python脚本，并记录详细的执行日志。

    参数:
        code_string (str): AI生成的完整Python代码。
        script_relative_path (str): AI指定的脚本相对于本次执行日志目录的子路径。
                                   例如 "." 或 "preprocessing_scripts"。
        script_filename (str): AI指定的脚本文件名，例如 "main.py"。
        use_gpu (bool): 是否请求GPU资源。
        timeout_seconds (int): 容器执行的超时时间（秒）。
        ai_code_description (str, optional): AI对本次执行代码的简短描述。
        ai_code_purpose (str, optional): AI执行本次代码的目的。

    返回:
        一个字典，包含详细的执行结果和日志信息。
    """
    # 1. 初始化和准备
    _ensure_directory_exists(HOST_DATASETS_MOUNT_SOURCE)
    _ensure_directory_exists(HOST_OUTPUTS_MOUNT_SOURCE)
    _ensure_directory_exists(HOST_EXECUTION_LOGS_ROOT)

    docker_client: Optional[docker.DockerClient] = None
    container: Optional[docker.models.containers.Container] = None
    
    # 生成本次执行的唯一标识和日志目录
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    sanitized_script_base = _sanitize_filename(os.path.splitext(script_filename)[0])
    execution_id = f"{sanitized_script_base}_{timestamp_str}" # 用于日志文件名和目录名
    host_current_execution_log_dir = os.path.join(HOST_EXECUTION_LOGS_ROOT, execution_id)
    _ensure_directory_exists(host_current_execution_log_dir) # 创建本次执行的专属日志目录
    
    print(f"--- [DockerMLSandboxTool] Execution ID: {execution_id} ---")
    print(f"--- [DockerMLSandboxTool] 日志将保存在宿主机: {host_current_execution_log_dir} ---")

    # 初始化返回结果字典，预填充一些信息
    result = {
        "execution_id": execution_id,
        "timestamp_utc_start": datetime.datetime.utcnow().isoformat() + "Z",
        "log_directory_host_path": host_current_execution_log_dir,
        "script_filename_by_ai": script_filename,
        "script_relative_path_by_ai": script_relative_path,
        "ai_code_description": ai_code_description,
        "ai_code_purpose": ai_code_purpose,
        "use_gpu_requested": use_gpu,
        "timeout_seconds_set": timeout_seconds,
        "stdout": "",
        "stderr": "",
        "success": False,
        "exit_code": -1, # 默认错误码
        "executed_script_container_path": None,
        "code_executed_host_path": None,
        "stdout_log_file_host_path": None,
        "stderr_log_file_host_path": None,
        "metadata_file_host_path": None,
        "execution_duration_seconds": None,
        "error_message_preprocessing": None, # 用于记录执行前发生的错误
        "error_message_runtime": None,       # 用于记录容器执行期间或之后发生的错误
    }

    start_time_process = datetime.datetime.now() # 记录整个函数处理的开始时间

    try:
        # 2. 连接Docker客户端
        try:
            docker_client = docker.from_env()
            docker_client.ping() # 验证连接
            print("--- [DockerMLSandboxTool] Docker客户端连接成功 ---")
        except docker.errors.DockerException as e:
            result["error_message_preprocessing"] = f"无法连接到Docker守护进程: {e}"
            print(f"--- [DockerMLSandboxTool] {result['error_message_preprocessing']} ---")
            # 即使连接失败，也尝试保存已有的元数据
            _save_metadata_log(result, host_current_execution_log_dir, execution_id, is_error_state=True)
            return result

        print(f"--- [DockerMLSandboxTool] 准备执行脚本 '{script_relative_path}/{script_filename}' (GPU: {use_gpu}) ---")
        
        # 3. 准备并写入AI代码到宿主机日志目录中的脚本文件
        normalized_script_relative_path = script_relative_path.lstrip('./').lstrip('.\\')
        host_script_target_dir_in_log = os.path.join(host_current_execution_log_dir, normalized_script_relative_path)
        _ensure_directory_exists(host_script_target_dir_in_log)
        
        # 使用包含时间戳和AI指定名称的脚本文件名，确保在日志目录中唯一且可识别
        persistent_script_filename = f"{execution_id}_{_sanitize_filename(script_filename)}"
        host_script_full_path_in_log = os.path.join(host_script_target_dir_in_log, persistent_script_filename)
        result["code_executed_host_path"] = host_script_full_path_in_log
        
        try:
            _write_to_file(host_script_full_path_in_log, code_string)
        except IOError as e:
            result["error_message_preprocessing"] = f"写入持久脚本文件失败: {host_script_full_path_in_log} - {e}"
            _save_metadata_log(result, host_current_execution_log_dir, execution_id, is_error_state=True)
            return result

        # 4. 配置Docker容器运行参数
        volumes = {
            host_current_execution_log_dir: {'bind': CONTAINER_CODE_EXECUTION_BASE_TARGET, 'mode': 'ro'},
            HOST_DATASETS_MOUNT_SOURCE: {'bind': CONTAINER_DATASETS_MOUNT_TARGET, 'mode': 'ro'},
            HOST_OUTPUTS_MOUNT_SOURCE: {'bind': CONTAINER_OUTPUTS_MOUNT_TARGET, 'mode': 'rw'}
        }
        device_requests = [docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])] if use_gpu else None
        
        # 容器内执行的脚本路径是相对于 CONTAINER_CODE_EXECUTION_BASE_TARGET 的
        container_script_execution_path = os.path.join(
            CONTAINER_CODE_EXECUTION_BASE_TARGET,
            normalized_script_relative_path,
            persistent_script_filename # 执行的是带时间戳的持久化脚本名
        ).replace("\\", "/")
        result["executed_script_container_path"] = container_script_execution_path
        container_command = ["python3", container_script_execution_path]
        
        if use_gpu: print("--- [DockerMLSandboxTool] 请求GPU资源 ---")
        print(f"--- [DockerMLSandboxTool] 启动容器 '{DOCKER_IMAGE_NAME}'，执行: '{' '.join(container_command)}' ---")
        
        # 5. 运行容器并捕获输出
        start_time_container = datetime.datetime.now()
        container = docker_client.containers.run(
            DOCKER_IMAGE_NAME, command=container_command, volumes=volumes,
            device_requests=device_requests, detach=True,
        )
        print(f"--- [DockerMLSandboxTool] 容器 '{container.short_id}' 已启动 ---")
        
        try:
            container_wait_result = container.wait(timeout=timeout_seconds)
            result["exit_code"] = container_wait_result.get("StatusCode", -1)
        except (docker.errors.APIError, requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            result["error_message_runtime"] = f"等待容器执行超时或连接错误: {e}"
            result["exit_code"] = -1 # 标记为错误退出
            print(f"--- [DockerMLSandboxTool] {result['error_message_runtime']} ---")
            # 尝试获取部分日志
            try:
                result["stdout"] = container.logs(stdout=True, stderr=False).decode('utf-8', errors='replace')
                partial_stderr = container.logs(stdout=False, stderr=True).decode('utf-8', errors='replace')
                if partial_stderr:
                    result["stderr"] = (result.get("stderr", "") + "\nPartial Stderr from container:\n" + partial_stderr).strip()
            except Exception as log_e:
                print(f"--- [DockerMLSandboxTool] 获取超时容器日志失败: {log_e} ---")
        
        end_time_container = datetime.datetime.now()
        result["execution_duration_seconds"] = round((end_time_container - start_time_container).total_seconds(), 3)
        result["success"] = (result["exit_code"] == 0)

        # 确保即使在超时后也尝试获取完整的stdout/stderr
        if not result["stdout"]:
            try: result["stdout"] = container.logs(stdout=True, stderr=False).decode('utf-8', errors='replace')
            except: pass
        if not result["stderr"] and not result["success"]: # 如果stderr为空且执行失败
            try: result["stderr"] = container.logs(stdout=False, stderr=True).decode('utf-8', errors='replace')
            except: pass

        print(f"--- [DockerMLSandboxTool] 容器 '{container.short_id}' 执行完毕，退出码: {result['exit_code']}, 耗时: {result['execution_duration_seconds']:.2f}s ---")
        if result["stdout"]: print(f"--- [DockerMLSandboxTool] Stdout (前500字符):\n{result['stdout'][:500]}... ---")
        if result["stderr"]: print(f"--- [DockerMLSandboxTool] Stderr (前500字符):\n{result['stderr'][:500]}... ---")

    except docker.errors.ImageNotFound as e:
        result["error_message_preprocessing"] = f"Docker镜像 '{DOCKER_IMAGE_NAME}' 未找到: {e}"
    except docker.errors.APIError as e:
        result["error_message_preprocessing"] = f"Docker API错误 (执行前): {e}"
    except Exception as e:
        result["error_message_preprocessing"] = f"执行Docker容器的准备阶段发生意外错误: {e}"
        result["stderr"] = (result.get("stderr", "") + "\nTraceback (Pre-execution):\n" + traceback.format_exc()).strip()
    finally:
        if container:
            try:
                container.remove(force=True)
                print(f"--- [DockerMLSandboxTool] 容器 '{container.short_id if hasattr(container, 'short_id') else 'unknown'}' 已移除 ---")
            except Exception as e:
                print(f"--- [DockerMLSandboxTool] 移除容器失败: {e} ---")

        # 6. 保存stdout, stderr到日志文件 (无论成功与否，只要有内容)
        result["stdout_log_file_host_path"] = os.path.join(host_current_execution_log_dir, f"{execution_id}.stdout.log")
        _write_to_file(result["stdout_log_file_host_path"], result["stdout"])
        
        result["stderr_log_file_host_path"] = os.path.join(host_current_execution_log_dir, f"{execution_id}.stderr.log")
        _write_to_file(result["stderr_log_file_host_path"], result["stderr"])

        # 7. 保存最终的元数据
        result["timestamp_utc_end_process"] = datetime.datetime.utcnow().isoformat() + "Z"
        result["total_tool_duration_seconds"] = round((datetime.datetime.now() - start_time_process).total_seconds(), 3)
        _save_metadata_log(result, host_current_execution_log_dir, execution_id)
        
        print(f"--- [DockerMLSandboxTool] 执行流程完毕 for Execution ID: {execution_id} ---")
        return result

def _save_metadata_log(metadata_dict: Dict[str, Any], log_dir: str, exec_id: str, is_error_state: bool = False) -> None:
    """辅助函数，用于保存元数据JSON文件。"""
    filename_suffix = "_ERROR" if is_error_state and not metadata_dict.get("success", True) else ""
    meta_log_filename = f"{exec_id}{filename_suffix}.meta.json"
    meta_log_path = os.path.join(log_dir, meta_log_filename)
    metadata_dict["metadata_file_host_path"] = meta_log_path # 更新元数据字典中的路径

    try:
        # 确保所有值都是可JSON序列化的 (例如datetime对象转换为字符串)
        serializable_metadata = {}
        for key, value in metadata_dict.items():
            if isinstance(value, datetime.datetime):
                serializable_metadata[key] = value.isoformat()
            else:
                serializable_metadata[key] = value
        
        with open(meta_log_path, "w", encoding="utf-8") as f_meta:
            json.dump(serializable_metadata, f_meta, indent=4, ensure_ascii=False)
        print(f"--- [元数据日志] 元数据已保存到: {meta_log_path} ---")
    except Exception as e:
        print(f"--- [元数据日志] 保存元数据日志失败: {meta_log_path} - {e} ---")


# --- 测试用例 (与之前类似，但现在会生成更详细的日志结构) ---
if __name__ == "__main__":
    print("--- 开始测试 execute_ml_code_in_docker (带持久化日志) ---")

    # 确保执行日志根目录存在，以便测试可以写入
    _ensure_directory_exists(HOST_EXECUTION_LOGS_ROOT)

    # 1. 简单打印测试
    print("\n--- 测试1: 简单打印 ---")
    code1 = "print('Hello from Docker Sandbox with persistent logging!')\nimport pandas as pd\nprint(f'Pandas version: {pd.__version__}')"
    result1 = execute_ml_code_in_docker(
        code_string=code1,
        script_relative_path="general_tests",
        script_filename="hello_persistent.py",
        ai_code_description="A simple test to print hello and pandas version.",
        ai_code_purpose="Verify basic execution and logging."
    )
    print(f"测试1结果: {json.dumps(result1, indent=2)}")
    assert result1["success"] is True
    assert os.path.exists(result1["log_directory_host_path"])
    assert os.path.exists(result1["code_executed_host_path"])
    assert os.path.exists(result1["stdout_log_file_host_path"])
    assert os.path.exists(result1["metadata_file_host_path"])

    # 2. 创建文件到outputs目录测试
    print("\n--- 测试2: 写入文件到 /sandbox/outputs/ ---")
    output_file_container_rel_path = "test_output_persistent/agent_generated_file.txt" # AI代码内使用的相对路径
    output_file_container_abs_path = os.path.join(CONTAINER_OUTPUTS_MOUNT_TARGET, output_file_container_rel_path)
    host_expected_output_file = os.path.join(HOST_OUTPUTS_MOUNT_SOURCE, output_file_container_rel_path)
    
    if os.path.exists(host_expected_output_file): os.remove(host_expected_output_file) # 清理

    code2 = f"""
import os
output_path = "{output_file_container_abs_path}"
output_dir = os.path.dirname(output_path)
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
with open(output_path, "w") as f:
    f.write("Persistent log test: File generated by AI.")
print(f"File '{{output_path}}' created.")
"""
    result2 = execute_ml_code_in_docker(
        code_string=code2,
        script_relative_path="file_io_tests",
        script_filename="create_output_file.py",
        ai_code_description="Tests writing a file to the mapped outputs directory.",
        ai_code_purpose="Verify output volume mount and file creation."
    )
    print(f"测试2结果: {json.dumps(result2, indent=2)}")
    assert result2["success"] is True
    assert os.path.exists(host_expected_output_file)
    with open(host_expected_output_file, "r") as f: assert "Persistent log test" in f.read()
    print(f"测试2成功: 文件 {host_expected_output_file} 已验证.")

    # 3. 错误代码测试
    print("\n--- 测试3: 错误代码 ---")
    code3 = "print('Intentional error incoming...')\nx = 1 / 0"
    result3 = execute_ml_code_in_docker(
        code_string=code3,
        script_relative_path="error_tests",
        script_filename="division_error.py",
        ai_code_description="Tests runtime error handling.",
        ai_code_purpose="Verify stderr logging and success=False."
    )
    print(f"测试3结果: {json.dumps(result3, indent=2)}")
    assert result3["success"] is False
    assert "ZeroDivisionError" in result3["stderr"]
    assert os.path.exists(result3["stderr_log_file_host_path"])
    with open(result3["stderr_log_file_host_path"], "r") as f_err_check:
        assert "ZeroDivisionError" in f_err_check.read()

    # 4. 模拟执行前错误 (例如Docker连接失败) - 这个较难直接在测试脚本中模拟
    # 但我们可以在函数内部的docker.from_env()处制造错误来测试_save_metadata_log的is_error_state
    # 这里我们只测试正常流程。

    print("\n--- 所有选定测试执行完毕 ---")
    print(f"--- 请检查宿主机目录 '{HOST_EXECUTION_LOGS_ROOT}' 下生成的日志文件和目录 ---")