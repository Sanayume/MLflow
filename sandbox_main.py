import docker
import tempfile
import os
import shutil
from typing import Dict, Any, Optional
import datetime
import json
import traceback
import requests
from enum import Enum # 引入Enum来定义结构化的错误类型
from config import (
    DOCKER_IMAGE_NAME,
    HOST_ROOT_WORKSPACE,
    HOST_DATASETS_DIR_NAME,
    HOST_OUTPUTS_DIR_NAME,
    HOST_EXECUTION_LOGS_DIR_NAME,
    HOST_DATASETS_MOUNT_SOURCE,
    HOST_OUTPUTS_MOUNT_SOURCE,
    HOST_EXECUTION_LOGS_ROOT,
    CONTAINER_DATASETS_MOUNT_TARGET,
    CONTAINER_OUTPUTS_MOUNT_TARGET,
    CONTAINER_CODE_EXECUTION_BASE_TARGET,
    DB_CONFIG,
    ExecutionErrorType
)
from db_utils import log_execution_to_database
import logging
logger = logging.getLogger(__name__)

#更鲁棒的错误
class ErrorType(Enum):
    """定义了沙箱执行过程中可能发生的、结构化的错误类型。"""
    PREPARATION_ERROR = "PREPARATION_ERROR"     # 容器启动前发生的错误 (如Docker连接、镜像查找、文件写入)
    RUNTIME_ERROR = "RUNTIME_ERROR"             # 容器内Python脚本执行时抛出异常 (由非零退出码判断)
    TIMEOUT_ERROR = "TIMEOUT_ERROR"             # 容器执行时间超过设定的超时限制
    DOCKER_API_ERROR = "DOCKER_API_ERROR"           # 与Docker守护进程交互时发生的其他运行时API错误

CUDA_INFO_MARKER = "https://docs.nvidia.com/datacenter/cloud-native/"

from pydantic import BaseModel, Field
from typing import Optional

class SandboxExecutionInput(BaseModel):
    code_string: str = Field(
        description="要在一个安全的、预装机器学习库的Docker容器中执行的完整Python代码字符串。代码中所有文件路径都必须使用Linux风格的正斜杠 '/'。"
    )
    script_relative_path: str = Field(
        default=".",
        description="你希望将这段代码保存为脚本文件时，脚本文件相对于本次执行的代码区根目录 (`/sandbox/code/`) 的相对路径。例如 '.' (直接在根下), 'preprocessing', 或 'feature_engineering/step1'。请使用简单、有效的目录名，不要使用 '..' 或绝对路径。"
    )
    script_filename: str = Field(
        description="你为生成的Python脚本指定的文件名，必须以 '.py' 结尾。例如 'train_model.py' 或 'data_analysis.py'。"
    )
    ai_code_description: str = Field(
        default="N/A",
        description="对你将要执行的这段Python代码的简短文字描述（例如“加载数据并进行初步清洗”）。"
    )
    ai_code_purpose: str = Field(
        default="N/A",
        description="执行这段Python代码的主要目的或预期达成的目标（例如“为后续模型训练准备特征”）。"
    )
    use_gpu: bool = Field(
        default=False,
        description="如果你的代码需要GPU加速并且宿主机有可用GPU，请将此设为 True。默认为 False (使用CPU)。"
    )
    # --- 新增的字段 ---
    cpu_core_limit: Optional[float] = Field(
        default=None,
        description="(未来功能) 可选参数，用于指定限制容器可以使用的CPU核心数量，例如 1.5。目前仅作为日志记录，暂不生效。"
    )
    memory_limit: Optional[str] = Field(
        default=None,
        description="(未来功能) 可选参数，用于指定限制容器的内存使用量，例如 '512m' 或 '2g'。目前仅作为日志记录，暂不生效。"
    )

# --- 辅助函数 (保持不变) ---
def _ensure_directory_exists(dir_path: str) -> None:
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path, exist_ok=True)
            print(f"--- [系统日志] 已创建目录: {dir_path} ---")
        except OSError as e:
            raise IOError(f"创建关键目录失败: {dir_path} - {e}")

def _sanitize_filename(name: str) -> str:
    name = name.replace("/", "_").replace("\\", "_")
    sanitized = "".join(c if c.isalnum() or c in ('_', '-', '.') else '_' for c in name)
    if sanitized.startswith('.') or sanitized.startswith('_'):
        sanitized = "file_" + sanitized
    return sanitized[:100]

def _write_to_file(file_path: str, content: str, mode: str = "w", encoding: str = "utf-8") -> None:
    try:
        with open(file_path, mode, encoding=encoding) as f:
            f.write(content)
        print(f"--- [文件操作] 内容已写入: {file_path} ---")
    except IOError as e:
        raise # 写入失败是关键错误，直接抛出

# --- 主执行函数 (优化) ---
def execute_ml_code_in_docker(
    code_string: str,
    script_relative_path: str,
    script_filename: str,
    use_gpu: bool = False,
    timeout_seconds: int = 3600,
    ai_code_description: Optional[str] = "N/A",
    ai_code_purpose: Optional[str] = "N/A",
    # --- 新增: 为资源限制预留的接口 ---
    cpu_core_limit: Optional[float] = None,
    memory_limit: Optional[str] = None,
) -> Dict[str, Any]:
    """
    在隔离的Docker容器中安全地执行由AI指定的Python脚本，并记录详细的执行日志。

    新增参数 (为未来功能预留接口):
        cpu_core_limit (float, optional): (未来实现) 限制容器可用的CPU核心数，例如 1.5。
        memory_limit (str, optional): (未来实现) 限制容器的内存使用，例如 "512m" 或 "2g"。
    """
    # 1. 初始化和准备
    try:
        _ensure_directory_exists(HOST_DATASETS_MOUNT_SOURCE)
        _ensure_directory_exists(HOST_OUTPUTS_MOUNT_SOURCE)
        _ensure_directory_exists(HOST_EXECUTION_LOGS_ROOT)
    except IOError as e:
        # 如果连最基础的目录都无法创建，直接返回一个准备阶段错误的字典
        return {
            "execution_id": f"PREP_FAIL_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            "success": False, "exit_code": -1, "error_type": ErrorType.PREPARATION_ERROR.value,
            "error_message_preprocessing": str(e),
            # 其他字段使用默认值或None
            **{k: None for k in ["stdout", "stderr", "log_directory_host_path"]} 
        }

    docker_client: Optional[docker.DockerClient] = None
    container: Optional[docker.models.containers.Container] = None
    
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    sanitized_script_base = _sanitize_filename(os.path.splitext(script_filename)[0])
    execution_id = f"{sanitized_script_base}_{timestamp_str}"
    host_current_execution_log_dir = os.path.join(HOST_EXECUTION_LOGS_ROOT, execution_id)
    _ensure_directory_exists(host_current_execution_log_dir)
    
    print(f"--- [DockerMLSandboxTool] Execution ID: {execution_id} ---")
    print(f"--- [DockerMLSandboxTool] 日志将保存在宿主机: {host_current_execution_log_dir} ---")

    # 初始化返回结果字典
    result = {
        "execution_id": execution_id,
        "timestamp_utc_start": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "log_directory_host_path": host_current_execution_log_dir,
        "script_filename_by_ai": script_filename,
        "script_relative_path_by_ai": script_relative_path,
        "ai_code_description": ai_code_description,
        "ai_code_purpose": ai_code_purpose,
        "use_gpu_requested": use_gpu,
        "timeout_seconds_set": timeout_seconds,
        "cpu_core_limit_set": cpu_core_limit, # 记录请求的资源限制
        "memory_limit_set": memory_limit,    # 记录请求的资源限制
        "stdout": "", "stderr": "", "success": False, "exit_code": -1,
        "executed_script_container_path": None, "code_executed_host_path": None,
        "stdout_log_file_host_path": None, "stderr_log_file_host_path": None,
        "metadata_file_host_path": None, "execution_duration_seconds": None,
        "error_message_preprocessing": None, "error_message_runtime": None,
        "error_type": None, # --- 新增: 错误类型字段，默认为None ---
    }
    start_time_process = datetime.datetime.now()

    try:
        # 2. 连接Docker客户端及准备阶段
        try:
            docker_client = docker.from_env()
            docker_client.ping()
            print("--- [DockerMLSandboxTool] Docker客户端连接成功 ---")
            
            normalized_script_relative_path = script_relative_path.lstrip('./').lstrip('.\\')
            host_script_target_dir_in_log = os.path.join(host_current_execution_log_dir, normalized_script_relative_path)
            _ensure_directory_exists(host_script_target_dir_in_log)
            
            persistent_script_filename = f"{execution_id}.py"
            host_script_full_path_in_log = os.path.join(host_script_target_dir_in_log, persistent_script_filename)
            result["code_executed_host_path"] = host_script_full_path_in_log
            
            _write_to_file(host_script_full_path_in_log, code_string)

        except (docker.errors.DockerException, docker.errors.ImageNotFound, IOError) as e:
            result["error_message_preprocessing"] = f"容器准备阶段失败: {e}"
            result["error_type"] = ErrorType.PREPARATION_ERROR.value # 设置错误类型
            print(f"--- [DockerMLSandboxTool] {result['error_message_preprocessing']} ---")
            # 准备阶段失败，直接跳到finally块保存日志并返回
            raise

        # 4. 配置Docker容器运行参数
        volumes = {
            host_current_execution_log_dir: {'bind': CONTAINER_CODE_EXECUTION_BASE_TARGET, 'mode': 'ro'},
            HOST_DATASETS_MOUNT_SOURCE: {'bind': CONTAINER_DATASETS_MOUNT_TARGET, 'mode': 'ro'},
            HOST_OUTPUTS_MOUNT_SOURCE: {'bind': CONTAINER_OUTPUTS_MOUNT_TARGET, 'mode': 'rw'}
        }
        device_requests = [docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])] if use_gpu else None
        
        container_script_execution_path = os.path.join(
            CONTAINER_CODE_EXECUTION_BASE_TARGET, normalized_script_relative_path, persistent_script_filename
        ).replace("\\", "/")
        result["executed_script_container_path"] = container_script_execution_path
        container_command = ["python3", container_script_execution_path]

        # (未来实现) 资源限制参数逻辑
        container_run_kwargs = {
            "command": container_command, "volumes": volumes, "device_requests": device_requests, "detach": True,
            # 'nano_cpus': int(cpu_core_limit * 1e9) if cpu_core_limit else None, # <-- 预留接口
            # 'mem_limit': memory_limit,                                         # <-- 预留接口
        }
        
        if use_gpu: print("--- [DockerMLSandboxTool] 请求GPU资源 ---")
        print(f"--- [DockerMLSandboxTool] 启动容器 '{DOCKER_IMAGE_NAME}'，执行: '{' '.join(container_command)}' ---")
        
        # 5. 运行容器并捕获输出
        start_time_container = datetime.datetime.now()
        container = docker_client.containers.run(DOCKER_IMAGE_NAME, **container_run_kwargs)
        print(f"--- [DockerMLSandboxTool] 容器 '{container.short_id}' 已启动 ---")
        
        try:
            container_wait_result = container.wait(timeout=timeout_seconds)
            result["exit_code"] = container_wait_result.get("StatusCode", -1)
        
        # 捕获所有可能的Docker相关运行时错误
        except (docker.errors.APIError, requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            error_string = str(e).lower() # 将错误信息转为小写以便不区分大小写地搜索
            
            # 检查错误字符串中是否包含超时的关键词
            if "read timed out" in error_string or "timeout" in error_string:
                result["error_message_runtime"] = f"容器执行超时 (超过 {timeout_seconds} 秒)"
                result["error_type"] = ErrorType.TIMEOUT_ERROR.value
                result["exit_code"] = 137 # 模拟超时退出码
                print(f"--- [DockerMLSandboxTool] {result['error_message_runtime']}: {e} ---")
            else:
                # 如果不是超时，那就是其他API或连接错误
                result["error_message_runtime"] = f"等待容器时发生Docker API或连接错误: {e}"
                result["error_type"] = ErrorType.DOCKER_API_ERROR.value
                result["exit_code"] = -1
                print(f"--- [DockerMLSandboxTool] {result['error_message_runtime']} ---")
        end_time_container = datetime.datetime.now()
        result["execution_duration_seconds"] = round((end_time_container - start_time_container).total_seconds(), 3)
        result["success"] = (result["exit_code"] == 0)

        # 如果执行失败且没有更具体的错误类型，则归类为运行时错误
        if not result["success"] and result["error_type"] is None:
            result["error_type"] = ErrorType.RUNTIME_ERROR.value

        # 确保即使在超时后也尝试获取完整的stdout/stderr
        try:
            result["stdout"] = container.logs(stdout=True, stderr=False).decode('utf-8', errors='replace')
            if not result["success"]:
                 result["stderr"] = container.logs(stdout=False, stderr=True).decode('utf-8', errors='replace')
        except Exception as log_e:
             print(f"--- [DockerMLSandboxTool] 获取容器日志时发生次要错误: {log_e} ---")

        print(f"--- [DockerMLSandboxTool] 容器 '{container.short_id}' 执行完毕，退出码: {result['exit_code']}, 耗时: {result['execution_duration_seconds']:.2f}s ---")

    except Exception as e:
        # 捕获准备阶段抛出的所有异常
        if result["error_type"] is None: # 避免覆盖已设置的错误类型
            result["error_type"] = ErrorType.PREPARATION_ERROR.value
            result["error_message_preprocessing"] = f"执行Docker容器的准备阶段发生意外错误: {e}"
            result["stderr"] = (result.get("stderr", "") + "\nTraceback (Pre-execution):\n" + traceback.format_exc()).strip()
        print(f"--- [DockerMLSandboxTool] 捕获到准备阶段异常，将终止执行并记录日志 ---")

    finally:
        if container:
            try:
                container.remove(force=True)
                print(f"--- [DockerMLSandboxTool] 容器 '{container.short_id if hasattr(container, 'short_id') else 'unknown'}' 已移除 ---")
            except Exception as e_rm:
                print(f"--- [DockerMLSandboxTool] 移除容器失败: {e_rm} ---")

        # 6. 保存stdout, stderr到日志文件
        if CUDA_INFO_MARKER in result["stdout"]:
            index = result["stdout"].find(CUDA_INFO_MARKER)
            if index != -1:
                result["stdout"] = result["stdout"][index + 2 + len(CUDA_INFO_MARKER):].strip().lstrip("=").strip()
        
        result["stdout_log_file_host_path"] = os.path.join(host_current_execution_log_dir, f"{execution_id}.stdout.log")
        _write_to_file(result["stdout_log_file_host_path"], result["stdout"])
        
        result["stderr_log_file_host_path"] = os.path.join(host_current_execution_log_dir, f"{execution_id}.stderr.log")
        _write_to_file(result["stderr_log_file_host_path"], result["stderr"])

        # 7. 保存最终的元数据
        result["timestamp_utc_end_process"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        result["total_tool_duration_seconds"] = round((datetime.datetime.now() - start_time_process).total_seconds(), 3)
        _save_metadata_log(result, host_current_execution_log_dir, execution_id)
        
    # 8. 将执行元数据记录到数据库
        try:
            logger.info(f"Attempting to log execution {execution_id} to database.")
            db_record_id = log_execution_to_database(result)
            if db_record_id:
                result["database_record_id"] = db_record_id
                logger.info(f"Successfully logged execution {execution_id} to database with DB ID: {db_record_id}.")
            else:
                logger.warning(f"Failed to log execution {execution_id} to database (or DB logging disabled/failed).")
        except Exception as db_e: # Catch any unexpected error from DB logging
            logger.error(f"Unexpected error during database logging for {execution_id}: {db_e}", exc_info=True)
            if result["error_type"] is None: result["error_type"] = ExecutionErrorType.LOGGING_FAILURE.value
            result["error_message_runtime"] = (result.get("error_message_runtime") or "") + f"; DB logging error: {db_e}"

        logger.info(f"Docker execution process finished for ID: {execution_id}")
        return result

def _save_metadata_log(metadata_dict: Dict[str, Any], log_dir: str, exec_id: str) -> None:
    # 简化is_error_state的判断
    is_error = not metadata_dict.get("success", True)
    filename_suffix = "_ERROR" if is_error else ""
    meta_log_filename = f"{exec_id}{filename_suffix}.meta.json"
    meta_log_path = os.path.join(log_dir, meta_log_filename)
    metadata_dict["metadata_file_host_path"] = meta_log_path

    try:
        serializable_metadata = {}
        for key, value in metadata_dict.items():
            if isinstance(value, datetime.datetime):
                serializable_metadata[key] = value.isoformat()
            elif isinstance(value, Enum): # --- 新增: 序列化Enum为字符串 ---
                serializable_metadata[key] = value.value
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
output_path_in_container = "{CONTAINER_OUTPUTS_MOUNT_TARGET}/{output_file_container_rel_path}" # 直接拼接，确保正斜杠
output_dir_in_container = os.path.dirname(output_path_in_container)
print(f"Attempting to create directory: {{output_dir_in_container}}")
if not os.path.exists(output_dir_in_container):
    os.makedirs(output_dir_in_container, exist_ok=True)
    print(f"Directory {{output_dir_in_container}} created or already exists.")
else:
    print(f"Directory {{output_dir_in_container}} already exists.")
    print(f"Attempting to write file: {{output_path_in_container}}")
try:
    with open(output_path_in_container, "w") as f:
        f.write("Persistent log test: File generated by AI.")
    print(f"File '{{output_path_in_container}}' created successfully.")
except Exception as e:
    print(f"Error writing file {{output_path_in_container}}: {{e}}")
import traceback
print(traceback.format_exc())
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

    # 测试4: 错误代码，验证 RUNTIME_ERROR
    print("\n--- 测试4: 错误代码 (RUNTIME_ERROR) ---")
    code4 = "print('Intentional error incoming...')\nx = 1 / 0"
    result4 = execute_ml_code_in_docker(
        code_string=code4, script_relative_path="error_tests", script_filename="division_error.py"
    )
    print(f"测试4结果: {json.dumps(result4, indent=2)}")
    assert result4["success"] is False
    assert result4["error_type"] == ErrorType.RUNTIME_ERROR.value
    assert "ZeroDivisionError" in result4["stderr"]

    # 测试5: 验证资源限制参数的日志记录
    print("\n--- 测试5: 验证资源限制参数日志 ---")
    code5 = "print('Testing resource limit logging.')"
    result5 = execute_ml_code_in_docker(
        code_string=code5,
        script_relative_path="config_tests",
        script_filename="resource_test.py",
        cpu_core_limit=2.5, # 传入新参数
        memory_limit="4g"   # 传入新参数
    )
    print(f"测试5结果: {json.dumps(result5, indent=2)}")
    assert result5["success"] is True
    assert result5["cpu_core_limit_set"] == 2.5
    assert result5["memory_limit_set"] == "4g"
    assert result5["error_type"] is None
    print("测试5成功: 资源限制参数已正确记录。")

    # 测试6: 模拟超时 (TIMEOUT_ERROR)
    print("\n--- 测试6: 模拟超时 (TIMEOUT_ERROR) ---")
    code6 = "import time\nprint('Starting long task...')\ntime.sleep(5)\nprint('This should not be printed.')"
    result6 = execute_ml_code_in_docker(
        code_string=code6,
        script_relative_path="timeout_tests",
        script_filename="sleepy.py",
        timeout_seconds=2 # 设置一个很短的超时
    )
    print(f"测试6结果: {json.dumps(result6, indent=2)}")
    assert result6["success"] is False
    assert result6["error_type"] == ErrorType.TIMEOUT_ERROR.value
    assert "超时" in result6["error_message_runtime"]
    print("测试6成功: 超时错误已正确捕获。")

    # (模拟PREPARATION_ERROR较困难，但我们的代码逻辑已经覆盖了它)

    print("\n--- 所有选定测试执行完毕 ---")

    # 基本日志设置，用于直接测试脚本
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    
    logger.info("--- 启动 execute_ml_code_in_docker 的测试套件（鲁棒版本） ---")

    # 测试7：简单打印
    logger.info("\n--- 测试7：简单打印 ---")
    # ...（其余测试用例，类似之前，但现在将使用logger）
    # ... 你将调用 execute_ml_code_in_docker 并根据新的 'result' 结构进行断言，
    # ... 特别是检查 'error_type' 和 'error_message_*' 字段。
    # 测试1的示例：
    code7 = "print('Hello from Robust Docker Sandbox!')\nimport pandas as pd\nprint(f'Pandas version: {pd.__version__}')"
    result7 = execute_ml_code_in_docker(
        code_string=code7, script_relative_path="general_robust", script_filename="hello_robust.py",
        ai_code_description="鲁棒测试：打印hello和pandas版本。",
        ai_code_purpose="验证基本执行、日志记录和新的错误处理。"
    )
    logger.info(f"测试7结果（JSON）：\n{json.dumps(result7, indent=2, default=str)}") # default=str 用于 Enum
    assert result7["success"] is True
    assert result7["error_type"] is None
    assert os.path.exists(result7["log_directory_host_path"])
    assert "Hello from Robust Docker Sandbox!" in result7["stdout"]

    # 测试8：运行时错误
    logger.info("\n--- 测试8：运行时错误 ---")
    code8 = "print('Runtime error incoming...')\nvalue = 1/0"
    result8 = execute_ml_code_in_docker(
        code_string=code8, script_relative_path="errors_robust", script_filename="runtime_err.py"
    )
    logger.info(f"测试8结果（JSON）：\n{json.dumps(result8, indent=2, default=str)}")
    assert result8["success"] is False
    assert result8["error_type"] == ExecutionErrorType.RUNTIME_ERROR.value # 检查特定错误类型
    assert "ZeroDivisionError" in result8["stderr"]

    logger.info("\n--- 所有鲁棒测试已完成。请检查 agent_workspace/execution_logs 中的日志 ---")