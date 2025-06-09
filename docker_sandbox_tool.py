# docker_sandbox_tool.py

import docker # 用于与Docker引擎交互
import tempfile # 用于创建临时文件和目录
import os # 用于路径操作
import shutil # 用于删除目录树（清理临时文件）
from typing import Dict, Any, Optional

import requests


# tool_models.py (或者放在 docker_sandbox_tool.py 顶部)
from langchain_core.pydantic_v1 import BaseModel, Field
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



# --- Docker和宿主机工作目录配置 ---
# 这个是你在 `docker build -t mlsandbox:latest ...` 时给镜像起的名字和标签
DOCKER_IMAGE_NAME = "mlsandbox:latest" 

# 定义宿主机上AI Agent的工作区根目录
# 这是在 app.py 所在的目录（当前工作目录）下创建一个名为 "agent_workspace" 的子目录
HOST_ROOT_WORKSPACE = os.path.abspath(os.path.join(os.getcwd(), "agent_workspace"))

# 在 HOST_ROOT_WORKSPACE 下定义用于Docker卷挂载的特定子目录
# 这些路径是宿主机上的路径
HOST_DATASETS_PATH = os.path.join(HOST_ROOT_WORKSPACE, "datasets")
HOST_OUTPUTS_PATH = os.path.join(HOST_ROOT_WORKSPACE, "outputs")
HOST_TEMP_CODE_BASE_PATH = os.path.join(HOST_ROOT_WORKSPACE, "temp_code") # 存放临时脚本的基目录

# 确保这些宿主机目录存在，如果不存在则创建它们
# 这段代码会在模块加载时执行一次
for p in [HOST_DATASETS_PATH, HOST_OUTPUTS_PATH, HOST_TEMP_CODE_BASE_PATH]:
    if not os.path.exists(p):
        try:
            os.makedirs(p, exist_ok=True) # exist_ok=True 表示如果目录已存在则不报错
            print(f"--- [系统初始化] 已创建目录: {p} ---")
        except OSError as e:
            print(f"--- [系统初始化] 创建目录失败: {p} - {e} ---")
            # 在实际应用中，这里可能需要更健壮的错误处理或直接抛出异常
            # 但对于初次运行，打印错误并继续可能有助于调试权限等问题


def execute_ml_code_in_docker(
    code_string: str,       # AI生成的Python代码字符串
    use_gpu: bool = False   # 是否请求GPU资源
) -> Dict[str, Any]:        # 返回一个包含执行结果的字典
    """
    在隔离的Docker容器 (mlsandbox:latest) 中安全地执行机器学习相关的Python代码。

    参数:
        code_string (str): 要执行的Python代码。
        use_gpu (bool): 是否尝试在有GPU的环境中执行代码。

    返回:
        dict: 包含以下键的字典:
            'stdout': str, 代码的标准输出。
            'stderr': str, 代码的标准错误输出。
            'success': bool, 代码是否成功执行 (基于退出码)。
            'exit_code': int, 容器中Python脚本的退出码。
            'error_message': Optional[str], 如果在Docker操作或执行中发生高级别错误。
    """
    
    # 初始化Docker客户端，它会自动尝试连接到本地运行的Docker引擎
    try:
        client = docker.from_env()
        # 尝试ping一下Docker引擎，确保连接正常
        client.ping()
        print("--- [DockerMLSandboxTool] Docker客户端初始化并连接成功 ---")
    except docker.errors.DockerException as e:
        error_msg = f"无法连接到Docker引擎，请确保Docker正在运行并且当前用户有权限访问: {e}"
        print(f"--- [DockerMLSandboxTool] {error_msg} ---")
        return {"stdout": None, "stderr": None, "success": False, "exit_code": -1, "error_message": error_msg}

    # 初始化返回值
    result_dict = {
        "stdout": None,
        "stderr": None,
        "success": False,
        "exit_code": -1,
        "error_message": None
    }

    container = None  # 初始化容器变量，以便在finally中可以检查
    temp_script_dir_on_host_for_this_run = None # 宿主机上为本次运行创建的唯一临时代码目录

    try:
        print(f"--- [DockerMLSandboxTool] 准备执行代码 (GPU请求: {use_gpu}) ---\n代码片段 (前300字符):\n{code_string[:300]}...\n---")

        # 1. 在宿主机的 HOST_TEMP_CODE_BASE_PATH 下为本次执行创建一个唯一的临时子目录
        #    这样做的好处是，如果同时有多个执行请求（虽然Streamlit是单线程，但未来可能扩展），它们不会互相干扰。
        #    并且清理时也更方便，直接删除这个子目录。
        temp_script_dir_on_host_for_this_run = tempfile.mkdtemp(prefix="run_", dir=HOST_TEMP_CODE_BASE_PATH)
        script_filename = "agent_script.py" # 脚本在容器内的名称不重要，重要的是宿主机上的路径
        host_script_path = os.path.join(temp_script_dir_on_host_for_this_run, script_filename)
        
        # 将AI生成的代码写入这个宿主机上的临时脚本文件
        with open(host_script_path, "w", encoding="utf-8") as f:
            f.write(code_string)
        print(f"--- [DockerMLSandboxTool] AI代码已写入临时脚本: {host_script_path} ---")

        # 2. 定义Docker卷挂载 (将宿主机目录映射到容器内目录)
        #    这些容器内的路径 (`/sandbox/...`) 需要在Agent的System Prompt中告知LLM
        volumes = {
            # 将包含本次执行脚本的临时目录挂载到容器的/sandbox/code (只读)
            temp_script_dir_on_host_for_this_run: {'bind': '/sandbox/code', 'mode': 'ro'},
            # 将宿主机的数据集目录挂载到容器的/sandbox/datasets (只读)
            HOST_DATASETS_PATH: {'bind': '/sandbox/datasets', 'mode': 'ro'},
            # 将宿主机的输出目录挂载到容器的/sandbox/outputs (可读写)
            HOST_OUTPUTS_PATH: {'bind': '/sandbox/outputs', 'mode': 'rw'}
        }
        print(f"--- [DockerMLSandboxTool] 配置的卷挂载: {volumes} ---")
        
        # 3. 配置GPU请求 (如果use_gpu为True)
        device_requests = None
        if use_gpu:
            try:
                # 尝试请求所有可用的GPU，这需要NVIDIA Container Toolkit在Docker中正确配置
                device_requests = [docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])]
                print("--- [DockerMLSandboxTool] 已配置GPU资源请求 ---")
            except Exception as gpu_err:
                # 如果docker.types.DeviceRequest不可用或配置失败，记录错误但继续尝试CPU执行
                print(f"--- [DockerMLSandboxTool] 配置GPU时发生警告/错误 (将尝试CPU执行): {gpu_err} ---")
                # result_dict["error_message"] = f"GPU配置警告: {gpu_err}" # 可以选择是否将此作为错误返回
        
        # 4. 定义在容器内要执行的命令
        #    我们要在容器内执行刚刚写入的脚本
        container_command = ["python3", f"/sandbox/code/{script_filename}"]
        
        print(f"--- [DockerMLSandboxTool] 准备启动容器 '{DOCKER_IMAGE_NAME}'...")
        print(f"--- [DockerMLSandboxTool] 容器内执行命令: '{' '.join(container_command)}' ---")
        
        # 5. 运行Docker容器
        container = client.containers.run(
            DOCKER_IMAGE_NAME,       # 使用我们构建的镜像
            command=container_command, # 在容器内执行的命令
            volumes=volumes,           # 挂载的卷
            device_requests=device_requests, # GPU配置 (如果有效)
            detach=True,               # 以分离模式启动容器，这样我们可以稍后获取日志和状态
            remove=False,              # 不要立即移除，我们需要在它结束后获取日志，然后再手动移除
            # working_dir="/sandbox/outputs" # (可选) 设置容器内脚本的默认工作目录，如果代码中的相对路径是基于输出目录的
                                         # 但通常让代码明确使用 /sandbox/outputs/xxx 更好
            # environment=["PYTHONUNBUFFERED=1"], # (可选) 确保Python输出不被缓冲，更快看到日志
        )
        print(f"--- [DockerMLSandboxTool] 容器 '{container.short_id}' 已启动 ---")
        
        # 6. 等待容器执行完成，并获取退出状态
        #    可以设置一个合理的超时时间，防止脚本无限运行
        #    机器学习任务可能需要较长时间，这里的超时需要根据实际情况调整
        timeout_seconds = 3600 # 例如，1小时。对于简单测试可以设短一些，如60秒。
        print(f"--- [DockerMLSandboxTool] 等待容器执行完成 (超时: {timeout_seconds}秒)... ---")
        
        container_result = container.wait(timeout=timeout_seconds) # .wait()返回一个包含StatusCode的字典
        result_dict["exit_code"] = container_result.get("StatusCode", -1) # 获取退出码
        result_dict["success"] = (result_dict["exit_code"] == 0)
        
        print(f"--- [DockerMLSandboxTool] 容器 '{container.short_id}' 执行完毕，退出码: {result_dict['exit_code']} ---")

        # 7. 获取容器的标准输出和标准错误
        stdout_bytes = container.logs(stdout=True, stderr=False, timestamps=False) # timestamps=False避免日志时间戳
        stderr_bytes = container.logs(stdout=False, stderr=True, timestamps=False)
        
        result_dict["stdout"] = stdout_bytes.decode('utf-8', errors='replace').strip() if stdout_bytes else ""
        result_dict["stderr"] = stderr_bytes.decode('utf-8', errors='replace').strip() if stderr_bytes else ""

        if result_dict["stdout"]:
            print(f"--- [DockerMLSandboxTool] 捕获到Stdout (前500字符):\n{result_dict['stdout'][:500]}... ---")
        if result_dict["stderr"]:
            print(f"--- [DockerMLSandboxTool] 捕获到Stderr (前500字符):\n{result_dict['stderr'][:500]}... ---")
        
        if not result_dict["success"] and not result_dict["stderr"]: # 如果失败但stderr为空，可能是超时或其他问题
            if result_dict["exit_code"] != 0 and result_dict["error_message"] is None: # 避免覆盖之前的错误
                result_dict["error_message"] = f"代码执行失败，退出码: {result_dict['exit_code']}。Stderr为空，可能发生了超时或容器被外部终止。"


    # --- 异常处理块 ---
    except docker.errors.ImageNotFound:
        error_msg = f"关键错误：Docker镜像 '{DOCKER_IMAGE_NAME}' 未找到。请确保您已成功构建该镜像，并且名称与Dockerfile中定义的一致。"
        print(f"--- [DockerMLSandboxTool] {error_msg} ---")
        result_dict["error_message"] = error_msg
        result_dict["success"] = False
    except docker.errors.APIError as docker_api_err:
        error_msg = f"Docker API错误: {docker_api_err}。这可能意味着Docker守护进程未运行、配置错误或与Docker的通信出现问题。"
        print(f"--- [DockerMLSandboxTool] {error_msg} ---")
        result_dict["error_message"] = error_msg
        result_dict["success"] = False
    except requests.exceptions.ConnectionError as conn_err: # docker.from_env() 底层可能用requests
        error_msg = f"无法连接到Docker守护进程: {conn_err}。请确保Docker正在运行。"
        print(f"--- [DockerMLSandboxTool] {error_msg} ---")
        result_dict["error_message"] = error_msg
        result_dict["success"] = False
    except tempfile. estadounidenses as tf_err: # 假设是tempfile的错误
        error_msg = f"创建或写入临时脚本文件时发生错误: {tf_err}"
        print(f"--- [DockerMLSandboxTool] {error_msg} ---")
        result_dict["error_message"] = error_msg
        result_dict["success"] = False
    except Exception as e:
        # 捕获所有其他可能的未知异常
        error_msg = f"执行Docker容器或处理结果时发生未知意外错误: {e}"
        print(f"--- [DockerMLSandboxTool] {error_msg} ---")
        import traceback
        result_dict["error_message"] = error_msg
        result_dict["stderr"] = (result_dict["stderr"] + "\n" + traceback.format_exc()).strip() # 将traceback加入stderr
        result_dict["success"] = False
    
    # --- 清理操作 ---
    finally:
        if container: # 只有当容器对象被成功创建后才尝试移除
            try:
                print(f"--- [DockerMLSandboxTool] 正在尝试移除容器 '{container.short_id}'... ---")
                container.remove(force=True) # 强制移除容器，即使它仍在运行（虽然wait之后应该停止了）
                print(f"--- [DockerMLSandboxTool] 容器 '{container.short_id}' 已移除 ---")
            except docker.errors.NotFound:
                print(f"--- [DockerMLSandboxTool] 容器 '{container.short_id}' 在尝试移除前已不存在 ---")
            except Exception as e_remove_container:
                # 记录移除容器时的错误，但不覆盖主要的执行结果
                print(f"--- [DockerMLSandboxTool] 移除容器 '{container.short_id}' 时发生错误: {e_remove_container} ---")
                if result_dict["error_message"] is None: # 如果还没有其他错误
                    result_dict["error_message"] = f"清理容器时出错: {e_remove_container}"
                elif result_dict.get("stderr") is not None:
                     result_dict["stderr"] += f"\n清理容器时出错: {e_remove_container}"


        if temp_script_dir_on_host_for_this_run and os.path.exists(temp_script_dir_on_host_for_this_run):
            try:
                shutil.rmtree(temp_script_dir_on_host_for_this_run) # 清理宿主机上的临时代码目录
                print(f"--- [DockerMLSandboxTool] 已清理临时代码目录: {temp_script_dir_on_host_for_this_run} ---")
            except Exception as e_rm_dir:
                print(f"--- [DockerMLSandboxTool] 清理临时代码目录 '{temp_script_dir_on_host_for_this_run}' 失败: {e_rm_dir} ---")
                # 记录清理目录时的错误
                if result_dict["error_message"] is None:
                    result_dict["error_message"] = f"清理临时目录时出错: {e_rm_dir}"
                elif result_dict.get("stderr") is not None:
                     result_dict["stderr"] += f"\n清理临时目录时出错: {e_rm_dir}"
        
    return result_dict

# --- (可选) 本地测试这个函数 ---
if __name__ == '__main__':
    print("开始本地测试 execute_ml_code_in_docker 函数...")

    # 确保你的Docker镜像 "mlsandbox:latest" 已经构建好
    # 确保 agent_workspace/datasets 和 agent_workspace/outputs 目录已创建

    # 测试1: 简单的print
    print("\n--- 测试1: 简单打印 ---")
    code1 = "print('Hello from Docker Sandbox!')\nprint('This is a test.')\na = 5 + 10\nprint(f'Result of a: {a}')"
    result1 = execute_ml_code_in_docker(code1)
    print(f"测试1结果: {result1}")

    # 测试2: 使用pandas (假设镜像中有pandas)
    print("\n--- 测试2: 使用pandas创建并打印DataFrame ---")
    code2 = """
import pandas as pd
data = {'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']}
df = pd.DataFrame(data)
print("DataFrame created in Docker:")
print(df)
print(f"Pandas version in Docker: {pd.__version__}")
"""
    result2 = execute_ml_code_in_docker(code2)
    print(f"测试2结果: {result2}")

    # 测试3: 写入文件到 /sandbox/outputs/
    print("\n--- 测试3: 写入文件到输出目录 ---")
    output_filename_in_container = "test_output.txt"
    code3 = f"""
import os
output_dir = "/sandbox/outputs" # 这是容器内的路径
file_path = os.path.join(output_dir, "{output_filename_in_container}")
with open(file_path, "w") as f:
    f.write("This file was written инсульт Docker container.\\n")
    f.write("你好，世界！from Docker.")
print(f"文件已写入到 {{file_path}}")

# 尝试读取刚写入的文件并打印其内容，验证写入成功
with open(file_path, "r") as f:
    content = f.read()
print("读取刚写入的文件内容:")
print(content)
"""
    result3 = execute_ml_code_in_docker(code3)
    print(f"测试3结果: {result3}")
    # 检查宿主机上的 HOST_OUTPUTS_PATH 是否真的生成了 test_output.txt
    host_output_file = os.path.join(HOST_OUTPUTS_PATH, output_filename_in_container)
    if os.path.exists(host_output_file):
        print(f"成功在宿主机找到输出文件: {host_output_file}")
        with open(host_output_file, "r", encoding="utf-8") as f_host:
            print(f"宿主机文件内容:\n{f_host.read()}")
    else:
        print(f"警告：未在宿主机找到预期的输出文件: {host_output_file}")


    # 测试4: 读取 /sandbox/datasets/ 中的文件 (你需要先在宿主机的 HOST_DATASETS_PATH 放一个文件)
    print("\n--- 测试4: 读取数据集目录中的文件 ---")
    # 先在宿主机创建测试数据文件
    host_dataset_filename = "sample_data.csv"
    host_dataset_file_path = os.path.join(HOST_DATASETS_PATH, host_dataset_filename)
    with open(host_dataset_file_path, "w", encoding="utf-8") as f_data:
        f_data.write("colA,colB\n1,apple\n2,banana\n3,cherry")
    print(f"已在宿主机创建测试数据文件: {host_dataset_file_path}")

    code4 = f"""
import pandas as pd
dataset_dir = "/sandbox/datasets" # 容器内路径
file_path = os.path.join(dataset_dir, "{host_dataset_filename}")
try:
    df = pd.read_csv(file_path)
    print("成功从容器内读取CSV文件:")
    print(df.head())
    print(f"DataFrame shape: {{df.shape}}")
except FileNotFoundError:
    print(f"错误：在容器内未找到文件 {{file_path}}")
except Exception as e_read:
    print(f"读取文件时发生错误: {{e_read}}")
"""
    result4 = execute_ml_code_in_docker(code4)
    print(f"测试4结果: {result4}")

    # 测试5: 产生错误的Python代码
    print("\n--- 测试5: 产生错误的Python代码 ---")
    code5 = "print('About to raise an error...')\nraise ValueError('This is a test error from agent code!')"
    result5 = execute_ml_code_in_docker(code5)
    print(f"测试5结果: {result5}")
    assert not result5["success"]
    assert "ValueError: This is a test error from agent code!" in result5["stderr"]

    # 测试6: (可选) GPU测试 (如果你的Docker和宿主机都配置了GPU)
    # print("\n--- 测试6: GPU测试 (如果可用) ---")
    # code6 = """
    # try:
    #     import torch
    #     if torch.cuda.is_available():
    #         print(f"PyTorch CUDA is available! Device: {torch.cuda.get_device_name(0)}")
    #         a = torch.randn(3,3).cuda()
    #         b = torch.randn(3,3).cuda()
    #         print(f"PyTorch tensor on GPU: {a @ b}")
    #     else:
    #         print("PyTorch CUDA is NOT available.")
    # except ImportError:
    #     print("PyTorch is not installed in the Docker image.")
    # except Exception as e_gpu:
    #     print(f"GPU test error: {e_gpu}")
    # """
    # result6 = execute_ml_code_in_docker(code6, use_gpu=True) # 尝试请求GPU
    # print(f"测试6 (GPU) 结果: {result6}")

    print("\n本地测试结束。")