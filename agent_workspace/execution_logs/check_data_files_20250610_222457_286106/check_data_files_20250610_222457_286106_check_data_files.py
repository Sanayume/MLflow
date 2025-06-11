
import os
import scipy.io
import numpy as np

# 定义数据路径
DATASETS_PATH = '/sandbox/datasets/'
MASK_PATH = os.path.join(DATASETS_PATH, 'mask', 'mask.mat')
RAINFALL_DATA_PATH = os.path.join(DATASETS_PATH, 'rainfalldata')

print(f"尝试加载掩码文件: {MASK_PATH}")
try:
    mask_data = scipy.io.loadmat(MASK_PATH)
    print("成功加载 mask.mat 文件。")
    if 'data' in mask_data:
        mask_matrix = mask_data['data']
        print(f"mask_matrix 的形状: {mask_matrix.shape}")
        print(f"mask_matrix 的数据类型: {mask_matrix.dtype}")
        # 检查一些值以确认逻辑
        print(f"mask_matrix 中大于等于1的值的数量: {np.sum(mask_matrix >= 1)}")
        print(f"mask_matrix 中大于等于2的值的数量 (长江流域): {np.sum(mask_matrix >= 2)}")
    else:
        print("警告: mask.mat 文件中未找到 'data' 键。请检查文件内容。")
except FileNotFoundError:
    print(f"错误: 未找到文件 {MASK_PATH}。请确保文件已放置在宿主机的 F:\MLflow\agent_workspace/datasets/mask/ 目录下。")
except Exception as e:
    print(f"加载 mask.mat 时发生错误: {e}")

print(f"\n列出 {RAINFALL_DATA_PATH} 目录下的 .mat 文件:")
try:
    mat_files = [f for f in os.listdir(RAINFALL_DATA_PATH) if f.endswith('.mat')]
    if mat_files:
        for f in mat_files:
            print(f)
    else:
        print(f"在 {RAINFALL_DATA_PATH} 目录中未找到 .mat 文件。请确保数据已放置。")
except FileNotFoundError:
    print(f"错误: 未找到目录 {RAINFALL_DATA_PATH}。请确保数据已放置在宿主机的 F:\MLflow\agent_workspace/datasets/rainfalldata/ 目录下。")
except Exception as e:
    print(f"列出 {RAINFALL_DATA_PATH} 目录时发生错误: {e}")

