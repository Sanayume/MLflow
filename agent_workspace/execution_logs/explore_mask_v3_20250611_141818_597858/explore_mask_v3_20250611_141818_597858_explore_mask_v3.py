
import scipy.io
import numpy as np
import os

mask_file_path = '/sandbox/datasets/mask/mask.mat'

if os.path.exists(mask_file_path):
    try:
        mat_data = scipy.io.loadmat(mask_file_path)
        print(f"成功加载文件: {mask_file_path}")
        print("文件中的变量 (keys):", mat_data.keys())

        # 尝试访问可能的mask变量名
        mask_variable_name = None
        if 'mask' in mat_data:
            mask_variable_name = 'mask'
        elif 'data' in mat_data: # 兼容旧的命名
            mask_variable_name = 'data'
        
        if mask_variable_name:
            mask_data = mat_data[mask_variable_name]
            print(f"变量 '{mask_variable_name}' 的形状: {mask_data.shape}")
            print(f"变量 '{mask_variable_name}' 的数据类型: {mask_data.dtype}")
            
            # 检查唯一值和范围
            unique_values = np.unique(mask_data)
            print(f"变量 '{mask_variable_name}' 的唯一值: {unique_values}")
            print(f"变量 '{mask_variable_name}' 的最小值: {np.min(mask_data)}")
            print(f"变量 '{mask_variable_name}' 的最大值: {np.max(mask_data)}")

            if np.array_equal(unique_values, [0, 1]) or np.array_equal(unique_values, [0]) or np.array_equal(unique_values, [1]):
                print(f"变量 '{mask_variable_name}' 看起来是一个二进制掩膜 (只包含0和1)。")
            else:
                print(f"变量 '{mask_variable_name}' 包含除0和1以外的值，可能不是一个纯粹的二进制掩膜。")
        else:
            print("未在文件中找到名为 'mask' 或 'data' 的变量。")

    except Exception as e:
        print(f"加载或处理 {mask_file_path} 时发生错误: {e}")
else:
    print(f"文件不存在: {mask_file_path}")
