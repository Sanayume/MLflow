
import scipy.io
import numpy as np
import os

rainfall_data_dir = '/sandbox/datasets/rainfalldata/'

files_to_check = {
    'CHIRPS_2016_2020': os.path.join(rainfall_data_dir, 'CHIRPSdata', 'chirps_2016_2020.mat'),
    'CMORPH_2016_2020': os.path.join(rainfall_data_dir, 'CMORPHdata', 'CMORPH_2016_2020.mat')
}

for name, file_path in files_to_check.items():
    print(f"\n--- 检查文件: {name} ({file_path}) ---")
    if os.path.exists(file_path):
        try:
            mat_data = scipy.io.loadmat(file_path)
            print("文件中的变量 (keys):", mat_data.keys())

            # 尝试访问可能的降雨数据变量名
            data_variable_name = None
            if 'data' in mat_data:
                data_variable_name = 'data'
            elif name.lower().startswith('chirps') and 'chirps' in mat_data: # 针对CHIRPS的特殊处理
                data_variable_name = 'chirps'
            elif name.lower().startswith('cmorph') and 'CMORPH' in mat_data: # 针对CMORPH的特殊处理
                data_variable_name = 'CMORPH'
            
            if data_variable_name:
                rainfall_data = mat_data[data_variable_name]
                print(f"变量 '{data_variable_name}' 的形状: {rainfall_data.shape}")
                print(f"变量 '{data_variable_name}' 的数据类型: {rainfall_data.dtype}")
                print(f"变量 '{data_variable_name}' 的最小值: {np.nanmin(rainfall_data)}")
                print(f"变量 '{data_variable_name}' 的最大值: {np.nanmax(rainfall_data)}")
                print(f"变量 '{data_variable_name}' 中 NaN 值的数量: {np.sum(np.isnan(rainfall_data))}")
            else:
                print("未在文件中找到常见的降雨数据变量名 (如 'data', 'chirps', 'CMORPH')")

        except Exception as e:
            print(f"加载或处理 {file_path} 时发生错误: {e}")
    else:
        print(f"文件不存在: {file_path}")
