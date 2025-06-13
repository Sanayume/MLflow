
import os
import scipy.io as sio
import numpy as np

data_dir = '/sandbox/datasets/rainfalldata/'
output_dir = '/sandbox/outputs/'
os.makedirs(output_dir, exist_ok=True)

mat_files = [f for f in os.listdir(data_dir) if f.endswith('.mat')]
print(f"在 {data_dir} 目录下找到的 .mat 文件: {mat_files}")

if mat_files:
    first_mat_file_path = os.path.join(data_dir, mat_files[0])
    print(f"尝试加载第一个 .mat 文件: {first_mat_file_path}")
    try:
        mat_contents = sio.loadmat(first_mat_file_path)
        print("第一个 .mat 文件的内容键:")
        for key in mat_contents.keys():
            if not key.startswith('__'): # 忽略Python内部键
                print(f"  - {key}: 类型={type(mat_contents[key])}, 形状={mat_contents[key].shape if isinstance(mat_contents[key], np.ndarray) else '非数组'}")
        
        # 尝试猜测哪个键是降雨数据，并打印其形状和一些统计信息
        # 假设降雨数据可能是 'data', 'rainfall', 'precip' 或类似名称的数组
        potential_data_keys = ['data', 'rainfall', 'precip', 'rain', 'value']
        found_data_key = None
        for k in potential_data_keys:
            if k in mat_contents and isinstance(mat_contents[k], np.ndarray) and mat_contents[k].ndim >= 2:
                found_data_key = k
                break
        
        if found_data_key:
            print(f"猜测降雨数据键为 '{found_data_key}'。其形状: {mat_contents[found_data_key].shape}")
            # 打印一些统计信息，帮助理解数据范围
            if mat_contents[found_data_key].size > 0:
                print(f"  最小值: {np.min(mat_contents[found_data_key])}")
                print(f"  最大值: {np.max(mat_contents[found_data_key])}")
                print(f"  平均值: {np.mean(mat_contents[found_data_key])}")
                print(f"  标准差: {np.std(mat_contents[found_data_key])}")
            else:
                print("  数据数组为空。")
        else:
            print("未能自动识别降雨数据键，请手动检查上述键列表。")

    except Exception as e:
        print(f"加载 .mat 文件时发生错误: {e}")
else:
    print("在指定目录中未找到任何 .mat 文件。")
