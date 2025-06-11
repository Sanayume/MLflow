
import scipy.io as sio
import numpy as np

try:
    mat_file_path = '/sandbox/datasets/mask/mask.mat'
    data = sio.loadmat(mat_file_path)

    print(f"成功加载文件: {mat_file_path}")
    print("文件中的变量键:")
    for key in data.keys():
        print(f"- {key}")

    # 尝试打印一个看起来像数据变量的键的形状和类型
    # 假设 'mask' 或类似名称是实际数据
    if 'mask' in data:
        print(f"变量 'mask' 的形状: {data['mask'].shape}")
        print(f"变量 'mask' 的数据类型: {data['mask'].dtype}")
    elif '__header__' in data and '__version__' in data and '__globals__' in data:
        # 排除matlab默认的元数据键，尝试寻找其他键
        data_keys = [key for key in data.keys() if not key.startswith('__')]
        if data_keys:
            first_data_key = data_keys[0]
            print(f"找到第一个数据变量键: '{first_data_key}'")
            print(f"变量 '{first_data_key}' 的形状: {data[first_data_key].shape}")
            print(f"变量 '{first_data_key}' 的数据类型: {data[first_data_key].dtype}")
        else:
            print("除了MATLAB元数据，没有找到其他数据变量键。")

except FileNotFoundError:
    print(f"错误: 文件未找到 - {mat_file_path}")
except Exception as e:
    print(f"加载或处理MAT文件时发生错误: {e}")
