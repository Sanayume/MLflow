
import scipy.io
import numpy as np
import os

# 定义2016年数据的路径列表
data_paths = [
    '/sandbox/datasets/CHMdata/CHM_2016.mat',
    '/sandbox/datasets/CHIRPSdata/chirps_2016.mat',
    '/sandbox/datasets/CMORPHdata/CMORPH_2016.mat',
    '/sandbox/datasets/GSMAPdata/GSMAP_2016.mat',
    '/sandbox/datasets/IMERGdata/IMERG_2016.mat',
    '/sandbox/datasets/PERSIANNdata/PERSIANNdata_2016.mat', # 注意这里的文件名是 PERSIANNdata_2016.mat
    '/sandbox/datasets/sm2raindata/sm2rain_2016.mat'
]

print("正在检查2016年所有降雨产品的数据质量...")

for path in data_paths:
    file_name = os.path.basename(path)
    print(f"\n--- 检查文件: {file_name} ---")
    
    try:
        mat_data = scipy.io.loadmat(path)
        # 假设数据存储在名为 'data' 的变量中
        if 'data' in mat_data:
            data_array = mat_data['data']
            
            total_elements = data_array.size
            
            # 检查NaN值
            nan_count = np.sum(np.isnan(data_array))
            nan_percentage = (nan_count / total_elements) * 100 if total_elements > 0 else 0
            print(f"NaN 缺失值数量: {nan_count} ({nan_percentage:.4f}%)")
            
            # 检查负值 (排除NaN)
            non_nan_data = data_array[~np.isnan(data_array)]
            negative_count = np.sum(non_nan_data < 0)
            print(f"负值数量 (排除NaN): {negative_count}")
            
            # 检查最大值和最小值 (排除NaN)
            if non_nan_data.size > 0:
                min_val = np.min(non_nan_data)
                max_val = np.max(non_nan_data)
                print(f"有效数据最小值: {min_val:.2f}")
                print(f"有效数据最大值: {max_val:.2f}")
            else:
                print("没有有效数据点（全部为NaN）。")
                
        else:
            print(f"警告: 文件 {file_name} 中未找到名为 'data' 的变量。")
            
    except Exception as e:
        print(f"加载或处理文件 {file_name} 时发生错误: {e}")

print("\n2016年数据质量检查完成。")
