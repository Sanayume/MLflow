
import scipy.io
import numpy as np
from sklearn.metrics import mean_squared_error

print("正在加载 CHM 和 CHIRPS 数据...")
chm_data_raw = scipy.io.loadmat('/sandbox/datasets/CHMdata/CHM_2016_2020.mat')
chirps_data_raw = scipy.io.loadmat('/sandbox/datasets/CHIRPSdata/chirps_2016_2020.mat')

chm_array = chm_data_raw['data']
chirps_array = chirps_data_raw['data']

print(f"CHM 数据形状: {chm_array.shape}, CHIRPS 数据形状: {chirps_array.shape}")

# 确保数据为浮点类型
chm_array = chm_array.astype(np.float64)
chirps_array = chirps_array.astype(np.float64)

# 识别共同的有效数据区域（即两者都不是NaN的区域）
# 我们之前已经确认NaN是地理掩膜，所以只比较非NaN的区域
valid_mask = ~np.isnan(chm_array) & ~np.isnan(chirps_array)

# 提取有效区域的数据
chm_valid = chm_array[valid_mask]
chirps_valid = chirps_array[valid_mask]

print(f"有效数据点数量: {len(chm_valid)}")

if len(chm_valid) == 0:
    print("错误：没有共同的有效数据点可用于比较。")
else:
    # 计算 RMSE
    rmse = np.sqrt(mean_squared_error(chm_valid, chirps_valid))
    print(f"CHIRPS vs CHM - 均方根误差 (RMSE): {rmse:.4f}")

    # 计算皮尔逊相关系数
    # np.corrcoef 返回一个相关矩阵，我们需要取非对角线元素
    correlation_matrix = np.corrcoef(chm_valid, chirps_valid)
    correlation = correlation_matrix[0, 1]
    print(f"CHIRPS vs CHM - 皮尔逊相关系数: {correlation:.4f}")

print("CHIRPS 产品评估完成。")
