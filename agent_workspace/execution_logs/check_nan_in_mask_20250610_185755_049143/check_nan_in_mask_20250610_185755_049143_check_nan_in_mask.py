
import scipy.io
import numpy as np

# 定义文件路径
chm_file = '/sandbox/datasets/CHMdata/CHM_2016.mat'
chirps_file = '/sandbox/datasets/CHIRPSdata/chirps_2016.mat'
sm2rain_file = '/sandbox/datasets/sm2raindata/sm2rain_2016.mat'

print("正在加载 CHM_2016.mat...")
chm_data_raw = scipy.io.loadmat(chm_file)['data']
print("CHM_2016.mat 加载完成。")

# 创建 CHM 的有效地理掩膜
chm_mask = ~np.isnan(chm_data_raw)
total_possible_valid_points = np.sum(chm_mask)
print(f"CHM 有效地理区域内的总点数: {total_possible_valid_points}")

# 检查 CHIRPS 数据
print("正在加载 chirps_2016.mat...")
chirps_data_raw = scipy.io.loadmat(chirps_file)['data']
print("chirps_2016.mat 加载完成。")

# 在 CHM 有效区域内检查 CHIRPS 的 NaN
# 我们只关心 CHM_mask 为 True 的地方，如果 chirps_data_raw 在这些地方是 NaN，则表示缺失
chirps_nan_in_chm_mask = np.sum(np.isnan(chirps_data_raw[chm_mask]))
print(f"在 CHM 有效地理区域内，chirps_2016.mat 中的 NaN 数量: {chirps_nan_in_chm_mask}")
if total_possible_valid_points > 0:
    chirps_nan_percentage = (chirps_nan_in_chm_mask / total_possible_valid_points) * 100
    print(f"在 CHM 有效地理区域内，chirps_2016.mat 中的 NaN 比例: {chirps_nan_percentage:.4f}%")
else:
    print("CHM 有效地理区域内没有数据点，无法计算 NaN 比例。")


# 检查 sm2rain 数据
print("正在加载 sm2rain_2016.mat...")
sm2rain_data_raw = scipy.io.loadmat(sm2rain_file)['data']
print("sm2rain_2016.mat 加载完成。")

# 在 CHM 有效区域内检查 sm2rain 的 NaN
sm2rain_nan_in_chm_mask = np.sum(np.isnan(sm2rain_data_raw[chm_mask]))
print(f"在 CHM 有效地理区域内，sm2rain_2016.mat 中的 NaN 数量: {sm2rain_nan_in_chm_mask}")
if total_possible_valid_points > 0:
    sm2rain_nan_percentage = (sm2rain_nan_in_chm_mask / total_possible_valid_points) * 100
    print(f"在 CHM 有效地理区域内，sm2rain_2016.mat 中的 NaN 比例: {sm2rain_nan_percentage:.4f}%")
else:
    print("CHM 有效地理区域内没有数据点，无法计算 NaN 比例。")

