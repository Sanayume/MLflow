
import scipy.io
import numpy as np

file_paths = [
    "/sandbox/datasets/CMORPHdata/CMORPH_2016_2020.mat",
    "/sandbox/datasets/GSMAPdata/GSMAP_2016_2020.mat",
    "/sandbox/datasets/IMERGdata/IMERG_2016_2020.mat",
    "/sandbox/datasets/PERSIANNdata/PERSIANN_2016_2020.mat",
    "/sandbox/datasets/sm2raindata/sm2rain_2016_2020.mat"
]

for file_path in file_paths:
    print(f"\n--- 检查文件: {file_path} ---")
    try:
        mat_data = scipy.io.loadmat(file_path)
        if 'data' in mat_data:
            data_array = mat_data['data']
            total_elements = data_array.size

            # 检查负值
            negative_values_count = np.sum(data_array < 0)
            if negative_values_count > 0:
                print(f"  检测到 {negative_values_count} 个负值。已将其设置为 0。")
                data_array[data_array < 0] = 0
            else:
                print("  未检测到负值。")

            # 检查 NaN 缺失值
            nan_count = np.sum(np.isnan(data_array))
            nan_percentage = (nan_count / total_elements) * 100 if total_elements > 0 else 0
            print(f"  检测到 {nan_count} 个 NaN 值，占总数据量的 {nan_percentage:.4f}%。")
            print("  （根据用户先前的说明，这些 NaN 值预计为地理区域掩膜，不视为数据缺失。）")

            # 检查最大值 (初步异常值检测)
            # 过滤掉 NaN 之后再计算最大值，避免 NaN 影响结果
            valid_data = data_array[~np.isnan(data_array)]
            if valid_data.size > 0:
                max_value = np.max(valid_data)
                min_value = np.min(valid_data)
                print(f"  有效数据范围: Min={min_value:.2f}, Max={max_value:.2f}")
                # 对于降雨数据，过大的值可能需要进一步判断，此处仅打印
            else:
                print("  无有效数据（所有值均为 NaN）。")

        else:
            print("  错误: .mat 文件中未找到 'data' 变量。")

    except FileNotFoundError:
        print(f"  错误: 文件 {file_path} 未找到。")
    except Exception as e:
        print(f"  处理文件时发生错误: {e}")
