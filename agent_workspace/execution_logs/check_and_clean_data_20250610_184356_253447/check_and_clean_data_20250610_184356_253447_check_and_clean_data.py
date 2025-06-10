
import scipy.io
import numpy as np

# 定义文件路径
chm_file = '/sandbox/datasets/CHMdata/CHM_2016_2020.mat'
chirps_file = '/sandbox/datasets/CHIRPSdata/chirps_2016_2020.mat'

# 加载数据函数
def load_and_check_data(filepath, dataset_name):
    print(f"\n--- 正在检查数据集: {dataset_name} ({filepath}) ---")
    try:
        mat_data = scipy.io.loadmat(filepath)
        data_array = mat_data['data'] # 假设变量名为 'data'
        print(f"原始数据形状: {data_array.shape}, 类型: {data_array.dtype}")

        # 检查 NaN 缺失值
        nan_count = np.sum(np.isnan(data_array))
        print(f"NaN 缺失值数量: {nan_count}")
        if nan_count > 0:
            print(f"NaN 缺失值占总数据量的比例: {nan_count / data_array.size * 100:.4f}%")

        # 检查负值（降雨量不应为负）
        negative_count = np.sum(data_array < 0)
        print(f"负值数量: {negative_count}")
        if negative_count > 0:
            print(f"负值占总数据量的比例: {negative_count / data_array.size * 100:.4f}%")
            # 初步清洗：将负值设为0
            cleaned_data_array = np.where(data_array < 0, 0, data_array)
            print("已将所有负值设置为 0。")
        else:
            cleaned_data_array = data_array.copy() # 如果没有负值，也复制一份以保持一致性

        # 检查异常大值（初步判断，不作为硬性清洗）
        # 简单统计
        print(f"数据最小值 (清洗负值后): {np.nanmin(cleaned_data_array)}")
        print(f"数据最大值 (清洗负值后): {np.nanmax(cleaned_data_array)}")
        print(f"数据平均值 (清洗负值后): {np.nanmean(cleaned_data_array)}")
        print(f"数据标准差 (清洗负值后): {np.nanstd(cleaned_data_array)}")

        # 返回清洗了负值的数据，NaN值保持不变，待用户决定如何处理
        return cleaned_data_array, nan_count

    except FileNotFoundError:
        print(f"错误：文件未找到在 {filepath}")
        return None, 0
    except KeyError:
        print(f"错误：在 {filepath} 中未找到名为 'data' 的变量。请确认变量名。")
        return None, 0
    except Exception as e:
        print(f"处理 {filepath} 时发生未知错误: {e}")
        return None, 0

# 执行检查
chm_cleaned_data, chm_nan_count = load_and_check_data(chm_file, "CHM")
chirps_cleaned_data, chirps_nan_count = load_and_check_data(chirps_file, "CHIRPS")

# 总结并询问用户对 NaN 的处理方式
print("\n--- 数据清洗与缺失值总结 ---")
if chm_nan_count > 0 or chirps_nan_count > 0:
    print("在CHM和/或CHIRPS数据中检测到NaN缺失值。")
    print("对于这些NaN值，您希望如何处理？常见的策略包括：")
    print("1. 删除包含NaN的整个时间步或格点（可能导致数据大量丢失）。")
    print("2. 使用0填充。")
    print("3. 使用邻近格点或时间步的平均值/中位数进行插值填充。")
    print("4. 使用更复杂的插值方法（如线性插值、样条插值）。")
    print("请告诉我您的偏好，我将为您执行相应的清洗操作。")
else:
    print("未在CHM和CHIRPS数据中检测到NaN缺失值。")
    print("所有负值已成功处理为0。")
    print("数据初步清洗完成。")
