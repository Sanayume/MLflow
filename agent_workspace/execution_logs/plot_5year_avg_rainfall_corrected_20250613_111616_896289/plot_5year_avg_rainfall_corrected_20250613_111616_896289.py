
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os

# 定义数据路径和输出路径
datasets_dir = '/sandbox/datasets/'
outputs_dir = '/sandbox/outputs/rainfall_maps/'
os.makedirs(outputs_dir, exist_ok=True)

# 降雨产品列表
rainfall_products = ['CHIRPS', 'CHM', 'CMORPH', 'GSMAP', 'IMERG', 'PERSIANN', 'sm2rain']
years = range(2016, 2021) # 5年

# 加载mask数据
mask_path = os.path.join(datasets_dir, 'mask', 'mask.mat')
try:
    mask_data = scipy.io.loadmat(mask_path)['data']
    print(f"Mask data shape: {mask_data.shape}")
except Exception as e:
    print(f"Error loading mask.mat: {e}")
    exit()

# 遍历每个降雨产品并处理
for product in rainfall_products:
    print(f"Processing {product}...")
    product_data_path = os.path.join(datasets_dir, 'rainfalldata', f'{product}data', f'{product}_2016_2020.mat')

    try:
        # 加载.mat文件
        mat_content = scipy.io.loadmat(product_data_path)
        # 假设数据存储在 'data' 键下
        # 根据用户指正，1827是时间维度，所以数据形状可能是 (纬度, 经度, 时间)
        rainfall_data = mat_content['data']
        print(f"Original {product} data shape: {rainfall_data.shape}")

        # 检查并确保时间维度是最后一个维度，如果是 (时间, 纬度, 经度) 需要转置
        # 如果是 (纬度, 经度, 时间) 就不需要
        # 这里假设时间维度是最后一个 (即轴2，索引为2)
        if rainfall_data.shape[0] == 1827: # 假设时间维度是第一个维度
            print(f"Transposing {product} data from {rainfall_data.shape} to (lat, lon, time)...")
            rainfall_data = np.transpose(rainfall_data, (1, 2, 0)) # 变为 (lat, lon, time)
            print(f"Transposed {product} data shape: {rainfall_data.shape}")
        
        # 确保时间维度是1827
        if rainfall_data.shape[2] != 1827:
            print(f"Warning: Unexpected time dimension for {product}: {rainfall_data.shape[2]}. Expected 1827.")
            # 尝试根据mask形状调整，但如果维度顺序不确定，这里可能导致问题
            # 更安全的做法是抛出错误或跳过
            continue

        # 计算5年平均降雨量，沿着时间维度 (轴2) 进行平均
        # 结果形状应为 (纬度, 经度)
        five_year_avg_rainfall = np.mean(rainfall_data, axis=2)
        print(f"5-year average {product} rainfall shape: {five_year_avg_rainfall.shape}")

        # 应用mask
        # 检查mask和降雨数据的空间维度是否匹配
        if five_year_avg_rainfall.shape == mask_data.shape:
            masked_rainfall = np.where(mask_data == 1, five_year_avg_rainfall, np.nan)
            print(f"Mask applied for {product}.")
        else:
            print(f"Warning: {product} average rainfall data shape {five_year_avg_rainfall.shape} does not match mask shape {mask_data.shape}. Mask not applied.")
            masked_rainfall = five_year_avg_rainfall # 不应用mask，直接使用原始平均值

        # 绘制图像
        plt.figure(figsize=(10, 8))
        plt.imshow(masked_rainfall, cmap='viridis', origin='lower') # origin='lower' 确保图像方向正确
        plt.colorbar(label='Average Rainfall')
        plt.title(f'{product} 5-Year Average Rainfall (2016-2020)')
        plt.xlabel('Longitude Index')
        plt.ylabel('Latitude Index')

        # 保存图像
        output_filename = f'{product}_5year_avg_rainfall.png'
        output_path = os.path.join(outputs_dir, output_filename)
        plt.savefig(output_path)
        plt.close()
        print(f"Saved {output_filename} to {outputs_path}")

    except FileNotFoundError:
        print(f"Error: {product}_2016_2020.mat not found at {product_data_path}")
    except Exception as e:
        print(f"Error processing {product}: {e}")

print("All rainfall maps generated and saved.")
