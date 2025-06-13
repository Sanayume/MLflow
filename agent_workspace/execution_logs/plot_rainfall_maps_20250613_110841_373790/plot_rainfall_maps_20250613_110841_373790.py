
import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

# 定义数据路径和产品列表
datasets_path = '/sandbox/datasets/'
output_path = '/sandbox/outputs/rainfall_maps/'
os.makedirs(output_path, exist_ok=True)

products = ['CHIRPS', 'CHM', 'CMORPH', 'GSMAP', 'IMERG', 'PERSIANN', 'sm2rain']
years = range(2016, 2021) # 2016, 2017, 2018, 2019, 2020

# 加载mask文件
mask_file = os.path.join(datasets_path, 'mask', 'mask.mat')
try:
    mask_data = sio.loadmat(mask_file)
    # 假设mask数据在'mask'键下，或者你需要根据实际文件内容调整
    mask = mask_data['mask']
    print(f"成功加载mask文件: {mask_file}, mask shape: {mask.shape}")
except Exception as e:
    print(f"加载mask文件失败: {mask_file}. 错误: {e}")
    # 如果mask加载失败，尝试继续而不使用mask
    mask = None

# 遍历每个产品并处理
for product in products:
    print(f"正在处理产品: {product}...")
    product_data_list = []
    
    for year in years:
        # 构建文件路径，假设数据在 'product_name/product_name_year.mat'
        file_path = os.path.join(datasets_path, 'rainfalldata', f'{product}data', f'{product}_{year}.mat')
        
        try:
            mat_data = sio.loadmat(file_path)
            # 假设降雨数据存储在名为 'data' 的键下
            # 如果不是，你可能需要根据实际文件结构调整这里的键名
            if 'data' in mat_data:
                rainfall_data = mat_data['data']
                product_data_list.append(rainfall_data)
                print(f"  成功加载 {product}_{year}.mat, 数据形状: {rainfall_data.shape}")
            else:
                print(f"  警告: {product}_{year}.mat 中未找到 'data' 键。请检查文件结构。")
        except FileNotFoundError:
            print(f"  文件未找到: {file_path}")
        except Exception as e:
            print(f"  加载或处理 {file_path} 时发生错误: {e}")

    if product_data_list:
        # 将所有年份的数据堆叠起来并计算平均值
        # 假设所有年份的数据形状一致
        avg_rainfall = np.mean(np.array(product_data_list), axis=0)
        print(f"  {product} 5年平均降雨数据形状: {avg_rainfall.shape}")

        # 应用mask (如果mask存在且形状匹配)
        if mask is not None and mask.shape == avg_rainfall.shape:
            # 假设mask是布尔值，True表示有效区域，False表示无效区域
            # 或者mask是0/1，0表示无效，1表示有效
            # 这里将mask为0或False的区域设为NaN，使其在图中透明
            avg_rainfall_masked = np.where(mask == 1, avg_rainfall, np.nan)
            print(f"  {product} 数据已应用mask。")
        else:
            avg_rainfall_masked = avg_rainfall
            if mask is not None:
                print(f"  警告: mask形状 ({mask.shape}) 与降雨数据形状 ({avg_rainfall.shape}) 不匹配，未应用mask。")
            else:
                print(f"  未应用mask，因为mask未加载。")

        # 绘制并保存图像
        plt.figure(figsize=(10, 8))
        plt.imshow(avg_rainfall_masked, cmap='viridis', origin='lower') # origin='lower' 对于地理数据通常是正确的
        plt.colorbar(label='Average Rainfall (mm/day)')
        plt.title(f'{product} 5-Year Average Rainfall (2016-2020)')
        plt.xlabel('Longitude Index')
        plt.ylabel('Latitude Index')
        
        output_filename = os.path.join(output_path, f'{product}_5year_avg_rainfall.png')
        plt.savefig(output_filename)
        plt.close() # 关闭图形，释放内存
        print(f"  {product} 5年平均降雨图已保存到: {output_filename}")
    else:
        print(f"  未找到 {product} 的任何数据，跳过绘图。")

print("所有产品处理完成。")
