
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
mask = None
try:
    mask_data = sio.loadmat(mask_file)
    print(f"mask.mat 文件中的键: {mask_data.keys()}")
    # 尝试寻找可能的mask键
    if 'mask' in mask_data: # 优先尝试'mask'键
        mask = mask_data['mask']
    else: # 如果没有，尝试取第一个非元数据键
        for key in mask_data:
            if not key.startswith('__'):
                mask = mask_data[key]
                print(f"使用键 '{key}' 作为mask数据。")
                break
    
    if mask is not None:
        print(f"成功加载mask文件: {mask_file}, mask shape: {mask.shape}")
    else:
        print(f"警告: 在 {mask_file} 中未找到合适的mask数据。")
except FileNotFoundError:
    print(f"mask文件未找到: {mask_file}")
except Exception as e:
    print(f"加载mask文件失败: {mask_file}. 错误: {e}")
    mask = None

# 遍历每个产品并处理
for product in products:
    print(f"正在处理产品: {product}...")
    # 存储每个年份的日平均降雨量（二维数组）
    annual_avg_rainfall_list = [] 
    
    for year in years:
        file_path = os.path.join(datasets_path, 'rainfalldata', f'{product}data', f'{product}_{year}.mat')
        
        try:
            mat_data = sio.loadmat(file_path)
            if 'data' in mat_data:
                rainfall_data = mat_data['data']
                print(f"  成功加载 {product}_{year}.mat, 原始数据形状: {rainfall_data.shape}")
                
                # 对当前年份的数据在时间维度（第三个维度）上计算平均，得到日平均降雨量
                # 假设数据是 (纬度, 经度, 时间)
                if rainfall_data.ndim == 3:
                    daily_avg = np.mean(rainfall_data, axis=2) # 沿时间维度取平均
                    annual_avg_rainfall_list.append(daily_avg)
                    print(f"  {product}_{year} 日平均降雨数据形状: {daily_avg.shape}")
                else:
                    print(f"  警告: {product}_{year}.mat 数据维度不是3维，无法计算日平均。跳过。")
            else:
                print(f"  警告: {product}_{year}.mat 中未找到 'data' 键。请检查文件结构。")
        except FileNotFoundError:
            print(f"  文件未找到: {file_path}")
        except Exception as e:
            print(f"  加载或处理 {file_path} 时发生错误: {e}")

    if annual_avg_rainfall_list:
        # 将所有年份的日平均降雨数据堆叠起来并计算5年平均值
        # 此时 annual_avg_rainfall_list 中的每个元素都是 (144, 256) 的二维数组
        five_year_avg_rainfall = np.mean(np.array(annual_avg_rainfall_list), axis=0)
        print(f"  {product} 5年平均降雨数据形状: {five_year_avg_rainfall.shape}")

        # 应用mask (如果mask存在且形状匹配)
        if mask is not None and mask.shape == five_year_avg_rainfall.shape:
            # 假设mask是布尔值或0/1，0表示无效，1表示有效
            # 将mask为0或False的区域设为NaN，使其在图中透明
            five_year_avg_rainfall_masked = np.where(mask == 1, five_year_avg_rainfall, np.nan)
            print(f"  {product} 数据已应用mask。")
        else:
            five_year_avg_rainfall_masked = five_year_avg_rainfall
            if mask is not None:
                print(f"  警告: mask形状 ({mask.shape}) 与降雨数据形状 ({five_year_avg_rainfall.shape}) 不匹配，未应用mask。")
            else:
                print(f"  未应用mask，因为mask未加载或不适用。")

        # 绘制并保存图像
        plt.figure(figsize=(10, 8))
        plt.imshow(five_year_avg_rainfall_masked, cmap='viridis', origin='lower') 
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
