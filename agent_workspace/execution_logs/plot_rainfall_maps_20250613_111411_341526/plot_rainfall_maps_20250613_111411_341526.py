
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os

# 定义数据和mask路径
mask_path = '/sandbox/datasets/mask/mask.mat'
rainfall_data_dir = '/sandbox/datasets/rainfalldata/'
output_dir = '/sandbox/outputs/rainfall_maps/'

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 加载mask数据
try:
    mask_data = scipy.io.loadmat(mask_path)
    # 假设mask数据在.mat文件的'data'键下
    mask = mask_data['data']
    print(f"成功加载mask文件: {mask_path}, mask形状: {mask.shape}")
except Exception as e:
    print(f"加载mask文件失败: {e}")
    exit()

# 定义降雨产品列表 (确保与文件名匹配)
# 根据用户指正，CHM是地面观测站点，其他是卫星产品
rainfall_products = [
    'CHIRPS',
    'CHM',
    'CMORPH',
    'GSMAP',
    'IMERG',
    'PERSIANN',
    'sm2rain'
]

# 定义年份范围
years = range(2016, 2021)

for product_name in rainfall_products:
    try:
        # 构建2016-2020年聚合数据的文件路径
        product_file_path = os.path.join(rainfall_data_dir, f'{product_name}data', f'{product_name}_2016_2020.mat')

        if not os.path.exists(product_file_path):
            print(f"错误: 未找到产品 {product_name} 的聚合数据文件: {product_file_path}")
            continue

        # 加载数据
        product_data_mat = scipy.io.loadmat(product_file_path)
        # 假设降雨数据在.mat文件的'data'键下
        product_data = product_data_mat['data']

        print(f"成功加载 {product_name} 数据文件: {product_file_path}, 形状: {product_data.shape}")

        # 确保数据是三维的 (时间, 纬度, 经度)
        if product_data.ndim != 3:
            print(f"警告: {product_name} 数据不是三维的，跳过处理。")
            continue

        # 计算5年平均降雨量
        # product_data的形状应为 (天数, 纬度, 经度)
        # 计算每个 (纬度, 经度) 网格点的平均值
        five_year_avg_rainfall = np.nanmean(product_data, axis=0)
        print(f"{product_name} 5年平均降雨量形状: {five_year_avg_rainfall.shape}")

        # 应用mask
        # 确保mask和降雨数据的空间维度匹配
        if mask.shape == five_year_avg_rainfall.shape:
            masked_rainfall = np.ma.array(five_year_avg_rainfall, mask=~mask)
        else:
            print(f"警告: {product_name} 的降雨数据 ({five_year_avg_rainfall.shape}) 与mask ({mask.shape}) 形状不匹配，不应用mask。")
            masked_rainfall = five_year_avg_rainfall

        # 绘制图像
        plt.figure(figsize=(10, 8))
        plt.imshow(masked_rainfall, cmap='viridis', origin='lower')
        plt.colorbar(label='Average Rainfall')
        plt.title(f'{product_name} 5-Year Average Rainfall (2016-2020)')
        plt.xlabel('Longitude Index')
        plt.ylabel('Latitude Index')

        # 保存图像
        output_filename = os.path.join(output_dir, f'{product_name}_5year_avg_rainfall.png')
        plt.savefig(output_filename)
        plt.close()
        print(f"成功保存 {product_name} 图像到: {output_filename}")

    except Exception as e:
        print(f"处理 {product_name} 时发生错误: {e}")

print("所有降雨图生成完毕。")
