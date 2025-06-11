
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os

# 定义产品列表
products = ['CHIRPS', 'CMORPH', 'GSMAP', 'IMERG', 'PERSIANN', 'sm2rain']
product_filenames = {
    'CHIRPS': 'chirps_2016_2020.mat',
    'CMORPH': 'CMORPH_2016_2020.mat',
    'GSMAP': 'GSMAP_2016_2020.mat',
    'IMERG': 'IMERG_2016_2020.mat',
    'PERSIANN': 'PERSIANN_2016_2020.mat',
    'sm2rain': 'sm2rain_2016_2020.mat'
}

data_dir = '/sandbox/datasets/rainfalldata/'
mask_path = '/sandbox/datasets/mask/mask.mat'
output_dir = '/sandbox/outputs/'

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 加载掩膜
try:
    mask_data = sio.loadmat(mask_path)
    # 假设mask变量名为'data'或'mask'，根据之前探索的结果，选择'data'
    mask = mask_data.get('data') or mask_data.get('mask')
    if mask is None:
        raise ValueError("Mask variable not found in mask.mat. Expected 'data' or 'mask'.")
    print(f"Mask loaded. Shape: {mask.shape}, Unique values: {np.unique(mask)}")
except Exception as e:
    print(f"Error loading mask: {e}")
    exit()

# 创建一个大图，包含所有子图
fig, axes = plt.subplots(2, 3, figsize=(18, 12)) # 2行3列
axes = axes.flatten() # 将axes展平，方便迭代

all_avg_precip_values = [] # 用于收集所有产品的平均降雨值，以便统一颜色条范围

for i, product_name in enumerate(products):
    file_name = product_filenames[product_name]
    file_path = os.path.join(data_dir, product_name + 'data', file_name)
    
    print(f"Processing {product_name} from {file_path}...")
    
    try:
        # 加载.mat文件
        mat_data = sio.loadmat(file_path)
        
        # 假设数据变量名为'data'
        precip_data = mat_data['data']
        
        # 计算五年平均降雨量，跳过NaN值
        # 降雨数据形状是 (lat, lon, time)，对时间维度求平均
        avg_precip = np.nanmean(precip_data, axis=2)
        
        # 应用掩膜：将mask中值为0的区域设置为NaN，以便imshow不显示
        # 假设mask中0代表非中国区域，1和2代表中国区域
        masked_avg_precip = np.where(mask > 0, avg_precip, np.nan)
        
        all_avg_precip_values.append(masked_avg_precip[~np.isnan(masked_avg_precip)])
        
        # 可视化
        ax = axes[i]
        # 使用imshow显示图像，origin='lower'确保图像方向正确，cmap选择合适的颜色图
        # vmin和vmax暂时不设置，让imshow自动调整，如果效果不好再手动统一
        im = ax.imshow(masked_avg_precip, cmap='viridis', origin='lower')
        ax.set_title(f'{product_name} 2016-2020 Average Precipitation')
        ax.axis('off') # 关闭坐标轴
        
        # 添加颜色条
        fig.colorbar(im, ax=ax, shrink=0.7)
        
    except FileNotFoundError:
        print(f"Error: File not found for {product_name} at {file_path}")
        ax = axes[i]
        ax.set_title(f'{product_name} (Data Missing)')
        ax.axis('off')
    except KeyError:
        print(f"Error: 'data' key not found in {file_name} for {product_name}")
        ax = axes[i]
        ax.set_title(f'{product_name} (Data Key Error)')
        ax.axis('off')
    except Exception as e:
        print(f"An unexpected error occurred for {product_name}: {e}")
        ax = axes[i]
        ax.set_title(f'{product_name} (Processing Error)')
        ax.axis('off')

plt.tight_layout() # 调整子图布局，防止重叠

# 保存图片
output_image_path = os.path.join(output_dir, 'average_precipitation_china.png')
plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
print(f"Visualization saved to {output_image_path}")

