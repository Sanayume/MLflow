
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os

# 定义数据路径和产品列表
data_dir = '/sandbox/datasets/rainfalldata/'
mask_path = '/sandbox/datasets/mask/mask.mat'
output_dir = '/sandbox/outputs/'

products = [
    {'name': 'CHIRPS', 'file': 'CHIRPSdata/chirps_2016_2020.mat'},
    {'name': 'CMORPH', 'file': 'CMORPHdata/CMORPH_2016_2020.mat'},
    {'name': 'GSMAP', 'file': 'GSMAPdata/GSMAP_2016_2020.mat'},
    {'name': 'IMERG', 'file': 'IMERGdata/IMERG_2016_2020.mat'},
    {'name': 'PERSIANN', 'file': 'PERSIANNdata/PERSIANN_2016_2020.mat'},
    {'name': 'sm2rain', 'file': 'sm2raindata/sm2rain_2016_2020.mat'} # Assuming this filename
]

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

print(f"Loading mask from: {mask_path}")
try:
    mask_data = scipy.io.loadmat(mask_path)
    mask = mask_data['data'] # Assuming 'data' is the variable name for the mask
    print(f"Mask shape: {mask.shape}, unique values: {np.unique(mask)}")
except Exception as e:
    print(f"Error loading mask: {e}")
    exit()

# 创建2x3的子图布局
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten() # 将axes展平，方便迭代

for i, product_info in enumerate(products):
    product_name = product_info['name']
    file_path = os.path.join(data_dir, product_info['file'])
    
    print(f"Processing {product_name} from: {file_path}")
    try:
        # 加载数据
        data_mat = scipy.io.loadmat(file_path)
        rainfall_data = data_mat['data'] # 假设降雨数据变量名为 'data'
        print(f"{product_name} data shape: {rainfall_data.shape}")

        # 计算5年平均降雨量
        # 忽略NaN值进行平均
        avg_rainfall = np.nanmean(rainfall_data, axis=2)
        print(f"{product_name} average rainfall shape: {avg_rainfall.shape}")

        # 应用掩膜：将掩膜值为0的区域设置为NaN
        # 假设mask中0代表非中国区域，1和2代表中国区域
        masked_avg_rainfall = np.copy(avg_rainfall)
        masked_avg_rainfall[mask == 0] = np.nan 
        
        # 绘制图像
        ax = axes[i]
        cax = ax.imshow(masked_avg_rainfall, cmap='viridis', origin='lower') # origin='lower' 确保图像方向正确
        ax.set_title(f'{product_name} (2016-2020 Average)')
        ax.axis('off') # 关闭坐标轴

        # 添加颜色条
        fig.colorbar(cax, ax=ax, orientation='vertical', shrink=0.7)

    except FileNotFoundError:
        print(f"Error: File not found for {product_name} at {file_path}. Skipping.")
    except Exception as e:
        print(f"Error processing {product_name}: {e}. Skipping.")

plt.suptitle('Average Rainfall (2016-2020) over China Region', fontsize=20)
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 调整布局以避免标题重叠

output_image_path = os.path.join(output_dir, 'average_rainfall_china_2016_2020.png')
plt.savefig(output_image_path, dpi=300)
print(f"Visualization saved to: {output_image_path}")
