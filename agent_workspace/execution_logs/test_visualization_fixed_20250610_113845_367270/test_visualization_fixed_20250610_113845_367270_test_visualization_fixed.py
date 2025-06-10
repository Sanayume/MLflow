
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# 生成一些随机数据
np.random.seed(42)
x = np.random.rand(100) * 10
y = 2 * x + np.random.randn(100) * 5 + 10 # 线性关系加噪声

# 定义输出目录和文件路径
output_dir = '/sandbox/outputs/visualization_test/'
output_path = os.path.join(output_dir, 'scatter_plot_example.png')

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 创建散点图
plt.figure(figsize=(8, 6))
plt.scatter(x, y, alpha=0.7, color='blue')
plt.title('Random Scatter Plot Example') # 使用英文标题
plt.xlabel('X-axis') # 使用英文标签
plt.ylabel('Y-axis') # 使用英文标签
plt.grid(True)

# 保存图表到 outputs 目录
plt.savefig(output_path)
print(f"散点图已成功保存到: {output_path}")

# 打印一些简单的统计信息
print(f"X data mean: {np.mean(x):.2f}")
print(f"Y data mean: {np.mean(y):.2f}")
