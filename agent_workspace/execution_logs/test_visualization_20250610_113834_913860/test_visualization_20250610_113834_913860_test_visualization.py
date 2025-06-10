
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 生成一些随机数据
np.random.seed(42)
x = np.random.rand(100) * 10
y = 2 * x + np.random.randn(100) * 5 + 10 # 线性关系加噪声

# 创建散点图
plt.figure(figsize=(8, 6))
plt.scatter(x, y, alpha=0.7, color='blue')
plt.title('随机散点图示例')
plt.xlabel('X轴')
plt.ylabel('Y轴')
plt.grid(True)

# 保存图表到 outputs 目录
output_path = '/sandbox/outputs/visualization_test/scatter_plot_example.png'
plt.savefig(output_path)
print(f"散点图已成功保存到: {output_path}")

# 也可以打印一些简单的统计信息
print(f"X数据的平均值: {np.mean(x):.2f}")
print(f"Y数据的平均值: {np.mean(y):.2f}")
