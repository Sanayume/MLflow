
import pandas as pd
import numpy as np
import os

# 生成模拟数据
np.random.seed(42)
X = np.random.rand(100, 5) * 100
y = X[:, 0] * 2 + X[:, 1] * 0.5 - X[:, 2] * 1.5 + np.random.randn(100) * 5 + 10

df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
df['target'] = y

output_dir = '/sandbox/outputs/multi_step_test/'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'raw_data.csv')

df.to_csv(output_path, index=False)
print(f"原始数据已保存到: {output_path}")
print(f"数据集形状: {df.shape}")
print(f"目标变量平均值: {df['target'].mean():.2f}")
