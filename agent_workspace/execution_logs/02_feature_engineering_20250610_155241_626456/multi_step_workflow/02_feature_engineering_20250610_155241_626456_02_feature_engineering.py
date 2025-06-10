
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import joblib

input_raw_data_path = '/sandbox/outputs/multi_step_test/raw_data.csv'
output_features_dir = '/sandbox/outputs/multi_step_test/'
output_features_path = os.path.join(output_features_dir, 'features.pkl')
output_scaler_path = os.path.join(output_features_dir, 'scaler.pkl')

os.makedirs(output_features_dir, exist_ok=True)

try:
    df = pd.read_csv(input_raw_data_path)
    print(f"成功加载数据: {input_raw_data_path}, 形状: {df.shape}")
except FileNotFoundError:
    print(f"错误: 未找到文件 {input_raw_data_path}。请确保第一步已成功执行并生成了该文件。")
    exit(1)

X = df.drop('target', axis=1)
y = df['target']

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 将处理后的特征和目标变量打包
processed_data = {
    'features': X_scaled,
    'target': y.values,
    'feature_names': X.columns.tolist()
}

# 保存处理后的特征和标准化器
joblib.dump(processed_data, output_features_path)
joblib.dump(scaler, output_scaler_path)

print(f"处理后的特征已保存到: {output_features_path}")
print(f"标准化器已保存到: {output_scaler_path}")
print(f"处理后的特征形状: {X_scaled.shape}")
