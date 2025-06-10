
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import json

input_features_path = '/sandbox/outputs/multi_step_test/features.pkl'
output_model_dir = '/sandbox/outputs/multi_step_test/'
output_model_path = os.path.join(output_model_dir, 'linear_regression_model.pkl')
output_metrics_path = os.path.join(output_model_dir, 'evaluation_metrics.json')

os.makedirs(output_model_dir, exist_ok=True)

try:
    processed_data = joblib.load(input_features_path)
    X = processed_data['features']
    y = processed_data['target']
    feature_names = processed_data['feature_names']
    print(f"成功加载处理后的特征数据，形状: {X.shape}")
except FileNotFoundError:
    print(f"错误: 未找到文件 {input_features_path}。请确保第二步已成功执行并生成了该文件。")
    exit(1)
except Exception as e:
    print(f"加载或解析特征文件时发生错误: {e}")
    exit(1)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
model = LinearRegression()
model.fit(X_train, y_train)

# 进行预测
y_pred = model.predict(X_test)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 保存模型
joblib.dump(model, output_model_path)

# 保存评估指标
metrics = {
    'model_type': 'LinearRegression',
    'mean_squared_error': mse,
    'r2_score': r2,
    'trained_model_path': output_model_path.replace('/sandbox/outputs/', '') # 记录相对路径
}
with open(output_metrics_path, 'w') as f:
    json.dump(metrics, f, indent=4)

print(f"模型已保存到: {output_model_path}")
print(f"评估指标已保存到: {output_metrics_path}")
print(f"均方误差 (MSE): {mse:.4f}")
print(f"R-squared (R2): {r2:.4f}")
